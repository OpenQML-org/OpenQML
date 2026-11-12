"""Lazy CUDA readouts with shared topologies and bounded native tensor batches."""

import re
from dataclasses import dataclass, field, replace
from importlib.util import find_spec

import numpy as np

from ..compute.kernel import memory_budget
from ..errors import ConversionError, ResourceError, ValidationError, _require_choice
from ..interop._parameter_batches import CircuitCache, binding_matrix, parameter_batches
from ..runtime.backends import BackendCapabilities


def _native_parameters(bindings, start, stop, like):
    """Stage one packed host buffer per chunk instead of one transfer per gate."""
    if not bindings:
        return {}
    import torch

    host = torch.from_numpy(np.ascontiguousarray(binding_matrix(bindings)[start:stop]))
    packed = host.to(device=like.device, dtype=like.dtype, non_blocking=True)
    return {name: packed[:, i] for i, name in enumerate(bindings)}


@dataclass(frozen=True)
class CUDABackend:
    name: str = "gpu"
    precision: str = "float64"
    device: str | None = None
    capabilities: BackendCapabilities = field(
        default_factory=lambda: BackendCapabilities(
            native_expectations=True,
            native_sampling=True,
            density_matrix=True,
            channels=True,
            reset=True,
            classical_control=True,
            mid_circuit_measurements=True,
            measurement_expectations=True,
            deterministic_expectations=True,
            exact_expectations=True,
            device_type="GPU",
        )
    )
    _cache: object = field(default_factory=lambda: CircuitCache(16), repr=False, compare=False)

    def __post_init__(self):
        _require_choice(self.precision, ("float32", "float64"), "precision")
        if self.device is not None and (
            not isinstance(self.device, str)
            or re.fullmatch(r"cuda(?::[0-9]+)?", self.device) is None
        ):
            raise ValidationError("device must be None, 'cuda' or 'cuda:<index>'.")

    def _like(self):
        import torch

        return torch.empty((), device=self.device or "cuda", dtype=getattr(torch, self.precision))

    def is_available(self):
        try:
            if find_spec("torch") is None:
                return False
            import torch
        except (ImportError, OSError):
            return False

        return bool(torch.cuda.is_available()) and (
            self.device is None
            or ":" not in self.device
            or int(self.device.split(":")[1]) < torch.cuda.device_count()
        )

    def _options(self, options):
        if not self.is_available():
            raise ResourceError("CUDA is unavailable; install a CUDA-enabled Torch runtime.")
        import torch

        try:
            free, _ = torch.cuda.mem_get_info(self.device)
        except (RuntimeError, AssertionError) as exc:
            raise ResourceError("CUDA memory admission is unavailable.") from exc
        budget = min(options.max_memory_mb, free / (2 * 2**20))
        if budget <= 0:
            raise ResourceError("No free CUDA working memory.")
        return replace(options, max_memory_mb=budget)

    def _readout(self, prototype, observables, options):
        from .components import ComponentPlan
        from .pauli import PauliPlan

        key = (
            prototype,
            tuple(observables),
            options.max_memory_mb,
            options.max_qubits,
            options.max_branches,
        )

        def compile_readout():
            if not prototype.has_effects:
                try:
                    return PauliPlan(prototype, observables, max_memory_mb=options.max_memory_mb)
                except (ResourceError, ConversionError):
                    pass
            try:
                return ComponentPlan(
                    prototype,
                    observables,
                    max_qubits=options.max_qubits,
                    max_memory_mb=options.max_memory_mb,
                    max_branches=options.max_branches,
                )
            except ResourceError:
                if prototype.has_effects:
                    raise
                from ..compute._contraction import TensorNetworkPlan

                return TensorNetworkPlan(
                    prototype, observables, max_memory_mb=options.max_memory_mb
                )

        return self._cache.get(key, compile_readout)

    def admit(self, task, circuits, observables, *, options):
        options = self._options(options)
        if task == "expectations":
            for _, prototype, _ in parameter_batches(circuits, options.max_memory_mb):
                self._readout(prototype, observables, options)
            return
        from .dense import _resource_check

        for circuit in circuits:
            if task == "density":
                from .density import _resource_check as density_check
                from .planning import plan_density

                branches = plan_density(circuit, max_branches=options.max_branches).peak_branches
                density_check(circuit, options.max_qubits, options.max_memory_mb, branches=branches)
            else:
                # This sampler handles terminal measurements; exact feedback
                # sampling is delegated to the CPU branch-ensemble engine.
                pure = circuit.unitary_part() if task == "sampling" else circuit
                _resource_check(pure, options.max_qubits, options.max_memory_mb)

    def expectations_batch(self, circuits, observables, *, options):
        import torch

        options = self._options(options)
        if 8 * len(circuits) * len(observables) > memory_budget(options.max_memory_mb) * 2**20:
            raise ResourceError("GPU readout output exceeds max_memory_mb.")
        like = self._like()
        output = np.empty((len(circuits), len(observables)))
        with torch.no_grad():
            for rows, prototype, bindings in parameter_batches(circuits, options.max_memory_mb):
                plan = self._readout(prototype, observables, options)
                per_row = getattr(plan, "working_bytes", None)
                if per_row is None:
                    per_row = 8 * (len(plan._nodes) + 8)
                per_row += 8 * len(bindings) + 64 * len(observables)
                chunk = max(
                    1, int(memory_budget(options.max_memory_mb) * 2**20) // (2 * max(1, per_row))
                )
                for start in range(0, len(rows), chunk):
                    selected = rows[start : start + chunk]
                    parameters = _native_parameters(bindings, start, start + chunk, like)
                    values = plan.evaluate(
                        parameters, backend="torch", like=like, batch_shape=(len(selected),)
                    )
                    output[list(selected)] = values.cpu().numpy()
        return output

    def run(self, circuit, *, options):
        import torch

        from .dense import statevector

        options = self._options(options)
        with torch.no_grad():
            state = statevector(
                circuit,
                backend="torch",
                like=self._like(),
                max_qubits=options.max_qubits,
                max_memory_mb=options.max_memory_mb,
            )
        return state.cpu().numpy()

    def run_batch(self, circuits, *, options):
        import torch

        from .batch import _vectorized_states

        options = self._options(options)
        width = max(c.num_qubits for c in circuits)
        chunk = max(1, int(memory_budget(options.max_memory_mb) * 2**20) // (64 * (1 << width)))
        output = []
        like = self._like()
        with torch.no_grad():
            for start in range(0, len(circuits), chunk):
                states = _vectorized_states(
                    circuits[start : start + chunk], options, backend="torch", like=like
                )
                if isinstance(states, torch.Tensor):
                    output.extend(states.cpu().numpy())
                else:
                    sizes = [state.numel() for state in states]
                    packed = torch.cat([state.reshape(-1) for state in states]).cpu().numpy()
                    offset = 0
                    for state, size in zip(states, sizes, strict=True):
                        output.append(packed[offset : offset + size].reshape(state.shape))
                        offset += size
        return output

    def run_density(self, circuit, *, options):
        import torch

        from .density import density_matrix

        options = self._options(options)
        with torch.no_grad():
            rho = density_matrix(
                circuit,
                backend="torch",
                like=self._like(),
                max_qubits=options.max_qubits,
                max_memory_mb=options.max_memory_mb,
                max_branches=options.max_branches,
            )
        return rho.cpu().numpy()

    def sample_batch(self, circuits, shots, *, options):
        import torch

        from .dense import statevector

        options = self._options(options)
        self.admit("sampling", circuits, (), options=options)
        like = self._like()
        generator = torch.Generator(device=like.device)
        generator.seed() if options.seed is None else generator.manual_seed(options.seed)
        output = [None] * len(circuits)
        pure = [c.unitary_part() for c in circuits]
        with torch.no_grad():
            for rows, prototype, bindings in parameter_batches(pure, options.max_memory_mb):
                width = max(
                    circuits[r].num_clbits
                    if circuits[r].has_measurements
                    else circuits[r].num_qubits
                    for r in rows
                )
                if width * shots > memory_budget(options.max_memory_mb) * 2**20:
                    raise ResourceError("Sampled bit strings exceed max_memory_mb.")
                states = 1 << prototype.num_qubits
                per_row = 64 * states + (8 + width) * shots + 8 * len(bindings)
                chunk = max(1, int(memory_budget(options.max_memory_mb) * 2**20) // per_row)
                for start in range(0, len(rows), chunk):
                    selected = rows[start : start + chunk]
                    parameters = _native_parameters(bindings, start, start + chunk, like)
                    state = statevector(
                        prototype,
                        parameters,
                        backend="torch",
                        like=like,
                        max_qubits=options.max_qubits,
                        max_memory_mb=options.max_memory_mb,
                        batch_shape=(len(selected),),
                    )
                    indices = torch.multinomial(
                        state.abs().square(), shots, replacement=True, generator=generator
                    )
                    encoded = (
                        indices + states * torch.arange(len(selected), device=like.device)[:, None]
                    )
                    unique, counts = torch.unique(encoded, return_counts=True)
                    for row in selected:
                        output[row] = {}
                    for index, count in torch.stack((unique, counts), dim=-1).cpu().tolist():
                        local, index = divmod(index, states)
                        row = selected[local]
                        circuit = circuits[row]
                        bits = format(index, f"0{circuit.num_qubits}b")
                        if circuit.has_measurements:
                            measured = ["0"] * circuit.num_clbits
                            for op in circuit.operations:
                                if op.name == "measure":
                                    measured[op.clbits[0]] = bits[op.qubits[0]]
                            bits = "".join(measured)
                        output[row][bits] = output[row].get(bits, 0) + count
        return output
