"""Bounded, inspectable selection of exact floating-point readout algorithms."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .backends import BackendCapabilities
from .compiler import _check_circuit_limits
from .compute import memory_budget
from .errors import ConversionError, ResourceError, ValidationError
from .ir import integer
from .parameters import real
from .semantics import _CLIFFORD


def _available_memory():
    """Observe host availability and the remaining container limit when present."""
    import psutil

    available = int(psutil.virtual_memory().available)
    for limit_path, used_path in (
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        (
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        ),
    ):
        try:
            limit = int(Path(limit_path).read_text().strip())
            used = int(Path(used_path).read_text().strip())
        except (OSError, ValueError):
            continue
        if 0 < limit < 1 << 60:
            available = min(available, max(0, limit - used))
    return available


def _budget(value):
    if value is None:
        available = _available_memory()
        budget = min(4096, available // (4 * 2**20))
        if budget < 1:
            raise ResourceError("Available memory cannot support the automatic working budget.")
        return float(budget), "quarter_available_host_or_container_memory_capped_at_4_GiB"
    value = real(value)
    if value <= 0:
        raise ValidationError("max_memory_mb must be positive.")
    return value, "explicit"


def _fits(exponent, budget, coefficient=1):
    numerator, denominator = budget.as_integer_ratio()
    capacity = (numerator << 20) // denominator
    return coefficient <= capacity >> exponent


def _dense_bytes(width, density=False):
    exponent = (2 if density else 1) * width + 6
    return (1 << exponent) if exponent < 63 else None


def _gpu_status():
    """Check the installed native tensor runtime; a CPU wheel needs no GPU extra."""
    from .adapters._common import dependency

    torch = dependency("torch", "torch")
    if not torch.cuda.is_available():
        return False, 0
    try:
        free, _ = torch.cuda.mem_get_info()
    except (RuntimeError, AssertionError):
        return False, 0
    return True, int(free)


@dataclass(frozen=True)
class ExecutionCandidate:
    method: str
    eligible: bool
    reason: str
    working_bytes: int | None = None


@dataclass(frozen=True)
class AdaptivePlan:
    circuit: object = field(repr=False)
    observables: tuple = field(repr=False)
    selected_method: str | None
    max_memory_mb: float
    memory_policy: str
    max_qubits: int | None
    max_branches: int
    candidates: tuple[ExecutionCandidate, ...]
    mps_profile: dict | None
    _pauli: object = field(default=None, repr=False, compare=False)
    _runner: object = field(default=None, repr=False, compare=False)
    dense_device: str = "CPU"
    _components: object = field(default=None, repr=False, compare=False)
    _tensor: object = field(default=None, repr=False, compare=False)

    @property
    def supported(self):
        return self.selected_method is not None

    def to_dict(self):
        return {
            "scope": "declared_expectations_from_zero_state",
            "selected_method": self.selected_method,
            "device": None
            if not self.supported
            else "GPU"
            if self.selected_method == "statevector-gpu"
            else "CPU",
            "requested_dense_device": self.dense_device,
            "supported": self.supported,
            "num_qubits": self.circuit.num_qubits,
            "max_qubits": self.max_qubits,
            "max_memory_mb": self.max_memory_mb,
            "memory_policy": self.memory_policy,
            "exactness": "no_intentional_truncation_or_shot_approximation; "
            "floating_point_roundoff_remains",
            "memory_guarantee": "representation_admission_model_not_process_RSS_limit",
            "mps_profile": self.mps_profile,
            "component_profile": None if self._components is None else self._components.to_dict(),
            "tensor_profile": None if self._tensor is None else self._tensor.to_dict(),
            "candidates": [asdict(candidate) for candidate in self.candidates],
        }

    def to_json(self, *, indent=2):
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    def raise_for_errors(self):
        if not self.supported:
            details = "; ".join(f"{c.method}: {c.reason}" for c in self.candidates)
            raise ResourceError(f"No admitted exact execution method. {details}")

    def evaluate(self, bindings=None, *, backend="numpy", like=None, batch_shape=None):
        self.raise_for_errors()
        if backend not in {"numpy", "torch", "jax"}:
            raise ValidationError("Evaluation backend must be numpy, torch or jax.")
        if self.selected_method == "pauli":
            return self._pauli.evaluate(
                bindings, backend=backend, like=like, batch_shape=batch_shape
            )
        if self.selected_method == "components":
            return self._components.evaluate(
                bindings, backend=backend, like=like, batch_shape=batch_shape
            )
        if self.selected_method == "tensor-network":
            return self._tensor.evaluate(
                bindings, backend=backend, like=like, batch_shape=batch_shape
            )
        if self.selected_method == "density":
            from .density import _expectations_from_density, density_matrix
            from .simulator import _TensorOps

            rho = density_matrix(
                self.circuit,
                bindings,
                backend=backend,
                like=like,
                max_qubits=self.circuit.num_qubits,
                max_memory_mb=self.max_memory_mb,
                max_branches=self.max_branches,
                batch_shape=batch_shape,
            )
            ops = _TensorOps(backend, rho.real if like is None else like)
            return _expectations_from_density(rho, self.observables, self.circuit.num_qubits, ops)
        if self.selected_method == "statevector":
            from .simulator import expectations

            return expectations(
                self.circuit,
                self.observables,
                bindings,
                backend=backend,
                like=like,
                max_qubits=self.circuit.num_qubits if self.max_qubits is None else self.max_qubits,
                max_memory_mb=self.max_memory_mb,
                max_branches=self.max_branches,
                batch_shape=batch_shape,
            )
        if self.selected_method == "statevector-gpu":
            import torch

            from .simulator import expectations

            if backend not in {"numpy", "torch"}:
                raise ValidationError(
                    "CUDA dense plans support numpy output or native torch autodiff."
                )
            if like is None:
                like = torch.empty((), dtype=torch.float64, device="cuda")
            if not isinstance(like, torch.Tensor) or like.device.type != "cuda":
                raise ValidationError("CUDA plans require a CUDA tensor context.")
            available, free = _gpu_status()
            if not available or _dense_bytes(self.circuit.num_qubits) > free // 2:
                raise ResourceError(
                    "Free CUDA memory changed after planning; re-plan the execution."
                )
            values = expectations(
                self.circuit,
                self.observables,
                bindings,
                backend="torch",
                like=like,
                max_qubits=self.circuit.num_qubits,
                max_memory_mb=self.max_memory_mb,
                batch_shape=batch_shape,
            )
            return values if backend == "torch" else values.detach().cpu().numpy()
        if backend != "numpy" or like is not None:
            raise ValidationError(
                "The selected MPS/stabilizer executor returns numerical readouts; "
                "use a numerical gradient bridge, or choose pauli/statevector for native autodiff."
            )
        from .batch import expectations_batch

        return expectations_batch(
            self.circuit,
            self.observables,
            [bindings or {}],
            backend=self._runner,
            max_qubits=self.max_qubits,
            max_memory_mb=self.max_memory_mb,
            max_branches=self.max_branches,
        )[0]


def plan_execution(
    circuit,
    observables,
    *,
    method="auto",
    max_memory_mb=None,
    max_qubits=None,
    max_branches=256,
    max_operations=100_000,
    max_work=1_000_000,
    max_pauli_terms=4096,
    max_bond_dimension=256,
    device="CPU",
):
    """Plan before allocating quantum states; selection is inspectable and bounded.

    Automatic memory uses one quarter of observed host/container availability,
    capped at 4 GiB. No fixed source-width ceiling applies unless requested.
    MPS admission uses structural upper bounds and disables truncation. It can
    reject circuits whose actual bond ranks would be lower. Selection never
    retries an execution failure with an unbudgeted algorithm.
    """
    from .batch import _observables
    from .pauli import PauliPlan
    from .planning import plan_density

    allowed = {
        "auto",
        "pauli",
        "stabilizer",
        "mps",
        "statevector",
        "density",
        "components",
        "tensor-network",
    }
    if method not in allowed or device not in {"CPU", "GPU", "auto"}:
        raise ValidationError("Unknown execution method or device (CPU, GPU, auto).")
    _check_circuit_limits(circuit, max_operations=max_operations)
    integer(max_work, "max_work", 1)
    integer(max_pauli_terms, "max_pauli_terms", 1)
    integer(max_bond_dimension, "max_bond_dimension", 1)
    integer(max_branches, "max_branches", 1)
    if max_qubits is not None:
        integer(max_qubits, "max_qubits", 1)
    observables = _observables(observables, [circuit])
    from .observables import Hamiltonian

    terms = sum(
        len(observable.terms) if isinstance(observable, Hamiltonian) else 1
        for observable in observables
    )
    if terms * (1 + circuit.num_qubits) > max_work:
        raise ResourceError("Readout descriptions exceed max_work.")
    explicit_width = max_qubits is not None and circuit.num_qubits > max_qubits
    budget, policy = (
        (0.0, "not_queried_after_explicit_width_rejection")
        if explicit_width and max_memory_mb is None
        else _budget(max_memory_mb)
    )
    candidates, selected, pauli, runner, mps_profile = [], None, None, None, None
    components = None
    tensor = None
    deferred_stabilizer = False
    if explicit_width:
        candidates.append(
            ExecutionCandidate(method, False, "Source exceeds the explicit max_qubits limit.")
        )
    else:
        unitary = not circuit.has_effects
        clifford = unitary and all(op.name in _CLIFFORD | {"barrier"} for op in circuit.operations)
        if method in {"auto", "stabilizer"}:
            from importlib.util import find_spec

            try:
                installed = find_spec("qiskit_aer") is not None
            except ImportError:
                installed = False
            # A packed tableau needs O(n^2) bits. Reserve a deliberately larger
            # byte-per-entry workspace model for native snapshots and readouts.
            workspace = (
                16 * circuit.num_qubits * circuit.num_qubits
                + 128 * len(observables) * circuit.num_qubits
            )
            eligible = clifford and workspace <= budget * 2**20 and installed
            candidates.append(
                ExecutionCandidate(
                    "stabilizer",
                    eligible,
                    "Named Clifford gates and bounded tableau workspace."
                    if eligible
                    else "Requires named Clifford gates, the aer extra "
                    "and sufficient tableau workspace.",
                    workspace,
                )
            )
            if eligible:
                if method == "stabilizer" or circuit.num_qubits >= 64 or len(observables) >= 16:
                    selected = "stabilizer"
                else:
                    # Avoid importing/compiling an SDK for tiny Clifford
                    # queries that the local symbolic engine can answer.
                    deferred_stabilizer = True
        if selected is None and method in {"auto", "pauli"}:
            try:
                if not unitary:
                    raise ResourceError("Pauli propagation requires a unitary circuit.")
                numerator, denominator = budget.as_integer_ratio()
                available_bytes = (numerator << 20) // denominator
                nodes = min(max_work, 100_000, available_bytes // 1024)
                term_limit = min(
                    max_pauli_terms,
                    available_bytes // (2 * (512 + 2 * ((circuit.num_qubits + 7) // 8))),
                )
                if nodes < 2 or term_limit < 1:
                    raise ResourceError("Pauli graph/frontier bookkeeping exceeds the budget.")
                pauli = PauliPlan(
                    circuit,
                    observables,
                    max_terms=term_limit,
                    max_nodes=nodes,
                    max_work=max_work,
                    max_operations=max_operations,
                    max_memory_mb=budget,
                )
                selected = "pauli"
                candidates.append(
                    ExecutionCandidate(
                        "pauli", True, "Bounded exact symbolic Pauli propagation completed."
                    )
                )
            except (ConversionError, ResourceError) as error:
                candidates.append(ExecutionCandidate("pauli", False, str(error)))
        if selected is None and deferred_stabilizer:
            selected = "stabilizer"
        if selected is None and method in {"auto", "mps"}:
            from ._mps import _layout

            mps_profile, reason = (
                _layout(circuit, max_bond_dimension, max_work)
                if unitary
                else (None, "MPS admission currently requires a unitary circuit.")
            )
            eligible = bool(
                mps_profile
                and mps_profile["analysis_complete"]
                and mps_profile["max_bond"] <= max_bond_dimension
                and mps_profile["workspace_bytes"] <= budget * 2**20
            )
            candidates.append(
                ExecutionCandidate(
                    "mps",
                    eligible,
                    reason
                    if eligible or mps_profile is None
                    else "Full-SVD MPS bond/workspace bound exceeds policy.",
                    None if mps_profile is None else mps_profile["workspace_bytes"],
                )
            )
            if eligible:
                selected = "mps"
        if selected is None and method in {"auto", "components"}:
            from .components import ComponentPlan

            try:
                components = ComponentPlan(
                    circuit,
                    observables,
                    max_qubits=max_qubits,
                    max_memory_mb=budget,
                    max_branches=max_branches,
                    max_work=max_work,
                    max_operations=max_operations,
                )
                useful = (
                    len(components.components) != 1
                    or len(components.components[0][0]) < circuit.num_qubits
                )
                eligible = method == "components" or useful
                candidates.append(
                    ExecutionCandidate(
                        "components",
                        eligible,
                        "Independent quantum/classical factors fit the memory budget."
                        if eligible
                        else "No smaller dependency factor; prefer the unfactored executor.",
                        components.working_bytes,
                    )
                )
                if eligible:
                    selected = "components"
            except ResourceError as error:
                candidates.append(ExecutionCandidate("components", False, str(error)))
        if selected is None and method in {"auto", "tensor-network"} and unitary:
            from ._tensor_contraction import TensorNetworkPlan

            try:
                tensor = TensorNetworkPlan(
                    circuit,
                    observables,
                    max_memory_mb=budget,
                    max_work=max_work,
                    max_operations=max_operations,
                )
                dense_size = _dense_bytes(circuit.num_qubits)
                preferred = (
                    method == "tensor-network"
                    or dense_size is None
                    or tensor.working_bytes < dense_size
                )
                if preferred:
                    selected = "tensor-network"
                candidates.append(
                    ExecutionCandidate(
                        "tensor-network",
                        preferred,
                        "An exact contraction path fits the slicing and memory budgets."
                        if preferred
                        else "A path fits, but the dense working model is smaller.",
                        tensor.working_bytes,
                    )
                )
            except ResourceError as error:
                candidates.append(ExecutionCandidate("tensor-network", False, str(error)))
        if selected is None and method in {"auto", "statevector", "density"}:
            density = not unitary or method == "density"
            chosen = "density" if density else "statevector"
            fits = _fits((2 if density else 1) * circuit.num_qubits + 6, budget)
            reason = "Four-buffer dense working model fits the memory budget."
            if fits and density:
                try:
                    records = plan_density(circuit, max_branches=max_branches)
                    from .density import _resource_check

                    _resource_check(circuit, max_qubits, budget, branches=records.peak_branches)
                except (ResourceError, ConversionError) as error:
                    fits, reason = False, str(error)
            candidates.append(
                ExecutionCandidate(
                    chosen,
                    fits,
                    reason
                    if fits
                    else "Dense storage or classical branches exceed the budget. " + reason,
                    _dense_bytes(circuit.num_qubits, density),
                )
            )
            if fits:
                selected = chosen
                if not density and device in {"GPU", "auto"}:
                    available, free = _gpu_status()
                    needed = _dense_bytes(circuit.num_qubits)
                    gpu_ok = available and needed is not None and needed <= free // 2
                    candidates.append(
                        ExecutionCandidate(
                            "statevector-gpu",
                            gpu_ok,
                            "CUDA is available; buffers fit half of free device memory."
                            if gpu_ok
                            else "CUDA unavailable or insufficient free device memory.",
                            needed,
                        )
                    )
                    if gpu_ok:
                        selected = "statevector-gpu"
                    elif device == "GPU":
                        selected = None
        if selected == "stabilizer":
            from .aer import AerBackend

            runner = AerBackend(
                name="adaptive-" + selected,
                method="stabilizer",
                parameter_batching=True,
                mps_truncation_threshold=0.0,
            )
        elif selected == "mps":
            from ._mps import MPSBackend

            runner = MPSBackend(
                max_bond_dimension=max_bond_dimension,
                max_operations=max_operations,
                max_work=max_work,
            )
    return AdaptivePlan(
        circuit,
        observables,
        selected,
        budget,
        policy,
        max_qubits,
        max_branches,
        tuple(candidates),
        mps_profile,
        pauli,
        runner,
        device,
        components,
        tensor,
    )


@dataclass(frozen=True)
class AdaptiveBackend:
    """Numeric adaptive readouts through backend='auto'; no hidden approximation."""

    name: str = "auto"
    capabilities: BackendCapabilities = BackendCapabilities(
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
    )
    _cache: object = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        from ._parameter_batches import CircuitCache

        object.__setattr__(self, "_cache", CircuitCache(16))

    def expectations_batch(self, circuits, observables, *, options):
        from ._parameter_batches import parameter_batches

        output = np.empty((len(circuits), len(observables)))
        for rows, prototype, bindings in parameter_batches(circuits, options.max_memory_mb):
            key = (
                prototype,
                tuple(observables),
                options.max_memory_mb,
                options.max_qubits,
                options.max_branches,
            )
            plan = self._cache.get(
                key,
                lambda prototype=prototype: plan_execution(
                    prototype,
                    observables,
                    max_memory_mb=options.max_memory_mb,
                    max_qubits=options.max_qubits,
                    max_branches=options.max_branches,
                ),
            )
            if plan.selected_method in {
                "pauli",
                "components",
                "density",
                "statevector",
                "tensor-network",
            }:
                # Bound each chunk by the representation's per-row admission
                # model; avoid retaining a full trajectory's quantum states.
                representation = (
                    plan._pauli
                    if plan.selected_method == "pauli"
                    else plan._tensor
                    if plan.selected_method == "tensor-network"
                    else plan._components
                )
                if representation is not None:
                    per_row = getattr(representation, "working_bytes", None)
                    if per_row is None:
                        per_row = 8 * (len(representation._nodes) + 8)
                else:
                    per_row = 64 * (
                        1 << ((2 if prototype.has_effects else 1) * prototype.num_qubits)
                    )
                per_row += 8 * len(bindings) + 64 * len(observables)
                chunk = max(
                    1, int(memory_budget(options.max_memory_mb) * 2**20) // (2 * max(1, per_row))
                )
                for start in range(0, len(rows), chunk):
                    selected = rows[start : start + chunk]
                    parameters = {k: v[start : start + chunk] for k, v in bindings.items()}
                    if representation is not None:
                        result = representation.evaluate(parameters, batch_shape=(len(selected),))
                    else:
                        result = plan.evaluate(parameters, batch_shape=(len(selected),))
                        if not bindings:
                            result = np.broadcast_to(result, (len(selected), len(observables)))
                    output[list(selected)] = result
            else:
                # MPS/tableau providers own their numeric batching contract.
                from .batch import expectations_batch

                output[list(rows)] = expectations_batch(
                    [circuits[i] for i in rows],
                    observables,
                    backend=plan._runner,
                    **options.__dict__,
                )
        return output

    def run(self, circuit, *, options):
        from .simulator import statevector

        # A full-state request necessarily has exponential output size. It
        # cannot inherit the capacity of a structured readout representation.
        return statevector(
            circuit,
            backend="numpy",
            max_qubits=options.max_qubits,
            max_memory_mb=options.max_memory_mb,
        )

    def run_density(self, circuit, *, options):
        from .density import density_matrix

        return density_matrix(
            circuit,
            backend="numpy",
            max_qubits=options.max_qubits,
            max_memory_mb=options.max_memory_mb,
            max_branches=options.max_branches,
        )

    def sample_batch(self, circuits, shots, *, options):
        from .batch import execute_batch

        return [
            r.counts
            for r in execute_batch(circuits, backend="numpy", shots=shots, **options.__dict__)
        ]
