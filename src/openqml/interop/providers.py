"""Adapters for caller-provided execution engines."""

from dataclasses import dataclass, field, replace
from threading import RLock

import numpy as np

from ..compute.kernel import get_compute_kernel
from ..errors import BackendError, ResourceError, ValidationError
from ..ir.observables import Hamiltonian
from ..runtime.backends import BackendCapabilities
from ..runtime.recovery import BackendJobError, PendingJob, RecoveryPolicy
from ._parameter_batches import CircuitCache, parameter_batches


def _layout(metadata):
    from ..ir.core import integer

    rows, columns = (integer(n, "provider result dimension") for n in metadata["shape"])
    get_compute_kernel().admit(rows * max(1, columns), metadata["max_memory_mb"])
    layouts = tuple(tuple(integer(i, "provider row") for i in row) for row in metadata["layouts"])
    if sorted(i for row in layouts for i in row) != list(range(rows)):
        raise BackendError("Provider receipt has invalid row coverage.")
    return rows, columns, layouts


def _estimator_result(results, metadata):
    rows, columns, layouts = _layout(metadata)
    if len(results) != len(layouts):
        raise BackendError("Estimator returned an unexpected number of PUB results.")
    output = np.empty((rows, columns), dtype=float)
    for indices, pub in zip(layouts, results, strict=True):
        data = np.asarray(pub.data.evs)
        if data.shape != (columns, len(indices)) or data.dtype.kind not in "fiu":
            raise BackendError("Estimator returned an unexpected PUB broadcast shape or dtype.")
        output[list(indices)] = data.T
    bounds = np.asarray(metadata["bounds"])
    policy = get_compute_kernel().policy
    if not np.isfinite(output).all() or (
        metadata.get("bounded_expectations", True)
        and np.any(
            np.abs(output) > bounds + policy.validation_atol + policy.validation_rtol * bounds
        )
    ):
        raise BackendError("Estimator returned non-finite or out-of-range expectations.")
    return output


def _sampler_result(results, metadata):
    from ..ir.core import integer

    rows, _, layouts = _layout(metadata)
    if len(results) != len(layouts):
        raise BackendError("Sampler returned an unexpected number of PUB results.")
    output = [None] * rows
    for indices, pub in zip(layouts, results, strict=True):
        if pub.data.c.shape != (len(indices),):
            raise BackendError("Sampler returned an unexpected PUB broadcast shape.")
        for i, row in enumerate(indices):
            counts = pub.data.c.get_counts((i,))
            width = metadata["widths"][row]
            if any(
                not isinstance(key, str) or len(key) != width or set(key) - {"0", "1"}
                for key in counts
            ):
                raise BackendError("Sampler returned invalid classical bit strings.")
            counts = {key[::-1]: integer(n, "provider count") for key, n in counts.items()}
            if sum(counts.values()) != metadata["shots"]:
                raise BackendError("Sampler counts do not sum to the submitted shots.")
            output[row] = counts
    return output


def recover_job(receipt, loader, *, recovery=None):
    """Retrieve a recorded job by ID and decode it without submitting new work."""
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema") != "openqml.provider_job.v1"
        or not isinstance(receipt.get("job_id"), str)
        or not receipt["job_id"]
    ):
        raise ValidationError("Provide a provider job receipt with a job ID.")
    metadata = receipt.get("metadata", {})
    decoders = {"estimator-v2": _estimator_result, "sampler-v2": _sampler_result}
    if not isinstance(metadata, dict) or metadata.get("kind") not in decoders:
        raise ValidationError("Unknown provider result decoder.")
    try:
        rows, columns, _ = _layout(metadata)
        if metadata["kind"] == "estimator-v2":
            if type(metadata.get("bounded_expectations", True)) is not bool:
                raise ValueError("Invalid estimator bounds contract.")
            bounds = np.asarray(metadata["bounds"])
            if bounds.shape != (columns,) or np.any(np.isnan(bounds) | (bounds < 0)):
                raise ValueError("Invalid observable bounds.")
        else:
            from ..ir.core import integer

            integer(metadata["shots"], "receipt shots", 1)
            if len(metadata["widths"]) != rows:
                raise ValueError("Invalid classical widths.")
            for width in metadata["widths"]:
                integer(width, "receipt classical width")
    except (KeyError, TypeError, ValueError, OverflowError, BackendError) as exc:
        raise ValidationError("Invalid provider job receipt metadata.") from exc
    pending = PendingJob(
        loader(receipt["job_id"]),
        decoders[metadata["kind"]],
        backend=receipt["backend"],
        metadata=metadata,
        policy=recovery,
    )
    if pending.job_id is not None and pending.job_id != receipt["job_id"]:
        raise ValidationError("Retrieved job ID differs from the retained receipt.")
    return pending.result()


def _provider_job(submit, decoder, metadata, backend, recovery):
    try:
        job = submit()
    except Exception as exc:
        # Submission state is unknown: never replay a potentially charged job.
        raise BackendJobError(
            f"Provider {backend!r} submission failed; "
            f"reconcile provider status before resubmitting: {exc}",
            stage="submission",
        ) from exc
    return PendingJob(job, decoder, backend=backend, metadata=metadata, policy=recovery).result()


@dataclass(frozen=True)
class QiskitEstimatorBackend:
    """Adapt a supplied BaseEstimatorV2 with observable layout mapping for ISA circuits.

    pass_manager, when supplied, must expose run(circuit). Each public V2 result
    contains one expectation per requested observable. Provider job management,
    precision, credentials and hardware access remain with the supplied object.
    Set mitigated=True when it returns error-mitigated statistical estimates:
    these may lie outside spectral bounds and are never silently clipped.
    """

    estimator: object
    name: str = "qiskit-estimator-v2"
    precision: float | None = None
    pass_manager: object = None
    cache_size: int = 16
    recovery: RecoveryPolicy | None = None
    _cache: object = field(init=False, repr=False, compare=False)
    _transpiler_lock: object = field(init=False, repr=False, compare=False)
    capabilities: BackendCapabilities = field(
        default_factory=lambda: BackendCapabilities(native_expectations=True)
    )
    mitigated: bool = False

    def __post_init__(self):
        if type(self.mitigated) is not bool:
            raise ValidationError("mitigated must be a boolean.")
        if self.mitigated:
            object.__setattr__(
                self, "capabilities", replace(self.capabilities, bounded_expectations=False)
            )
        object.__setattr__(self, "_cache", CircuitCache(self.cache_size))
        object.__setattr__(self, "_transpiler_lock", RLock())
        if self.recovery is not None and not isinstance(self.recovery, RecoveryPolicy):
            raise ValidationError("recovery must be RecoveryPolicy.")
        if not callable(getattr(self.estimator, "run", None)):
            raise ValidationError("Provide an initialized Qiskit V2 estimator.")
        if self.precision is not None:
            from ..ir.parameters import real

            if real(self.precision) < 0:
                raise ValidationError("Estimator precision must be nonnegative.")

    def clear_cache(self):
        """Invalidate ISA circuits after changing a caller-owned pass manager/target."""
        self._cache.clear()

    def with_shots(self, shots):
        """Request equivalent statistical precision, not a provider billing cap.

        V2 estimators accept precision; providers may allocate additional shots
        for randomization/mitigation. Quote callbacks must include that overhead.
        """
        from ..ir.core import integer

        shots = integer(shots, "shots", 1)
        configured = replace(self, precision=shots**-0.5)
        object.__setattr__(configured, "_cache", self._cache)
        object.__setattr__(configured, "_transpiler_lock", self._transpiler_lock)
        return configured

    def expectations_batch(self, circuits, observables, *, options):
        from .adapters._common import dependency

        dependency("qiskit", "qiskit")
        from qiskit.quantum_info import SparsePauliOp

        from .adapters.qiskit import to_qiskit

        operators = []
        for observable in observables:
            terms = observable.terms if isinstance(observable, Hamiltonian) else (observable,)
            operators.append(
                SparsePauliOp.from_list([(t.paulis[::-1], t.coefficient) for t in terms])
            )
        pubs, layouts = [], []
        for rows, prototype, bindings in parameter_batches(circuits, options.max_memory_mb):

            def compile_circuit(prototype=prototype):
                native = to_qiskit(prototype)
                if self.pass_manager is None:
                    return native
                with self._transpiler_lock:
                    return self.pass_manager.run(native)

            native = self._cache.get(prototype, compile_circuit)
            mapped = operators
            if self.pass_manager is not None:
                if native.layout is None:
                    if native.num_qubits != prototype.num_qubits:
                        raise BackendError(
                            "Transpilation changed width without providing a layout."
                        )
                else:
                    mapped = [operator.apply_layout(native.layout) for operator in operators]
            parameters = [bindings[p.name] for p in native.parameters]
            values = np.stack(parameters, axis=-1) if parameters else np.empty((len(rows), 0))
            # Observable (M,1) and parameter (B,) shapes broadcast to (M,B).
            pubs.append((native, [[operator] for operator in mapped], values))
            layouts.append(rows)
        metadata = {
            "kind": "estimator-v2",
            "shape": (len(circuits), len(observables)),
            "layouts": layouts,
            "max_memory_mb": options.max_memory_mb,
            "bounded_expectations": self.capabilities.bounded_expectations,
            "bounds": [
                sum(abs(t.coefficient) for t in o.terms)
                if isinstance(o, Hamiltonian)
                else abs(o.coefficient)
                for o in observables
            ],
        }
        return _provider_job(
            lambda: self.estimator.run(pubs, precision=self.precision),
            _estimator_result,
            metadata,
            self.name,
            self.recovery,
        )


@dataclass(frozen=True)
class QiskitSamplerBackend:
    """Adapt a supplied BaseSamplerV2 and preserve the IR classical register bit order."""

    sampler: object
    name: str = "qiskit-sampler-v2"
    pass_manager: object = None
    cache_size: int = 16
    recovery: RecoveryPolicy | None = None
    _cache: object = field(init=False, repr=False, compare=False)
    _transpiler_lock: object = field(init=False, repr=False, compare=False)
    capabilities: BackendCapabilities = field(
        default_factory=lambda: BackendCapabilities(
            native_sampling=True, mid_circuit_measurements=True
        )
    )

    def __post_init__(self):
        object.__setattr__(self, "_cache", CircuitCache(self.cache_size))
        object.__setattr__(self, "_transpiler_lock", RLock())
        if self.recovery is not None and not isinstance(self.recovery, RecoveryPolicy):
            raise ValidationError("recovery must be RecoveryPolicy.")
        if not callable(getattr(self.sampler, "run", None)):
            raise ValidationError("Provide an initialized Qiskit V2 sampler.")

    def clear_cache(self):
        """Invalidate ISA circuits after changing a caller-owned pass manager/target."""
        self._cache.clear()

    def sample_batch(self, circuits, shots, *, options):
        from .adapters.qiskit import to_qiskit

        measured = [
            c if c.has_measurements else replace(c, num_clbits=0).measure_all() for c in circuits
        ]
        jobs, layouts = [], []
        for rows, prototype, bindings in parameter_batches(measured, options.max_memory_mb):

            def compile_circuit(prototype=prototype):
                native = to_qiskit(prototype)
                if self.pass_manager is None:
                    return native
                with self._transpiler_lock:
                    return self.pass_manager.run(native)

            native = self._cache.get(prototype, compile_circuit)
            parameters = [bindings[p.name] for p in native.parameters]
            values = np.stack(parameters, axis=-1) if parameters else np.empty((len(rows), 0))
            jobs.append((native, values))
            layouts.append(rows)
        metadata = {
            "kind": "sampler-v2",
            "shape": (len(circuits), 1),
            "layouts": layouts,
            "max_memory_mb": options.max_memory_mb,
            "shots": shots,
            "widths": [c.num_clbits for c in measured],
        }
        return _provider_job(
            lambda: self.sampler.run(jobs, shots=shots),
            _sampler_result,
            metadata,
            self.name,
            self.recovery,
        )


@dataclass(frozen=True)
class CirqDensityBackend:
    """Exact mixed-state readouts and sampling with an optional native Cirq noise model."""

    noise: object = None
    name: str = "cirq-density"
    capabilities: BackendCapabilities = field(
        default_factory=lambda: BackendCapabilities(
            native_expectations=True,
            native_sampling=True,
            mid_circuit_measurements=True,
            density_matrix=True,
            channels=True,
            reset=True,
            classical_control=True,
        )
    )

    def prepare_gradient_circuit(self, circuit):
        # Noise acts on the exported Cirq basis. Differentiate those physical
        # rotation occurrences, retaining noise between decomposed operations.
        from ..ir.compiler import decompose

        return decompose(circuit).with_global_phase(0)

    def _engine(self, circuits, options):
        import math

        from .adapters._common import dependency

        # Density output scales as 4**n. Guard it before constructing the engine.
        limit = options.max_qubits
        memory_limit = math.floor((math.log2(options.max_memory_mb) + 20 - 6) / 2)
        if any(
            c.num_qubits > memory_limit or (limit is not None and c.num_qubits > limit)
            for c in circuits
        ):
            raise ResourceError(
                "Cirq density simulation exceeds the requested working-memory/qubit budget."
            )
        cirq = dependency("cirq", "cirq")
        # Cirq 1.7's factored-density strategy fails on controlled zero-wire
        # global phases. Select its full-state strategy only for those programs.
        conditional_phase = any(
            b.global_phase.parameters or b.global_phase.constant
            for c in circuits
            for _, op in c.walk_operations()
            for b in op.blocks
        )
        return cirq.DensityMatrixSimulator(
            dtype=np.complex128,
            noise=self.noise,
            seed=options.seed,
            split_untangled_states=not conditional_phase,
        )

    def run_density(self, circuit, *, options):
        if circuit.has_measurements:
            raise ValidationError(
                "Cirq simulate() selects a measurement trajectory; use numpy for exact branches."
            )
        import cirq

        from ..ir.compiler import decompose
        from .adapters.cirq import to_cirq

        engine = self._engine([circuit], options)
        return engine.simulate(
            to_cirq(decompose(circuit).with_global_phase(0)),
            qubit_order=cirq.LineQubit.range(circuit.num_qubits),
        ).final_density_matrix

    def expectations_batch(self, circuits, observables, *, options):
        if any(c.has_measurements for c in circuits):
            from ..sim.dense import _TensorOps
            from ..sim.density import _expectations_from_density, density_matrix

            rows = []
            for c in circuits:
                rho = density_matrix(
                    c,
                    backend=self,
                    max_qubits=options.max_qubits,
                    max_memory_mb=options.max_memory_mb,
                    max_branches=options.max_branches,
                )
                rows.append(
                    _expectations_from_density(rho, observables, c.num_qubits, _TensorOps())
                )
            return np.asarray(rows)
        from ..ir.compiler import decompose
        from ..sim.dense import _TensorOps
        from ..sim.density import _expectations_from_density
        from .adapters.cirq import to_cirq

        engine = self._engine(circuits, options)
        import cirq

        rows = []
        for circuit in circuits:
            rho = engine.simulate(
                to_cirq(decompose(circuit).with_global_phase(0)),
                qubit_order=cirq.LineQubit.range(circuit.num_qubits),
            ).final_density_matrix
            rows.append(
                _expectations_from_density(rho, observables, circuit.num_qubits, _TensorOps())
            )
        return np.asarray(rows)

    def sample_batch(self, circuits, shots, *, options):
        from ..ir.compiler import decompose
        from .adapters._cirq_control import remap_storage_block
        from .adapters.cirq import to_cirq

        engine = self._engine(circuits, options)
        # Cirq records measurement events separately, whereas IR clbits are
        # storage. Rename each event before native execution, then replay writes.
        rows = []
        for circuit in circuits:
            operations, events = [], []
            if any(b.has_measurements for _, op in circuit.walk_operations() for b in op.blocks):
                from ..errors import conversion_error

                raise conversion_error(
                    "CONTROL_FLOW_UNSUPPORTED",
                    "Cirq cannot preserve an atomic feedback body containing measurement.",
                )
            latest = {}

            for op in circuit.operations:
                if op.name == "measure":
                    key = f"c{len(events)}"
                    operations.append(replace(op, clbits=(len(events),)))
                    latest[op.clbits[0]] = len(events)
                    events.append((key, op.clbits[0]))
                elif op.name == "if_else":
                    # All writes are top-level. A condition reads the latest
                    # event for each storage bit, never the final event mapping.
                    block = replace(circuit, operations=(op,))
                    operations.extend(remap_storage_block(block, latest, len(events)).operations)
                elif op.name != "barrier":
                    operations.append(op)
            if not events:
                from ..ir.core import Operation

                for q in range(circuit.num_qubits):
                    operations.append(Operation("measure", (q,), clbits=(q,)))
                    events.append((f"c{q}", q))
            portable = replace(circuit, operations=tuple(operations), num_clbits=len(events))
            native = to_cirq(decompose(portable).with_global_phase(0))
            result = engine.run(native, repetitions=shots)
            width = circuit.num_clbits if circuit.has_measurements else circuit.num_qubits
            bits = np.zeros((shots, width), dtype=np.uint8)
            for key, bit in events:
                bits[:, bit] = result.measurements[key][:, 0]
            counts = {}
            for row in bits:
                key = "".join(str(int(bit)) for bit in row)
                counts[key] = counts.get(key, 0) + 1
            rows.append(counts)
        return rows
