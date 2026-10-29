"""Adapters for caller-provided execution engines."""

from dataclasses import dataclass, field, replace

import numpy as np

from .backends import BackendCapabilities
from .errors import BackendError, ResourceError, ValidationError
from .observables import Hamiltonian


@dataclass(frozen=True)
class QiskitEstimatorBackend:
    """Adapt a supplied BaseEstimatorV2 with observable layout mapping for ISA circuits.

    pass_manager, when supplied, must expose run(circuit). Each public V2 result
    contains one expectation per requested observable. Provider job management,
    precision, credentials and hardware access remain with the supplied object.
    """

    estimator: object
    name: str = "qiskit-estimator-v2"
    precision: float | None = None
    pass_manager: object = None
    capabilities: BackendCapabilities = field(
        default_factory=lambda: BackendCapabilities(native_expectations=True)
    )

    def __post_init__(self):
        if not callable(getattr(self.estimator, "run", None)):
            raise ValidationError("Provide an initialized Qiskit V2 estimator.")
        if self.precision is not None:
            from .parameters import real

            if real(self.precision) < 0:
                raise ValidationError("Estimator precision must be nonnegative.")

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
        pubs = []
        for circuit in circuits:
            native = to_qiskit(circuit)
            mapped = operators
            if self.pass_manager is not None:
                native = self.pass_manager.run(native)
                if native.layout is None:
                    if native.num_qubits != circuit.num_qubits:
                        raise BackendError(
                            "Transpilation changed width without providing a layout."
                        )
                else:
                    mapped = [operator.apply_layout(native.layout) for operator in operators]
            pubs.append((native, mapped))
        try:
            results = self.estimator.run(pubs, precision=self.precision).result()
            if len(results) != len(circuits):
                raise BackendError("Estimator returned an unexpected number of PUB results.")
            return np.asarray(
                [np.asarray(pub.data.evs).reshape(-1) for pub in results], dtype=float
            )
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError(f"Qiskit EstimatorV2 failed: {exc}") from exc


@dataclass(frozen=True)
class QiskitSamplerBackend:
    """Adapt a supplied BaseSamplerV2 and preserve the IR classical register bit order."""

    sampler: object
    name: str = "qiskit-sampler-v2"
    pass_manager: object = None
    capabilities: BackendCapabilities = field(
        default_factory=lambda: BackendCapabilities(
            native_sampling=True, mid_circuit_measurements=True
        )
    )

    def __post_init__(self):
        if not callable(getattr(self.sampler, "run", None)):
            raise ValidationError("Provide an initialized Qiskit V2 sampler.")

    def sample_batch(self, circuits, shots, *, options):
        from .adapters.qiskit import to_qiskit

        jobs = []
        for circuit in circuits:
            if not circuit.has_measurements:
                circuit = replace(circuit, num_clbits=0).measure_all()
            native = to_qiskit(circuit)
            if self.pass_manager is not None:
                native = self.pass_manager.run(native)
            jobs.append(native)
        try:
            results = self.sampler.run([(job,) for job in jobs], shots=shots).result()
            if len(results) != len(jobs):
                raise BackendError("Sampler returned an unexpected number of PUB results.")
            # to_qiskit creates one register named c, in c0..cN IR order. A pass
            # manager may add ancillas but must preserve that classical interface.
            return [
                {key[::-1]: int(n) for key, n in pub.data.c.get_counts().items()} for pub in results
            ]
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError(f"Qiskit SamplerV2 failed: {exc}") from exc


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
        from .compiler import decompose

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

        from .adapters.cirq import to_cirq
        from .compiler import decompose

        engine = self._engine([circuit], options)
        return engine.simulate(
            to_cirq(decompose(circuit).with_global_phase(0)),
            qubit_order=cirq.LineQubit.range(circuit.num_qubits),
        ).final_density_matrix

    def expectations_batch(self, circuits, observables, *, options):
        if any(c.has_measurements for c in circuits):
            from .density import _expectations_from_density, density_matrix
            from .simulator import _TensorOps

            rows = []
            for c in circuits:
                rho = density_matrix(
                    c,
                    backend=self,
                    max_qubits=options.max_qubits or 20,
                    max_memory_mb=options.max_memory_mb,
                    max_branches=options.max_branches,
                )
                rows.append(
                    _expectations_from_density(rho, observables, c.num_qubits, _TensorOps())
                )
            return np.asarray(rows)
        from .adapters.cirq import to_cirq
        from .compiler import decompose
        from .density import _expectations_from_density
        from .simulator import _TensorOps

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
        from .adapters._cirq_control import remap_storage_block
        from .adapters.cirq import to_cirq
        from .compiler import decompose

        engine = self._engine(circuits, options)
        # Cirq records measurement events separately, whereas IR clbits are
        # storage. Rename each event before native execution, then replay writes.
        rows = []
        for circuit in circuits:
            operations, events = [], []
            if any(b.has_measurements for _, op in circuit.walk_operations() for b in op.blocks):
                from .errors import conversion_error

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
                from .ir import Operation

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
