"""Qiskit Aer engines: batch jobs and native MPS readouts without dense state output."""

from collections import OrderedDict
from dataclasses import dataclass, field, replace
from threading import RLock

import numpy as np

from ..errors import BackendError, ResourceError, ValidationError, _require_choice
from ..ir.observables import Hamiltonian
from ..runtime.backends import BackendCapabilities


@dataclass(frozen=True)
class AerBackend:
    name: str = "aer"
    method: str = "statevector"
    max_parallel_threads: int = 1
    noise_model: object = None
    device: str = "CPU"
    parameter_batching: bool = True
    compile_cache_size: int = 8
    mps_truncation_threshold: float = 0.0
    mps_max_bond_dimension: int | None = None
    precision: str = "double"
    cu_statevec: bool = False
    target_gpus: tuple[int, ...] | None = None
    blocking_qubits: int | None = None
    batched_shots_gpu: bool = False
    runtime_parameter_binding: bool = False
    _compiled: object = field(default_factory=OrderedDict, repr=False, compare=False)
    _compile_lock: object = field(default_factory=RLock, repr=False, compare=False)
    _cache_stats: dict = field(
        default_factory=lambda: {"hits": 0, "misses": 0}, repr=False, compare=False
    )
    capabilities: BackendCapabilities = field(
        default_factory=lambda: BackendCapabilities(
            native_expectations=True,
            native_sampling=True,
            mid_circuit_measurements=True,
            channels=True,
            reset=True,
            classical_control=True,
        )
    )

    def __post_init__(self):
        from ..ir.core import integer
        from ..ir.parameters import real

        _require_choice(self.device, ("CPU", "GPU"), "Aer device")
        _require_choice(self.precision, ("single", "double"), "Aer precision")
        _require_choice(
            self.method,
            (
                "automatic",
                "statevector",
                "density_matrix",
                "matrix_product_state",
                "stabilizer",
                "tensor_network",
                "extended_stabilizer",
            ),
            "Aer method",
        )
        integer(self.max_parallel_threads, "max_parallel_threads")
        for name in (
            "parameter_batching",
            "cu_statevec",
            "batched_shots_gpu",
            "runtime_parameter_binding",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValidationError(f"{name} must be a boolean.")
        if self.device != "GPU" and (
            self.cu_statevec
            or self.batched_shots_gpu
            or self.target_gpus is not None
            or self.method == "tensor_network"
        ):
            raise ValidationError("GPU acceleration options require device='GPU'.")
        if self.device == "GPU" and self.method in {
            "stabilizer",
            "matrix_product_state",
            "extended_stabilizer",
        }:
            raise ValidationError("This Aer method does not support GPU execution.")
        if self.cu_statevec and (self.batched_shots_gpu or self.method != "statevector"):
            raise ValidationError(
                "cuStateVec requires statevector and cannot combine with batched_shots_gpu."
            )
        if self.target_gpus is not None:
            if not isinstance(self.target_gpus, (tuple, list)) or not self.target_gpus:
                raise ValidationError("target_gpus requires a nonempty sequence of GPU indices.")
            indices = tuple(integer(i, "GPU index") for i in self.target_gpus)
            if len(set(indices)) != len(indices):
                raise ValidationError("target_gpus must contain distinct GPU indices.")
            object.__setattr__(self, "target_gpus", indices)
        if self.blocking_qubits is not None:
            integer(self.blocking_qubits, "blocking_qubits", 1)
            if self.method not in {"statevector", "density_matrix"}:
                raise ValidationError("Cache blocking requires statevector or density_matrix.")
        object.__setattr__(
            self,
            "capabilities",
            replace(
                self.capabilities,
                device_type=self.device,
                deterministic_expectations=self.method != "extended_stabilizer",
                exact_expectations=self.method
                in {"statevector", "density_matrix", "stabilizer", "tensor_network"},
            ),
        )
        integer(self.compile_cache_size, "compile_cache_size", 1)
        if real(self.mps_truncation_threshold) < 0:
            raise ValidationError("mps_truncation_threshold must be nonnegative.")
        if self.mps_max_bond_dimension is not None:
            integer(self.mps_max_bond_dimension, "mps_max_bond_dimension", 1)
        if self.method == "stabilizer":
            from .semantics import _CLIFFORD

            object.__setattr__(
                self,
                "capabilities",
                replace(
                    self.capabilities,
                    gates=_CLIFFORD,
                    channels=False,
                    deterministic_expectations=True,
                ),
            )
        if self.method == "density_matrix":
            object.__setattr__(
                self, "capabilities", replace(self.capabilities, density_matrix=True)
            )
        if self.noise_model is not None and self.method != "density_matrix":
            raise ValidationError("Use method='density_matrix' for exact noisy Aer readouts.")

    def clear_cache(self):
        """Clear compiled topology/readout plans, e.g. after mutating a native noise model."""
        with self._compile_lock:
            self._compiled.clear()
            self._cache_stats.update(hits=0, misses=0)

    def execution_info(self):
        """Describe configured implementation sources without importing/probing Aer."""
        return {
            "engine": "qiskit-aer",
            "algorithm_owner": "Qiskit Aer",
            "kernel_provider": "qiskit-aer",
            "device": self.device,
            "method": self.method,
            "precision": self.precision,
            "requested_accelerator": (
                "cuStateVec"
                if self.cu_statevec
                else "cuTensorNet"
                if self.method == "tensor_network"
                else None
            ),
            "execution_verified": False,
        }

    def is_available(self):
        """Query the installed native engine without submitting simulation work."""
        from ..errors import MissingDependencyError
        from .adapters._common import dependency

        try:
            engine = dependency(
                "qiskit_aer", "aer-gpu" if self.device == "GPU" else "aer"
            ).AerSimulator()
            return (
                self.device in engine.available_devices()
                and self.method in engine.available_methods()
            )
        except (MissingDependencyError, ImportError, OSError):
            return False

    def admit(self, task, circuits, observables, *, options):
        from ..sim.dense import _resource_check
        from ..sim.density import _resource_check as density_check

        if not self.is_available():
            raise ResourceError(
                f"Native Aer {self.device} execution is unavailable; install its matching runtime."
            )
        for circuit in circuits:
            if options.max_qubits is not None and circuit.num_qubits > options.max_qubits:
                raise ResourceError("Circuit exceeds the requested qubit limit.")
            if task == "expectations" and circuit.has_effects and self.method != "density_matrix":
                raise ValidationError("Exact effectful Aer readouts require density_matrix.")
            if self.method == "density_matrix":
                density_check(circuit, options.max_qubits, options.max_memory_mb)
            elif self.method in {"statevector"} or task == "statevector":
                _resource_check(circuit, options.max_qubits, options.max_memory_mb)

    def cache_info(self):
        with self._compile_lock:
            return {
                **self._cache_stats,
                "size": len(self._compiled),
                "maxsize": self.compile_cache_size,
            }

    @staticmethod
    def _signature(circuit):
        return circuit.num_qubits, tuple(
            (op.name, op.qubits, op.matrix, len(op.params)) for op in circuit.operations
        )

    def _parameter_batch(self, circuits, observables, options):
        signature = self._signature(circuits[0])
        if any(self._signature(c) != signature for c in circuits[1:]):
            return None
        engine = self._engine(options)
        if circuits[0].num_qubits > engine.num_qubits:
            engine.set_max_qubits(circuits[0].num_qubits)
        # Readouts do not depend on global phase. Cache a symbolic gate-angle
        # prototype rather than bound circuits, retaining a bounded LRU per engine.
        key = (
            self.method,
            self.device,
            self.precision,
            self.cu_statevec,
            self.target_gpus,
            self.blocking_qubits,
            signature,
            tuple(observables),
        )
        with self._compile_lock:
            if key in self._compiled:
                compiled = self._compiled.pop(key)
                self._compiled[key] = compiled
                self._cache_stats["hits"] += 1
            else:
                from qiskit import transpile

                from ..ir.core import Circuit
                from ..ir.parameters import Parameter, expression
                from .adapters.qiskit import to_qiskit

                operations = tuple(
                    replace(
                        op,
                        params=tuple(
                            expression(Parameter(f"a_{i}_{j}")) for j in range(len(op.params))
                        ),
                    )
                    for i, op in enumerate(circuits[0].operations)
                )
                prototype = Circuit(circuits[0].num_qubits, operations)
                native = to_qiskit(prototype)
                self._add_readouts(native, observables)
                compiled = transpile(native, engine, optimization_level=0, num_processes=1)
                self._compiled[key] = compiled
                while len(self._compiled) > self.compile_cache_size:
                    self._compiled.popitem(last=False)
                self._cache_stats["misses"] += 1
        if compiled.parameters:
            parameters = {p.name: p for p in compiled.parameters}
            bindings = {
                parameters[f"a_{i}_{j}"]: [c.operations[i].params[j].constant for c in circuits]
                for i, op in enumerate(circuits[0].operations)
                for j in range(len(op.params))
            }
            if any(op.name in {"cp", "crx", "cry", "crz"} for op in circuits[0].operations):
                # Aer 0.18/Qiskit 2.6 can omit ControlledGate base parameters in
                # parameter_binds. Bind the cached native circuit instead; never
                # lower away gates that a native noise model may act on.
                jobs = [
                    compiled.assign_parameters({p: values[i] for p, values in bindings.items()})
                    for i in range(len(circuits))
                ]
                result = engine.run(jobs, shots=1, seed_simulator=options.seed).result()
            else:
                result = engine.run(
                    compiled, parameter_binds=[bindings], shots=1, seed_simulator=options.seed
                ).result()
            repetitions = len(circuits)
        else:
            result = engine.run(compiled, shots=1, seed_simulator=options.seed).result()
            repetitions = 1
        if not result.success:
            raise BackendError(f"Aer parameter batch failed: {result.status}")
        values = self._read_values(result, repetitions, len(observables))
        return np.repeat(values, len(circuits), axis=0) if repetitions == 1 else values

    @staticmethod
    def _add_readouts(native, observables):
        from qiskit.quantum_info import SparsePauliOp

        for i, observable in enumerate(observables):
            terms = observable.terms if isinstance(observable, Hamiltonian) else (observable,)
            operator = SparsePauliOp.from_list([(t.paulis[::-1], t.coefficient) for t in terms])
            native.save_expectation_value(operator, list(range(native.num_qubits)), label=f"e{i}")

    @staticmethod
    def _read_values(result, batch_size, num_observables):
        return np.asarray(
            [
                [float(np.real(result.data(i)[f"e{j}"])) for j in range(num_observables)]
                for i in range(batch_size)
            ]
        )

    def _engine(self, options):
        from .adapters._common import dependency

        aer = dependency("qiskit_aer", "aer-gpu" if self.device == "GPU" else "aer")
        if self.device not in aer.AerSimulator().available_devices():
            raise ResourceError(
                f"Installed Aer cannot execute on {self.device}; no CPU fallback was selected."
            )
        settings = dict(
            precision=self.precision,
            fusion_enable=True,
            runtime_parameter_bind_enable=self.runtime_parameter_binding,
        )
        if self.device == "GPU":
            settings.update(
                cuStateVec_enable=self.cu_statevec, batched_shots_gpu=self.batched_shots_gpu
            )
            if self.target_gpus is not None:
                settings["target_gpus"] = list(self.target_gpus)
        if self.blocking_qubits is not None:
            settings.update(blocking_enable=True, blocking_qubits=self.blocking_qubits)
        if self.method == "matrix_product_state":
            settings["matrix_product_state_truncation_threshold"] = self.mps_truncation_threshold
            settings["matrix_product_state_max_bond_dimension"] = self.mps_max_bond_dimension
        return aer.AerSimulator(
            method=self.method,
            noise_model=self.noise_model,
            device=self.device,
            max_parallel_threads=self.max_parallel_threads,
            max_memory_mb=max(1, int(np.ceil(options.max_memory_mb))),
            **settings,
        )

    def _run(self, circuits, options, shots=1):
        from qiskit import transpile

        engine = self._engine(options)
        width = max(c.num_qubits for c in circuits)
        if width > engine.num_qubits:
            engine.set_max_qubits(width)
        circuits = transpile(circuits, engine, optimization_level=0, num_processes=1)
        result = engine.run(circuits, shots=shots, seed_simulator=options.seed).result()
        if not result.success:
            raise BackendError(f"Aer failed: {result.status}")
        return result

    def run(self, circuit, *, options):
        return self.run_batch([circuit], options=options)[0]

    def run_batch(self, circuits, *, options):
        if self.method == "stabilizer":
            raise BackendError(
                "Stabilizer execution provides native readouts and samples, not dense state output."
            )
        if (
            self.noise_model is not None
            or self.method == "density_matrix"
            or any(c.has_effects for c in circuits)
        ):
            raise BackendError(
                "Mixed-state engines provide expectations/samples, not a pure statevector."
            )
        self._engine(options)
        from .adapters.qiskit import to_qiskit

        jobs = []
        for circuit in circuits:
            native = to_qiskit(circuit)
            native.save_statevector()
            jobs.append(native)
        result = self._run(jobs, options)
        states = []
        for i, circuit in enumerate(circuits):
            state = np.asarray(result.data(i)["statevector"])
            states.append(
                state.reshape((2,) * circuit.num_qubits)
                .transpose(tuple(reversed(range(circuit.num_qubits))))
                .reshape(-1)
            )
        return states

    def run_density(self, circuit, *, options):
        from ..sim.density import _resource_check
        from .adapters.qiskit import to_qiskit

        if self.method != "density_matrix":
            raise ValidationError("Use the aer-density method for exact density output.")
        if circuit.has_measurements:
            raise ValidationError(
                "Aer samples measurement branches; use numpy for exact ensembles."
            )
        _resource_check(circuit, options.max_qubits, options.max_memory_mb)
        self._engine(options)
        native = to_qiskit(circuit)
        native.save_density_matrix(pershot=False)
        result = self._run([native], options, shots=1)
        rho = np.asarray(result.data(0)["density_matrix"])
        n = circuit.num_qubits
        axes = tuple(reversed(range(n))) + tuple(reversed(range(n, 2 * n)))
        return rho.reshape((2,) * (2 * n)).transpose(axes).reshape(rho.shape)

    def expectations_batch(self, circuits, observables, *, options):
        if any(c.has_effects for c in circuits):
            if self.method != "density_matrix":
                raise ValidationError(
                    "Use aer-density for exact effectful readouts, or ShotBackend for sampling."
                )
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
        if self.method == "density_matrix":
            from ..sim.density import _resource_check

            for c in circuits:
                _resource_check(c, options.max_qubits, options.max_memory_mb)
        self._engine(options)
        if self.parameter_batching:
            values = self._parameter_batch(circuits, observables, options)
            if values is not None:
                return values

        from .adapters.qiskit import to_qiskit

        jobs = []
        for circuit in circuits:
            native = to_qiskit(circuit)
            self._add_readouts(native, observables)
            jobs.append(native)
        result = self._run(jobs, options)
        return self._read_values(result, len(circuits), len(observables))

    def sample_batch(self, circuits, shots, *, options):
        from .adapters.qiskit import to_qiskit

        jobs = []
        for circuit in circuits:
            if not circuit.has_measurements:
                circuit = replace(circuit, num_clbits=0).measure_all()
            jobs.append(to_qiskit(circuit))
        result = self._run(jobs, options, shots=shots)
        return [
            {key.replace(" ", "")[::-1]: int(value) for key, value in result.get_counts(i).items()}
            for i in range(len(jobs))
        ]
