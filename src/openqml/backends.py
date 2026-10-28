"""An explicit, local backend registry. Registering never imports or executes a plugin."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from threading import RLock
from typing import Protocol, runtime_checkable

import numpy as np

from .errors import ResourceError, ValidationError
from .ir import UNITARY_GATES, Circuit, integer


@dataclass(frozen=True)
class BackendCapabilities:
    """Execution features, including an explicit contract for expectation reuse.

    deterministic_expectations means identical bound circuits and options have
    identical readouts independent of batch composition or execution order.
    Sampling and stateful estimators leave it disabled.
    """

    gates: frozenset[str] = UNITARY_GATES
    max_qubits: int | None = None
    native_expectations: bool = False
    native_sampling: bool = False
    mid_circuit_measurements: bool = False
    density_matrix: bool = False
    channels: bool = False
    reset: bool = False
    classical_control: bool = False
    measurement_expectations: bool = False
    deterministic_expectations: bool = False

    def __post_init__(self):
        if any(
            type(value) is not bool
            for value in (
                self.native_expectations,
                self.native_sampling,
                self.mid_circuit_measurements,
                self.density_matrix,
                self.channels,
                self.reset,
                self.classical_control,
                self.measurement_expectations,
                self.deterministic_expectations,
            )
        ):
            raise ValidationError("Backend capability flags must be booleans.")
        gates = frozenset(self.gates)
        if not gates or gates - UNITARY_GATES:
            raise ValidationError("Backend gates must be known unitary gate names.")
        if self.max_qubits is not None:
            integer(self.max_qubits, "backend max_qubits", 1)
        object.__setattr__(self, "gates", gates)


@dataclass(frozen=True)
class ExecutionOptions:
    max_qubits: int | None = 20
    max_memory_mb: float = 256
    seed: int | None = None
    max_branches: int = 256

    @classmethod
    def for_statevector(cls, max_memory_mb, *, batch_size=1, max_qubits=None, seed=None):
        """Derive an explicit width cap from four complex128 state buffers per row.

        Defaults stay fixed and reproducible. This opt-in policy sizes a dense
        unitary workload; density branches, local matrices and autodiff tapes
        have separate costs. Every execution still checks its actual resources.
        """
        from .parameters import real

        budget = real(max_memory_mb)
        count = integer(batch_size, "batch_size", 1)
        if budget <= 0:
            raise ValidationError("max_memory_mb must be positive.")
        width = math.floor(math.log2(budget) + 20 - math.log2(4 * 16) - math.log2(count))
        if max_qubits is not None:
            width = min(width, integer(max_qubits, "max_qubits", 1))
        if width < 1:
            raise ResourceError("The statevector budget cannot hold one quantum wire.")
        return cls(width, budget, seed)

    def __post_init__(self):
        from .parameters import real

        if self.max_qubits is not None:
            integer(self.max_qubits, "max_qubits", 1)
        if real(self.max_memory_mb) <= 0:
            raise ValidationError("max_memory_mb must be positive.")
        if self.seed is not None:
            integer(self.seed, "seed")
        integer(self.max_branches, "max_branches", 1)


@runtime_checkable
class StatevectorBackend(Protocol):
    """Receive a bound, barrier-free unitary circuit and return a q0-MSB state.

    Adapters must preserve the full complex phase. Validation, basis lowering,
    terminal sampling and bit formatting are owned by execute(). Backends may
    use options.seed; any other stochastic state evolution is outside this contract.
    """

    name: str
    capabilities: BackendCapabilities

    def run(self, circuit: Circuit, *, options: ExecutionOptions) -> np.ndarray: ...


@runtime_checkable
class ExpectationBackend(Protocol):
    name: str
    capabilities: BackendCapabilities

    def expectations_batch(
        self, circuits, observables, *, options: ExecutionOptions
    ) -> np.ndarray: ...


@runtime_checkable
class SamplingBackend(Protocol):
    name: str
    capabilities: BackendCapabilities

    def sample_batch(
        self, circuits, shots: int, *, options: ExecutionOptions
    ) -> list[dict[str, int]]: ...


@runtime_checkable
class DensityMatrixBackend(Protocol):
    name: str
    capabilities: BackendCapabilities

    def run_density(self, circuit: Circuit, *, options: ExecutionOptions) -> np.ndarray: ...


def check_effects(circuit, device, *, expectation=False):
    from .ir import CHANNEL_GATES

    if (
        expectation
        and circuit.has_measurements
        and not device.capabilities.measurement_expectations
        and not (
            device.capabilities.density_matrix
            and device.capabilities.channels
            and all(op.name != "if_else" for _, op in circuit.walk_operations())
        )
    ):
        raise ValidationError(
            f"Backend {device.name!r} lacks exact feedback ensembles. Use numpy or ShotBackend."
        )
    for _, op in circuit.walk_operations():
        if op.name in CHANNEL_GATES and not device.capabilities.channels:
            raise ValidationError(f"Backend {device.name!r} does not support quantum channels.")
        if op.name == "reset" and not device.capabilities.reset:
            raise ValidationError(f"Backend {device.name!r} does not support reset.")
        if op.name == "if_else" and not device.capabilities.classical_control:
            raise ValidationError(f"Backend {device.name!r} does not support classical control.")


@dataclass(frozen=True)
class _LocalBackend:
    name: str
    capabilities: BackendCapabilities = BackendCapabilities(deterministic_expectations=True)

    def run(self, circuit, *, options):
        from .simulator import _external_state, statevector

        if self.name == "numpy":
            return statevector(
                circuit, max_qubits=options.max_qubits, max_memory_mb=options.max_memory_mb
            )
        return _external_state(circuit, self.name)

    def run_density(self, circuit, *, options):
        if self.name == "numpy":
            from .density import density_matrix

            return density_matrix(
                circuit,
                max_qubits=options.max_qubits or 20,
                max_memory_mb=options.max_memory_mb,
                max_branches=options.max_branches,
            )
        from .density import external_density

        return external_density(circuit, self.name)


_BUILTIN_NAMES = frozenset(
    {"numpy", "qiskit", "pennylane", "cirq", "aer", "aer-mps", "aer-density", "cirq-density"}
)
_REGISTRY = {name: _LocalBackend(name) for name in ("numpy", "qiskit", "pennylane", "cirq")}
_REGISTRY["numpy"] = _LocalBackend(
    "numpy",
    BackendCapabilities(
        density_matrix=True,
        channels=True,
        reset=True,
        classical_control=True,
        mid_circuit_measurements=True,
        measurement_expectations=True,
        deterministic_expectations=True,
    ),
)
for _name in ("qiskit", "pennylane", "cirq"):
    _REGISTRY[_name] = _LocalBackend(
        _name,
        BackendCapabilities(
            density_matrix=True, channels=True, reset=True, deterministic_expectations=True
        ),
    )
_LOCK = RLock()


def _validate_backend(backend):
    if (
        not isinstance(getattr(backend, "name", None), str)
        or not re.fullmatch(r"[a-z][a-z0-9_-]*", backend.name)
        or not isinstance(getattr(backend, "capabilities", None), BackendCapabilities)
    ):
        raise ValidationError("A backend needs a lowercase name and BackendCapabilities.")
    capabilities = backend.capabilities
    states = callable(getattr(backend, "run", None))
    expectations = callable(getattr(backend, "expectations_batch", None))
    sampling = callable(getattr(backend, "sample_batch", None))
    if capabilities.native_expectations and not expectations:
        raise ValidationError("native_expectations requires expectations_batch().")
    if capabilities.native_sampling and not sampling:
        raise ValidationError("native_sampling requires sample_batch().")
    if capabilities.density_matrix and not callable(getattr(backend, "run_density", None)):
        raise ValidationError("density_matrix requires run_density().")
    if not (
        states
        or (capabilities.density_matrix and callable(getattr(backend, "run_density", None)))
        or (expectations and capabilities.native_expectations)
        or (sampling and capabilities.native_sampling)
    ):
        raise ValidationError(
            "A backend needs a statevector, density, expectation or sampling implementation."
        )
    return backend


def register_backend(backend: StatevectorBackend, *, replace=False) -> None:
    """Explicitly register an instance. Built-in names are protected."""
    _validate_backend(backend)
    with _LOCK:
        if backend.name in _BUILTIN_NAMES:
            raise ValidationError("Built-in backend names cannot be replaced.")
        if backend.name in _REGISTRY and not replace:
            raise ValidationError(f"Backend {backend.name!r} is already registered.")
        _REGISTRY[backend.name] = backend


def unregister_backend(name: str) -> StatevectorBackend:
    with _LOCK:
        if name in _BUILTIN_NAMES or name not in _REGISTRY:
            raise ValidationError("Only an existing custom backend can be unregistered.")
        return _REGISTRY.pop(name)


def list_backends() -> tuple[str, ...]:
    """Return registered names; built-in implementations load on first selection."""
    with _LOCK:
        return tuple(sorted(set(_REGISTRY) | _BUILTIN_NAMES))


def get_backend(backend: str | StatevectorBackend) -> StatevectorBackend:
    if not isinstance(backend, str):
        return _validate_backend(backend)
    with _LOCK:
        if backend not in _REGISTRY and backend in {"aer", "aer-mps", "aer-density"}:
            from .aer import AerBackend

            _REGISTRY[backend] = AerBackend(
                name=backend,
                method=(
                    "matrix_product_state"
                    if backend == "aer-mps"
                    else "density_matrix"
                    if backend == "aer-density"
                    else "statevector"
                ),
            )
        elif backend not in _REGISTRY and backend == "cirq-density":
            from .providers import CirqDensityBackend

            _REGISTRY[backend] = CirqDensityBackend()
        if backend not in _REGISTRY:
            raise ValidationError(f"Unknown backend {backend!r}. Registered: {list_backends()}.")
        return _REGISTRY[backend]
