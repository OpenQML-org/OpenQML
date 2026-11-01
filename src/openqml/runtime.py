"""Capability-preserving QPU -> GPU -> CPU selection before submitting work."""

from dataclasses import asdict, dataclass, field, replace

from .backends import BackendCapabilities, ExecutionOptions
from .errors import ConversionError, MissingDependencyError, ResourceError, ValidationError


@dataclass(frozen=True)
class DispatchCandidate:
    backend: str
    device: str
    eligible: bool
    reason: str


@dataclass(frozen=True)
class DispatchPlan:
    backend: str
    device: str
    task: str
    candidates: tuple[DispatchCandidate, ...]
    _backend: object = field(repr=False, compare=False)

    def to_dict(self):
        return {
            "backend": self.backend,
            "device": self.device,
            "task": self.task,
            "candidates": [asdict(c) for c in self.candidates],
        }


@dataclass(frozen=True)
class QPUBackend:
    """Mark an explicitly configured provider as hardware eligible for auto.

    No credentials or cloud devices are discovered. available is a boolean or a
    caller-owned zero-argument availability check; it must not submit jobs.
    Provider execution errors propagate and are never retried on another device.
    """

    backend: object
    available: object = True

    def __post_init__(self):
        from .backends import _validate_backend

        _validate_backend(self.backend)
        if type(self.available) is not bool and not callable(self.available):
            raise ValidationError("QPU availability must be a boolean or callable.")

    @property
    def name(self):
        return self.backend.name

    @property
    def capabilities(self):
        return replace(self.backend.capabilities, device_type="QPU")

    def is_available(self):
        value = self.available() if callable(self.available) else self.available
        if type(value) is not bool:
            raise ValidationError("QPU availability checks must return a boolean.")
        return value

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "backend"), name)


def register_qpu_backend(backend, *, available=True, replace=False):
    """Register a caller-configured QPU with highest automatic priority."""
    from .backends import register_backend

    wrapped = QPUBackend(backend, available)
    register_backend(wrapped, replace=replace)
    return wrapped


def resolve_backend(backend, task, circuits, observables=(), *, options):
    """Resolve only automatic routing; explicit backend instances remain pinned."""
    return (
        backend.plan(circuits, task=task, observables=observables, options=options)._backend
        if isinstance(backend, PriorityBackend)
        else backend
    )


@dataclass(frozen=True)
class PriorityBackend:
    name: str = "auto"
    require_exact: bool = False
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
        )
    )

    def __post_init__(self):
        if type(self.require_exact) is not bool:
            raise ValidationError("require_exact must be a boolean.")

    def plan(self, circuits, *, task="expectations", observables=(), options=None):
        """Inspect an automatic route without executing a circuit or cloud job."""
        from .backends import _LOCK, _REGISTRY, check_effects, get_backend
        from .compiler import decompose
        from .ir import Circuit

        methods = {
            "expectations": "expectations_batch",
            "sampling": "sample_batch",
            "statevector": "run",
            "density": "run_density",
        }
        if task not in methods:
            raise ValidationError("Unknown dispatch task.")
        circuits = (circuits,) if isinstance(circuits, Circuit) else tuple(circuits)
        observables = tuple(observables)
        if not circuits:
            raise ValidationError("Dispatch planning requires at least one circuit.")
        if any(not isinstance(c, Circuit) for c in circuits):
            raise ValidationError("Dispatch inputs must be Circuit objects.")
        options = ExecutionOptions() if options is None else options
        if not isinstance(options, ExecutionOptions):
            raise ValidationError("options must be ExecutionOptions.")
        if any(
            options.max_qubits is not None and c.num_qubits > options.max_qubits for c in circuits
        ):
            raise ResourceError("Circuit exceeds the requested qubit limit.")
        with _LOCK:
            devices = [d for d in _REGISTRY.values() if d.capabilities.device_type == "QPU"]
        candidates = []
        # Lazy resolution keeps Torch and cloud SDKs outside a core-only import.
        for device in (*devices, "gpu", "cpu"):
            if isinstance(device, str):
                device = get_backend(device)
            try:
                caps = device.capabilities
                available = getattr(device, "is_available", lambda: True)()
                if type(available) is not bool:
                    raise ValidationError("Device availability must be a boolean.")
                if not available:
                    raise ResourceError("Device is unavailable or has not been enabled.")
                if not callable(getattr(device, methods[task], None)):
                    raise ValidationError(f"No {task} interface.")
                if task == "expectations" and not caps.native_expectations:
                    raise ValidationError("No native expectation interface.")
                if task == "sampling" and not caps.native_sampling:
                    raise ValidationError("No native sampling interface.")
                if (
                    task == "sampling"
                    and caps.device_type != "CPU"
                    and any(c.num_clbits > options.max_memory_mb * 2**20 for c in circuits)
                ):
                    raise ResourceError(
                        "The native sampler's full classical strings exceed the budget."
                    )
                if task == "density" and not caps.density_matrix:
                    raise ValidationError("No density matrix interface.")
                if task == "expectations" and self.require_exact and not caps.exact_expectations:
                    raise ValidationError("Exact expectations are required by this policy.")
                for circuit in circuits:
                    if caps.max_qubits is not None and circuit.num_qubits > caps.max_qubits:
                        raise ResourceError("Circuit exceeds the device's physical width.")
                    check_effects(circuit, device, expectation=task == "expectations")
                    if task == "statevector" and circuit.has_effects:
                        raise ValidationError("Effects do not have a single pure output state.")
                    decompose(circuit, basis=caps.gates)
                admit = getattr(device, "admit", None)
                if admit is not None:
                    admit(task, circuits, observables, options=options)
            except (MissingDependencyError, ResourceError, ConversionError, ValidationError) as exc:
                candidates.append(
                    DispatchCandidate(device.name, device.capabilities.device_type, False, str(exc))
                )
                continue
            candidates.append(
                DispatchCandidate(
                    device.name,
                    device.capabilities.device_type,
                    True,
                    "Available and supports the requested task and resource policy.",
                )
            )
            return DispatchPlan(
                device.name, device.capabilities.device_type, task, tuple(candidates), device
            )
        raise ResourceError(
            "No eligible device. " + "; ".join(f"{c.backend}: {c.reason}" for c in candidates)
        )

    def expectations_batch(self, circuits, observables, *, options):
        from .batch import expectations_batch

        device = resolve_backend(self, "expectations", circuits, observables, options=options)
        return expectations_batch(circuits, observables, backend=device, **options.__dict__)

    def sample_batch(self, circuits, shots, *, options):
        from .batch import execute_batch

        device = resolve_backend(self, "sampling", circuits, options=options)
        return [
            r.counts
            for r in execute_batch(circuits, shots=shots, backend=device, **options.__dict__)
        ]

    def run(self, circuit, *, options):
        from ._validation import invoke

        device = resolve_backend(self, "statevector", [circuit], options=options)
        return invoke(device, "run", circuit, options=options)

    def run_density(self, circuit, *, options):
        from ._validation import invoke

        device = resolve_backend(self, "density", [circuit], options=options)
        return invoke(device, "run_density", circuit, options=options)
