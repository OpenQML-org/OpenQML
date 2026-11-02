"""Capability admission and cost-aware dispatch before submitting work."""

import hashlib
import time
from dataclasses import asdict, dataclass, field, replace
from types import MappingProxyType

from .backends import BackendCapabilities, ExecutionOptions
from .costs import (
    BudgetLedger,
    ExecutionFeedback,
    ExecutionProfile,
    SchedulingPolicy,
    SchedulingRequest,
    TrainingWorkload,
    check_cost,
    estimate_cost,
    select_cost,
)
from .errors import ConversionError, MissingDependencyError, ResourceError, ValidationError


@dataclass(frozen=True)
class DispatchCandidate:
    backend: str
    device: str
    eligible: bool
    reason: str
    estimate: dict | None = None


@dataclass(frozen=True)
class DispatchPlan:
    backend: str
    device: str
    task: str
    candidates: tuple[DispatchCandidate, ...]
    _backend: object = field(repr=False, compare=False)
    decision: str = "hardware_priority"
    pareto_backends: tuple[str, ...] = ()

    def to_dict(self):
        return {
            "backend": self.backend,
            "device": self.device,
            "task": self.task,
            "candidates": [asdict(c) for c in self.candidates],
            "decision": self.decision,
            "pareto_backends": list(self.pareto_backends),
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
    execution_profile: object = None

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


def register_qpu_backend(backend, *, available=True, replace=False, profile=None):
    """Register configured hardware; auto requires a fresh cost/quality quote."""
    from .backends import register_backend

    if profile is not None and not isinstance(profile, ExecutionProfile) and not callable(profile):
        raise ValidationError("profile must be ExecutionProfile or a read-only quote callback.")
    wrapped = QPUBackend(backend, available, profile)
    register_backend(wrapped, replace=replace)
    return wrapped


def resolve_backend(backend, task, circuits, observables=(), *, options, shots=None, workload=None):
    """Resolve only automatic routing; explicit backend instances remain pinned."""
    return (
        backend.plan(
            circuits,
            task=task,
            observables=observables,
            options=options,
            shots=shots,
            workload=workload,
        )._backend
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

    def plan(
        self,
        circuits,
        *,
        task="expectations",
        observables=(),
        options=None,
        shots=None,
        workload=None,
        _collect=False,
    ):
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
        candidates, eligible = [], []
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
            eligible.append(device)
            if _collect:
                continue
            return DispatchPlan(
                device.name, device.capabilities.device_type, task, tuple(candidates), device
            )
        if _collect:
            return tuple(eligible), tuple(candidates)
        raise ResourceError(
            "No eligible device. " + "; ".join(f"{c.backend}: {c.reason}" for c in candidates)
        )

    def expectations_batch(self, circuits, observables, *, options):
        from .batch import expectations_batch

        device = resolve_backend(self, "expectations", circuits, observables, options=options)
        return expectations_batch(circuits, observables, backend=device, **options.__dict__)

    def sample_batch(self, circuits, shots, *, options):
        from .batch import execute_batch

        device = resolve_backend(self, "sampling", circuits, options=options, shots=shots)
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


def _feedback_key(device, request, shots):
    # Shape/resource class, not parameter values or user data. Profile sources
    # distinguish layouts and calibration regimes chosen by a quote callback.
    width = max(c.num_qubits for c in request.circuits)
    digest = hashlib.sha256()
    for circuit in dict.fromkeys(request.circuits):
        for path, op in circuit.walk_operations():
            digest.update(
                repr((path, op.name, op.qubits, op.clbits, len(op.params), op.condition)).encode()
            )
    digest.update(repr(request.observables).encode())
    rows = len(request.circuits)
    return device.name, request.task, width, digest.hexdigest(), rows.bit_length(), shots


@dataclass(frozen=True)
class CostAwareBackend(PriorityBackend):
    """Rank feasible quotes, with QPU/GPU/CPU order only breaking equal scores.

    Missing QPU prices/noise/queue estimates never imply free or fast hardware.
    Without quotes, unconstrained local execution remains available and is
    explicitly labelled unprofiled. PriorityBackend retains the legacy policy.
    """

    policy: SchedulingPolicy = field(default_factory=SchedulingPolicy)
    workload: TrainingWorkload = field(default_factory=TrainingWorkload)
    profiles: object = field(default_factory=dict, repr=False, compare=False)
    feedback: object = field(default_factory=ExecutionFeedback, repr=False, compare=False)
    ledger: object = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.policy, SchedulingPolicy) or not isinstance(
            self.workload, TrainingWorkload
        ):
            raise ValidationError("Expected SchedulingPolicy and TrainingWorkload.")
        try:
            profiles = dict(self.profiles)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "profiles must map backend names to profiles or callbacks."
            ) from exc
        if any(
            not isinstance(k, str) or (not isinstance(v, ExecutionProfile) and not callable(v))
            for k, v in profiles.items()
        ):
            raise ValidationError("profiles must map backend names to profiles or callbacks.")
        if not isinstance(self.feedback, ExecutionFeedback):
            raise ValidationError("feedback must be ExecutionFeedback.")
        if self.ledger is not None and not isinstance(self.ledger, BudgetLedger):
            raise ValidationError("ledger must be BudgetLedger or None.")
        object.__setattr__(self, "profiles", MappingProxyType(profiles))

    def _quote(self, device, request):
        profile = self.profiles.get(device.name, getattr(device, "execution_profile", None))
        if callable(profile):
            profile = profile(request)
        if profile is None:
            raise ResourceError("No workload cost/quality profile; QPU capability is not a quote.")
        if not isinstance(profile, ExecutionProfile):
            raise ValidationError("Quote callbacks must return ExecutionProfile.")
        if device.capabilities.device_type == "QPU" and any(
            getattr(profile, name) is None
            for name in ("queue_seconds", "cost_per_second", "noise_bias")
        ):
            raise ResourceError("QPU quotes require explicit queue, pricing and noise estimates.")
        if self.ledger is not None and profile.cost_per_second is None:
            raise ResourceError("Spending admission requires explicit pricing.")
        exact = request.task in {"statevector", "density"} or (
            request.task == "expectations" and device.capabilities.exact_expectations
        )
        adjustable = request.task == "expectations" and callable(
            getattr(device, "with_shots", None)
        )
        cost = estimate_cost(
            profile, request, self.policy, exact=exact, adjustable_shots=adjustable
        )
        key = (*_feedback_key(device, request, cost.shots), profile.source)
        median, upper, count = self.feedback.correction(key)
        corrected = replace(
            cost,
            seconds=cost.seconds * median,
            upper_seconds=cost.upper_seconds * upper,
            observations=count,
        )
        check_cost(corrected, self.policy)
        if (
            self.ledger is not None
            and self.ledger.reserved + corrected.upper_cost > self.ledger.limit
        ):
            raise ResourceError("Forecast exceeds the remaining session spending budget.")
        return profile, corrected, key

    def native_plan(self, circuit, bindings, batch_shape, *, task, options):
        import math

        import numpy as np

        try:
            shape = np.broadcast_shapes(
                *(np.shape(v) for v in (bindings or {}).values()),
                () if batch_shape is None else tuple(batch_shape),
            )
        except ValueError as exc:
            raise ValidationError("Native binding shapes cannot be broadcast together.") from exc
        count = max(1, math.prod(shape))
        if count * 8 * (1 + len(bindings or {})) > options.max_memory_mb * 2**20:
            raise ResourceError("Native batch metadata exceeds the memory budget.")
        work = replace(self.workload, batch_size=max(self.workload.batch_size or 0, count))
        return self.plan([circuit], task=task, options=options, workload=work), count

    def plan(
        self,
        circuits,
        *,
        task="expectations",
        observables=(),
        options=None,
        shots=None,
        workload=None,
    ):
        from .ir import Circuit, integer

        circuits = (circuits,) if isinstance(circuits, Circuit) else tuple(circuits)
        observables = tuple(observables)
        if shots is not None:
            integer(shots, "shots", 1)
        work = self.workload if workload is None else workload
        if not isinstance(work, TrainingWorkload):
            raise ValidationError("workload must be TrainingWorkload.")
        options = ExecutionOptions() if options is None else options
        devices, admission = super().plan(
            circuits, task=task, observables=observables, options=options, _collect=True
        )
        request = SchedulingRequest(circuits, observables, task, options, work, shots)
        candidates = list(admission)
        by_name = {c.backend: i for i, c in enumerate(candidates)}
        quoted, costs, unprofiled = [], [], []
        constrained = (
            self.policy.require_profiles
            or self.ledger is not None
            or any(
                getattr(self.policy, k) is not None for k in ("max_cost", "max_seconds", "max_rmse")
            )
        )
        for device in devices:
            try:
                profile, cost, key = self._quote(device, request)
            except ResourceError as exc:
                missing = (
                    device.name not in self.profiles
                    and getattr(device, "execution_profile", None) is None
                )
                fallback = missing and device.capabilities.device_type != "QPU" and not constrained
                if fallback:
                    unprofiled.append(device)
                candidates[by_name[device.name]] = DispatchCandidate(
                    device.name, device.capabilities.device_type, fallback, str(exc)
                )
                continue
            candidates[by_name[device.name]] = DispatchCandidate(
                device.name,
                device.capabilities.device_type,
                True,
                "Feasible workload forecast; estimates are not measured speedup evidence.",
                cost.to_dict(),
            )
            quoted.append((device, profile, key))
            costs.append(cost)
        if costs:
            selected, frontier = select_cost(costs, self.policy)
            device, profile, key = quoted[selected]
            if (
                task == "expectations"
                and costs[selected].shots
                and callable(getattr(device, "with_shots", None))
            ):
                configured = device.with_shots(costs[selected].shots)
                device = (
                    QPUBackend(configured, device.available, device.execution_profile)
                    if isinstance(device, QPUBackend)
                    else configured
                )
            observed = _ObservedBackend(
                device, self, profile, task, observables, costs[selected].shots
            )
            return DispatchPlan(
                device.name,
                device.capabilities.device_type,
                task,
                tuple(candidates),
                observed,
                "pareto_cost_quality_forecast",
                tuple(quoted[i][0].name for i in frontier),
            )
        if unprofiled:
            device = unprofiled[0]
            return DispatchPlan(
                device.name,
                device.capabilities.device_type,
                task,
                tuple(candidates),
                device,
                "unprofiled_local_fallback",
            )
        raise ResourceError(
            "No cost/quality-feasible device. "
            + "; ".join(f"{c.backend}: {c.reason}" for c in candidates)
        )


class _ObservedBackend:
    def __init__(self, backend, scheduler, profile, task, observables, shots):
        self.backend, self.scheduler, self.profile = backend, scheduler, profile
        self.task, self.observables, self.shots = task, observables, shots

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def _invoke(self, method, circuits, *args, options, callback=None):
        request = SchedulingRequest(
            tuple(circuits),
            self.observables,
            self.task,
            options,
            TrainingWorkload(),
            self.shots or None,
        )
        # Refresh queue/calibration quotes at each actual submission, including
        # a ShiftPlan pinned for training. Never migrate its noise model mid-VJP.
        profile, cost, key = self.scheduler._quote(self.backend, request)
        if self.scheduler.ledger is not None:
            self.scheduler.ledger.reserve(cost.upper_cost)
        start = time.perf_counter()
        result = (
            callback()
            if callback is not None
            else getattr(self.backend, method)(*args, options=options)
        )
        raw = estimate_cost(
            profile,
            request,
            self.scheduler.policy,
            exact=self.task in {"statevector", "density"}
            or (self.task == "expectations" and self.backend.capabilities.exact_expectations),
        )
        self.scheduler.feedback.observe(key, raw.seconds, time.perf_counter() - start)
        return result

    def expectations_batch(self, circuits, observables, *, options):
        return self._invoke("expectations_batch", circuits, circuits, observables, options=options)

    def sample_batch(self, circuits, shots, *, options):
        return self._invoke("sample_batch", circuits, circuits, shots, options=options)

    def run(self, circuit, *, options):
        return self._invoke("run", [circuit], circuit, options=options)

    def run_density(self, circuit, *, options):
        return self._invoke("run_density", [circuit], circuit, options=options)

    def run_native(self, callback, circuit, rows, *, options):
        return self._invoke(None, (circuit,) * rows, options=options, callback=callback)


def configure_runtime(*, policy=None, workload=None, profiles=None, ledger=None):
    """Install the process-local automatic policy; no credentials/jobs are accessed."""
    from .backends import _LOCK, _REGISTRY

    backend = CostAwareBackend(
        policy=SchedulingPolicy() if policy is None else policy,
        workload=TrainingWorkload() if workload is None else workload,
        profiles={} if profiles is None else profiles,
        ledger=ledger,
    )
    with _LOCK:
        _REGISTRY["auto"] = backend
    return backend
