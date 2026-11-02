"""Workload-level time, price and statistical-error admission without cloud probes.

Profiles are supplied by the resource owner, or by a read-only quote callback.
They describe a compiled workload, not advertised hardware peak throughput.
Predictions and observed wall times are deliberately separate.
"""

import math
import time
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass
from threading import RLock

from .errors import ResourceError, ValidationError
from .ir import integer
from .parameters import real


@dataclass(frozen=True)
class TrainingWorkload:
    """Forecast a sequence of dependent optimizer steps.

    batch_size is the total circuit rows per step (None uses the actual batch).
    gradient_evaluations counts shifted circuits per row, not parameter names.
    Classical optimizer/NN work is included but never submitted by this object.
    """

    batch_size: int | None = None
    steps: int = 1
    gradient_evaluations: int = 0
    native_gradient_evaluations: int | None = None
    classical_seconds_per_step: float = 0

    def __post_init__(self):
        if self.batch_size is not None:
            integer(self.batch_size, "batch_size", 1)
        integer(self.steps, "steps", 1)
        integer(self.gradient_evaluations, "gradient_evaluations", 0)
        if self.native_gradient_evaluations is not None:
            integer(self.native_gradient_evaluations, "native_gradient_evaluations", 0)
        if real(self.classical_seconds_per_step) < 0:
            raise ValidationError("classical_seconds_per_step must be nonnegative.")

    @classmethod
    def parameter_shift(cls, circuit, observables, *, parameters=None, **kwargs):
        from .gradients import ShiftPlan

        # Explicit local planning does not dispatch a primal or cloud request.
        plan = ShiftPlan(circuit, observables, parameters=parameters, backend="numpy")
        return cls(gradient_evaluations=plan.shift_evaluations, **kwargs)


@dataclass(frozen=True)
class SchedulingPolicy:
    """Hard constraints plus an augmented Chebyshev preference on the Pareto set.

    All prices must use the same caller-selected currency. RMSE is a readout
    forecast, not a bound on final training quality or a noise certificate.
    """

    max_cost: float | None = None
    max_seconds: float | None = None
    max_rmse: float | None = None
    time_weight: float = 1
    cost_weight: float = 0
    error_weight: float = 0
    risk_sigma: float = 2
    require_profiles: bool = False
    currency: str = "USD"

    def __post_init__(self):
        for name in ("max_cost", "max_seconds", "max_rmse"):
            value = getattr(self, name)
            if value is not None and real(value) < 0:
                raise ValidationError(f"{name} must be nonnegative.")
        for name in ("time_weight", "cost_weight", "error_weight", "risk_sigma"):
            if real(getattr(self, name)) < 0:
                raise ValidationError(f"{name} must be nonnegative.")
        if not self.time_weight + self.cost_weight + self.error_weight:
            raise ValidationError("At least one scheduling weight must be positive.")
        if type(self.require_profiles) is not bool:
            raise ValidationError("require_profiles must be a boolean.")
        if not isinstance(self.currency, str) or not self.currency.strip():
            raise ValidationError("currency must be a nonempty label.")


@dataclass(frozen=True)
class ExecutionProfile:
    """A quote for this circuit family, ISA, readouts and provider configuration.

    Queue/latency apply per job wave. Circuit and shot times apply per row;
    shot prices apply to each physical shot, including mitigation overhead.
    noise_bias/variance describe a normalized readout; coefficient norms scale
    them. A callback can supply tighter compiled, layout-aware values.
    """

    circuit_seconds: float
    queue_seconds: float | None = None
    latency_seconds: float = 0
    shot_seconds: float = 0
    cost_per_second: float | None = None
    cost_per_job: float = 0
    cost_per_circuit: float = 0
    cost_per_shot: float = 0
    noise_bias: float | None = None
    variance: float = 1
    shots: int = 1024
    min_shots: int = 1
    max_shots: int = 1_000_000
    max_batch_size: int = 64
    concurrency: int = 1
    measurement_groups: int = 1
    mitigation_overhead: float = 1
    relative_uncertainty: float = 0.25
    observed_at: float | None = None
    valid_for_seconds: float = 300
    source: str = "declared"
    currency: str = "USD"

    def __post_init__(self):
        for name in (
            "circuit_seconds",
            "queue_seconds",
            "latency_seconds",
            "shot_seconds",
            "cost_per_second",
            "cost_per_job",
            "cost_per_circuit",
            "cost_per_shot",
            "noise_bias",
            "variance",
            "relative_uncertainty",
            "valid_for_seconds",
        ):
            value = getattr(self, name)
            if value is None and name in {"queue_seconds", "cost_per_second", "noise_bias"}:
                continue
            if real(value) < 0:
                raise ValidationError(f"{name} must be nonnegative.")
        if real(self.mitigation_overhead) < 1:
            raise ValidationError("mitigation_overhead must be at least one.")
        for name in (
            "shots",
            "min_shots",
            "max_shots",
            "max_batch_size",
            "concurrency",
            "measurement_groups",
        ):
            integer(getattr(self, name), name, 1)
        if not self.min_shots <= self.shots <= self.max_shots:
            raise ValidationError("Profile shots must lie within min_shots/max_shots.")
        if self.observed_at is not None and real(self.observed_at) < 0:
            raise ValidationError("observed_at must be a nonnegative Unix timestamp.")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValidationError("A profile source is required.")
        if not isinstance(self.currency, str) or not self.currency.strip():
            raise ValidationError("A currency label is required.")

    def check_fresh(self, now=None):
        if self.observed_at is not None:
            age = (time.time() if now is None else now) - self.observed_at
            if age < -1 or age > self.valid_for_seconds:
                raise ResourceError("Execution profile is stale or has a future timestamp.")


@dataclass(frozen=True)
class SchedulingRequest:
    circuits: tuple
    observables: tuple
    task: str
    options: object
    workload: TrainingWorkload
    shots: int | None = None


@dataclass(frozen=True)
class ExecutionCost:
    seconds: float
    upper_seconds: float
    cost: float
    upper_cost: float
    estimated_rmse: float
    shots: int
    physical_shots: int
    circuit_evaluations: int
    provider_jobs: int
    source: str
    observations: int = 0

    def to_dict(self):
        return asdict(self)


def _norm(observables):
    from .observables import Hamiltonian

    return max(
        (
            sum(abs(t.coefficient) for t in (o.terms if isinstance(o, Hamiltonian) else (o,)))
            for o in observables
        ),
        default=1,
    )


def estimate_cost(profile, request, policy, *, exact=False, adjustable_shots=False):
    profile.check_fresh()
    norm = _norm(request.observables)
    if profile.currency != policy.currency:
        raise ResourceError("Profile currency differs from policy currency.")
    if (policy.max_cost is not None or policy.cost_weight) and profile.cost_per_second is None:
        raise ResourceError("Cost-constrained scheduling requires explicit pricing.")
    if policy.max_rmse is not None and not exact and profile.noise_bias is None:
        raise ResourceError("Quality-constrained scheduling requires a noise estimate.")
    bias = 0 if exact else norm * (profile.noise_bias or 0)
    variance = 0 if exact else norm * norm * profile.variance
    shots = 0 if exact else (profile.shots if request.shots is None else request.shots)
    if not exact and request.shots is None and policy.max_rmse is not None:
        slack = policy.max_rmse**2 - bias**2
        if slack <= 0 and (variance or bias > policy.max_rmse):
            raise ResourceError("Noise bias cannot meet max_rmse by adding shots.")
        needed = max(profile.min_shots, math.ceil(variance / slack) if variance else 1)
        if adjustable_shots:
            shots = needed
        elif shots < needed:
            raise ResourceError(
                "Configured shots cannot meet max_rmse; backend cannot adjust shots."
            )
    if shots and not profile.min_shots <= shots <= profile.max_shots:
        raise ResourceError("Required shots exceed the provider's configured shot range.")
    rmse = math.sqrt(bias**2 + (variance / shots if shots else 0))
    work = request.workload
    rows = max(len(request.circuits), work.batch_size or 0)
    groups = 1 if exact else profile.measurement_groups

    def phase(count):
        if not count:
            return 0, 0, 0, 0, 0
        count *= groups
        physical = math.ceil(count * shots * profile.mitigation_overhead)
        # All optimizer steps are sequential. Only jobs within a phase may
        # overlap, and a partial final wave cannot borrow future step work.
        jobs = math.ceil(count / profile.max_batch_size)
        full, tail = divmod(count, profile.max_batch_size)
        load = math.ceil(full / profile.concurrency) * profile.max_batch_size
        if tail and full % profile.concurrency == 0:
            load += tail
        waves = math.ceil(jobs / profile.concurrency)
        active = load * (
            profile.circuit_seconds + shots * profile.shot_seconds * profile.mitigation_overhead
        )
        wall = waves * ((profile.queue_seconds or 0) + profile.latency_seconds) + active
        # Billing may charge total device seconds even when jobs overlap.
        device_seconds = count * (
            profile.circuit_seconds + shots * profile.shot_seconds * profile.mitigation_overhead
        )
        cost = (
            jobs * profile.cost_per_job
            + count * profile.cost_per_circuit
            + physical * profile.cost_per_shot
            + device_seconds * (profile.cost_per_second or 0)
        )
        return wall, cost, count, physical, jobs

    forward = phase(rows)
    gradient_count = (
        work.native_gradient_evaluations
        if exact and work.native_gradient_evaluations is not None
        else work.gradient_evaluations
    )
    gradient = phase(rows * gradient_count)
    wall, cost, count, physical, jobs = (
        (a + b) * work.steps for a, b in zip(forward, gradient, strict=True)
    )
    wall += work.steps * work.classical_seconds_per_step
    margin = 1 + policy.risk_sigma * profile.relative_uncertainty
    return ExecutionCost(
        wall, wall * margin, cost, cost * margin, rmse, shots, physical, count, jobs, profile.source
    )


def check_cost(cost, policy):
    for name, value in (
        ("max_cost", cost.upper_cost),
        ("max_seconds", cost.upper_seconds),
        ("max_rmse", cost.estimated_rmse),
    ):
        limit = getattr(policy, name)
        if limit is not None and value > limit:
            raise ResourceError(f"Forecast exceeds {name}: {value:g} > {limit:g}.")


def select_cost(costs, policy):
    """Exact Pareto enumeration over the (usually small) available device set."""
    values = [(c.upper_seconds, c.upper_cost, c.estimated_rmse) for c in costs]
    frontier = [
        i
        for i, x in enumerate(values)
        if not any(
            all(a <= b for a, b in zip(y, x, strict=True))
            and any(a < b for a, b in zip(y, x, strict=True))
            for y in values
        )
    ]
    weights = (policy.time_weight, policy.cost_weight, policy.error_weight)
    lo = tuple(min(x[k] for x in values) for k in range(3))
    span = tuple(max(x[k] for x in values) - lo[k] for k in range(3))

    def score(i):
        v = [w * (values[i][k] - lo[k]) / span[k] if span[k] else 0 for k, w in enumerate(weights)]
        return max(v) + 1e-6 * sum(v), i

    return min(frontier, key=score), tuple(frontier)


class BudgetLedger:
    """Atomic admission of quoted spending across concurrent submitted calls.

    A failed or cancelled provider request retains its reservation because a
    failure does not establish that it was free. Reconcile only from billing.
    """

    def __init__(self, limit):
        self.limit = real(limit)
        if self.limit < 0:
            raise ValidationError("Budget limit must be nonnegative.")
        self._reserved, self._lock = 0.0, RLock()

    @property
    def reserved(self):
        with self._lock:
            return self._reserved

    def reserve(self, cost):
        value = real(cost)
        with self._lock:
            if value < 0 or self._reserved + value > self.limit:
                raise ResourceError("Session spending budget would be exceeded.")
            self._reserved += value


class ExecutionFeedback:
    """Bounded recent residual calibration; never submits calibration jobs.

    The empirical upper residual is a conservative ranking aid. Changing queue
    regimes violate exchangeability, so it is not a guaranteed confidence bound.
    """

    def __init__(self, capacity=128, window=64, max_age_seconds=900):
        self.capacity = integer(capacity, "capacity", 1)
        self.window = integer(window, "window", 1)
        self.max_age_seconds = real(max_age_seconds)
        if self.max_age_seconds <= 0:
            raise ValidationError("max_age_seconds must be positive.")
        self._rows, self._lock = OrderedDict(), RLock()

    def observe(self, key, predicted_seconds, elapsed_seconds):
        predicted, elapsed = real(predicted_seconds), real(elapsed_seconds)
        if predicted <= 0 or elapsed < 0:
            return
        with self._lock:
            rows = self._rows.setdefault(key, deque(maxlen=self.window))
            rows.append((time.monotonic(), elapsed / predicted))
            self._rows.move_to_end(key)
            while len(self._rows) > self.capacity:
                self._rows.popitem(last=False)

    def correction(self, key):
        with self._lock:
            rows = self._rows.get(key, ())
            ratios = sorted(r for t, r in rows if time.monotonic() - t <= self.max_age_seconds)
        if not ratios:
            return 1.0, 1.0, 0
        median = ratios[len(ratios) // 2]
        upper = ratios[min(len(ratios) - 1, math.ceil(0.95 * (len(ratios) + 1)) - 1)]
        return median, max(median, upper, 1.0), len(ratios)
