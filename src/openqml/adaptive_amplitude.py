"""Cost-aware amplitude sampling with time-uniform binomial confidence sets."""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from .amplitude import (
    AmplitudeEstimate,
    AmplitudeEstimationPlan,
    AmplitudeOracle,
    _aliases,
    _binomial_interval,
    _intersect,
)
from .errors import BackendError, ResourceError, ValidationError
from .ir import integer
from .parameters import real


def _confidence_sequence(successes, shots, log_threshold):
    """Invert a Jeffreys beta-binomial mixture e-process, including n=0."""
    if not shots:
        return 0.0, 1.0
    p = successes / shots
    log_beta = math.lgamma(successes + 0.5) + math.lgamma(shots - successes + 0.5)
    log_beta -= math.lgamma(shots + 1) + 2 * math.lgamma(0.5)
    log_mle = (successes * math.log(p) if successes else 0.0) + (
        (shots - successes) * math.log1p(-p) if successes < shots else 0.0
    )
    return _binomial_interval(successes, shots, log_threshold - log_beta + log_mle)


@dataclass(frozen=True)
class AdaptiveAmplitudeEstimate(AmplitudeEstimate):
    tolerance: float
    error_bound: float
    converged: bool
    rounds: int | None
    total_cost: float | None


@dataclass(frozen=True)
class AdaptiveAmplitudeEstimationPlan:
    """Sample until a requested probability error or explicit budget is reached.

    Depth selection uses calibrated contrast and information per supplied cost,
    restricted to monotone phase windows. The confidence sequence, rather than
    a local-information approximation, determines stopping and coverage.
    """

    tolerance: float = 0.01
    confidence: float = 0.95
    max_power: int = 32
    shots_per_batch: int = 64
    max_queries: int = 100_000
    max_rounds: int = 4096
    visibility: object = 1
    costs: object = None
    batch_cost: float = 0
    ratio: float = 1.45
    max_memory_mb: float = 128
    max_work: int = 100_000_000
    powers: tuple = field(init=False)

    def __post_init__(self):
        tolerance = real(self.tolerance)
        if not 0 < tolerance <= 1:
            raise ValidationError("Probability tolerance must lie in (0,1].")
        base = AmplitudeEstimationPlan.geometric(
            self.max_power,
            ratio=self.ratio,
            shots=1,
            confidence=self.confidence,
            visibility=self.visibility,
            max_memory_mb=self.max_memory_mb,
            max_work=self.max_work,
        )
        costs = (
            np.asarray([2 * m + 1 for m in base.powers], dtype=float)
            if self.costs is None
            else np.broadcast_to(np.asarray(self.costs, dtype=float), (len(base.powers),))
        )
        overhead = real(self.batch_cost)
        if not np.all(np.isfinite(costs)) or np.any(costs <= 0) or overhead < 0:
            raise ValidationError("Depth costs must be positive and batch overhead nonnegative.")
        object.__setattr__(self, "tolerance", tolerance)
        for name in ("confidence", "visibility", "max_memory_mb", "max_work"):
            object.__setattr__(self, name, getattr(base, name))
        object.__setattr__(self, "powers", base.powers)
        object.__setattr__(self, "max_power", base.powers[-1])
        object.__setattr__(self, "costs", tuple(float(v) for v in costs))
        object.__setattr__(self, "batch_cost", overhead)
        for name in ("shots_per_batch", "max_queries", "max_rounds"):
            object.__setattr__(self, name, integer(getattr(self, name), name, 1))

    @property
    def _threshold(self):
        return math.log(len(self.powers) / (1 - self.confidence))

    def _intervals(self, successes, shots, initial=None):
        feasible = [(0.0, math.pi / 2)] if initial is None else initial
        for s, n, m, v in zip(successes, shots, self.powers, self.visibility, strict=True):
            if n:
                feasible = _intersect(
                    feasible, _aliases(*_confidence_sequence(s, n, self._threshold), m, v)
                )
                if not feasible:
                    raise BackendError(
                        "Amplitude records contradict the calibrated time-uniform confidence model."
                    )
        return feasible

    def _result(self, successes, shots, feasible, rounds, total_cost):
        intervals = tuple((math.sin(a) ** 2, math.sin(b) ** 2) for a, b in feasible)
        lower, upper = intervals[0][0], intervals[-1][1]
        probability, error = (lower + upper) / 2, (upper - lower) / 2
        theta = math.asin(math.sqrt(probability))
        observed = [i for i, n in enumerate(shots) if n]
        log_likelihood = 0.0
        for i in observed:
            v, m, n, s = self.visibility[i], self.powers[i], shots[i], successes[i]
            p = (1 - v) / 2 + v * math.sin((2 * m + 1) * theta) ** 2
            p = min(1 - np.finfo(float).eps, max(np.finfo(float).tiny, p))
            log_likelihood += s * math.log(p) + (n - s) * math.log1p(-p)
        return AdaptiveAmplitudeEstimate(
            probability,
            intervals,
            self.confidence,
            tuple(self.powers[i] for i in observed),
            tuple(successes[i] for i in observed),
            tuple(shots[i] for i in observed),
            log_likelihood,
            self.tolerance,
            error,
            error <= self.tolerance,
            rounds,
            total_cost,
        )

    def fit_counts(self, records):
        """Read aggregated, possibly adaptively stopped marked-qubit records."""
        if not isinstance(records, Mapping) or set(records) - set(self.powers):
            raise ValidationError(
                "Records must map registered Grover powers to marked-qubit counts."
            )
        shots, successes = [0] * len(self.powers), [0] * len(self.powers)
        for i, m in enumerate(self.powers):
            record = records.get(m, {})
            if not isinstance(record, Mapping) or set(record) - {"0", "1"}:
                raise ValidationError("Measure only the marked qubit, using '0'/'1' counts.")
            zero, one = (
                integer(record.get("0", 0), "zero count"),
                integer(record.get("1", 0), "one count"),
            )
            shots[i], successes[i] = zero + one, one
        feasible = self._intervals(successes, shots)
        return self._result(successes, shots, feasible, None, None)

    def _choose(self, feasible, available, remaining):
        lo, hi = feasible[0][0], feasible[-1][1]
        theta = np.asarray([lo, (lo + hi) / 2, hi])
        chosen, best = None, -1.0
        for i, (m, visibility, cost) in enumerate(
            zip(self.powers, self.visibility, self.costs, strict=True)
        ):
            d = 2 * m + 1
            count = min(self.shots_per_batch, remaining // d)
            if not available[i] or not count or visibility == 0:
                continue
            branch = math.floor(2 * d * lo / math.pi)
            if d * hi > (branch + 1) * math.pi / 2 + 1e-13:
                continue
            if visibility == 1:
                information = 4.0 * d * d
            else:
                q = (1 - visibility) / 2 + visibility * np.sin(d * theta) ** 2
                information = float(
                    np.mean(
                        (visibility * d * np.sin(2 * d * theta)) ** 2
                        / np.maximum(q * (1 - q), 1e-15)
                    )
                )
            score = count * information / (cost * count + self.batch_cost)
            if score > best:
                chosen, best = (i, count), score
        return chosen

    def run(self, oracle, *, backend="auto", seed=None, **options):
        from .batch import execute_batch

        if not isinstance(oracle, AmplitudeOracle):
            raise ValidationError("Supply an AmplitudeOracle.")
        if self.visibility[0] <= 0:
            raise ResourceError(
                "Adaptive estimation needs informative calibrated depth-zero measurements."
            )
        seed = None if seed is None else integer(seed, "seed")
        options.setdefault("max_memory_mb", min(self.max_memory_mb, oracle.max_memory_mb))
        successes, shots, available = [0] * len(self.powers), [0] * len(self.powers), []
        for m in self.powers:
            try:
                oracle.resource_profile(m)
                available.append(True)
            except ResourceError:
                available.append(False)
        feasible, circuits = [(0.0, math.pi / 2)], {}
        queries, work, descriptor, total_cost = 0, 0, 0, 0.0
        rounds = 0
        for _ in range(self.max_rounds):
            current = self._result(successes, shots, feasible, rounds, total_cost)
            if current.converged:
                return current
            selected = self._choose(feasible, available, self.max_queries - queries)
            if selected is None:
                break
            i, count = selected
            m = self.powers[i]
            work += 8 * len(self.powers) + 4 * sum(power + 1 for power in self.powers)
            if work > self.max_work:
                break
            if i not in circuits:
                size = 512 * oracle.resource_profile(m)["operations"]
                if descriptor + size > min(self.max_memory_mb, oracle.max_memory_mb) * 2**20:
                    available[i] = False
                    continue
                circuits[i] = oracle.circuit(m)
                descriptor += size
            result = execute_batch(
                [circuits[i]],
                backend=backend,
                shots=count,
                seed=None if seed is None else seed + rounds,
                **options,
            )[0]
            record = result.counts
            if not isinstance(record, Mapping) or set(record) - {"0", "1"}:
                raise BackendError("Sampler did not return marked-qubit counts.")
            one, zero = (
                integer(record.get("1", 0), "one count"),
                integer(record.get("0", 0), "zero count"),
            )
            if one + zero != count:
                raise BackendError("Sampler returned an incorrect number of shots.")
            successes[i] += one
            shots[i] += count
            queries += count * (2 * m + 1)
            total_cost += count * self.costs[i] + self.batch_cost
            rounds += 1
            feasible = self._intervals(successes, shots, feasible)
        return self._result(successes, shots, feasible, rounds, total_cost)
