"""Coherent, phase-estimation-free quantum Monte Carlo and finite-shot intervals.

The estimated quantity is a probability, not its square root. Confidence sets
retain every trigonometric alias; likelihood optimization does not certify them.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from ..compute import get_compute_kernel, memory_budget
from ..errors import BackendError, ResourceError, ValidationError
from ..ir import Circuit, CircuitBuilder, integer
from ..ir.compiler import _check_circuit_limits, inverse
from ..ir.parameters import real


@dataclass(frozen=True)
class AmplitudeOracle:
    """Bound unitary A, a marked qubit, and clean reflection workspace.

    A always acts on the all-zero state. Amplification uses A and A inverse,
    never a simulation-derived probability or a controlled A operation.
    """

    preparation: Circuit
    good_qubit: int
    max_operations: int = 100_000
    max_memory_mb: float = 256
    _adjoint: Circuit = field(init=False, repr=False)
    _matrix_entries: int = field(init=False, repr=False)

    def __post_init__(self):
        limit = integer(self.max_operations, "max_operations", 1)
        _check_circuit_limits(self.preparation, max_operations=limit)
        good = integer(self.good_qubit, "good_qubit")
        memory = memory_budget(self.max_memory_mb)
        if (
            self.preparation.has_effects
            or self.preparation.num_clbits
            or self.preparation.parameters
            or good >= self.preparation.num_qubits
            or memory <= 0
        ):
            raise ValidationError("Amplitude preparation needs a bound unitary and a valid qubit.")
        matrix_entries = sum(
            len(op.matrix) ** 2 for op in self.preparation.operations if op.matrix is not None
        )
        if 512 * len(self.preparation.operations) + 32 * matrix_entries > memory * 2**20:
            raise ResourceError("Amplitude preparation exceeds descriptor memory before inversion.")
        adjoint = inverse(self.preparation)
        object.__setattr__(self, "good_qubit", good)
        object.__setattr__(self, "max_operations", limit)
        object.__setattr__(self, "max_memory_mb", memory)
        object.__setattr__(self, "_adjoint", adjoint)
        object.__setattr__(self, "_matrix_entries", matrix_entries)
        self.resource_profile(0)

    def resource_profile(self, power):
        m = integer(power, "Grover power")
        n, forward, backward = (
            self.preparation.num_qubits,
            len(self.preparation.operations),
            len(self._adjoint.operations),
        )
        reflection = 3 if n == 1 else 5 if n == 2 else 4 * n - 3
        count = forward + m * (1 + backward + reflection + forward) + 1
        if (
            count > self.max_operations
            or 512 * count + 32 * self._matrix_entries > memory_budget(self.max_memory_mb) * 2**20
        ):
            raise ResourceError("Amplitude circuit exceeds descriptor or operation budgets.")
        return dict(
            preparation_qubits=n,
            work_qubits=max(0, n - 2) if m else 0,
            operations=count,
            forward_queries=m + 1,
            inverse_queries=m,
            oracle_queries=2 * m + 1,
            reflection_queries=m,
        )

    def circuit(self, power, *, measure=True):
        profile = self.resource_profile(power)
        n = self.preparation.num_qubits
        builder = CircuitBuilder(
            n + profile["work_qubits"], global_phase=self.preparation.global_phase + math.pi * power
        )
        builder.extend(self.preparation.operations)
        for _ in range(power):
            builder.z(self.good_qubit).extend(self._adjoint.operations)
            for q in range(n):
                builder.x(q)
            if n == 1:
                builder.z(0)
            elif n == 2:
                builder.cz(0, 1)
            else:
                builder.ccx(0, 1, n)
                for q in range(2, n - 1):
                    builder.ccx(n + q - 2, q, n + q - 1)
                builder.cz(2 * n - 3, n - 1)
                for q in reversed(range(2, n - 1)):
                    builder.ccx(n + q - 2, q, n + q - 1)
                builder.ccx(0, 1, n)
            for q in range(n):
                builder.x(q)
            builder.extend(self.preparation.operations)
        if measure:
            builder.measure(self.good_qubit, 0)
        return builder.build()


@dataclass(frozen=True)
class AmplitudeEstimate:
    probability: float
    intervals: tuple
    confidence: float
    powers: tuple
    successes: tuple
    shots: tuple
    log_likelihood: float

    @property
    def interval(self):
        return self.intervals[0][0], self.intervals[-1][1]

    @property
    def oracle_queries(self):
        return sum(n * (2 * m + 1) for n, m in zip(self.shots, self.powers, strict=True))


def _merge(intervals):
    merged = []
    for lo, hi in sorted(intervals):
        if lo > hi:
            continue
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
        else:
            merged.append((lo, hi))
    return merged


def _intersect(first, second):
    output, i, j = [], 0, 0
    while i < len(first) and j < len(second):
        lo, hi = max(first[i][0], second[j][0]), min(first[i][1], second[j][1])
        if lo <= hi:
            output.append((lo, hi))
        if first[i][1] < second[j][1]:
            i += 1
        else:
            j += 1
    return _merge(output)


def _binomial_interval(successes, shots, threshold):
    p = successes / shots
    if p == 0:
        return 0.0, -math.expm1(-threshold / shots)
    if p == 1:
        return math.exp(-threshold / shots), 1.0

    def kl(q):
        return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))

    lo, hi = 0.0, p
    for _ in range(56):
        mid = (lo + hi) / 2
        if kl(mid) > threshold / shots:
            lo = mid
        else:
            hi = mid
    lower = lo  # outward rounding keeps the numerical interval conservative
    lo, hi = p, 1.0
    for _ in range(56):
        mid = (lo + hi) / 2
        if mid == 1 or kl(mid) > threshold / shots:
            hi = mid
        else:
            lo = mid
    return lower, hi


def _aliases(lower, upper, power, visibility):
    floor = (1 - visibility) / 2
    if upper < floor or lower > floor + visibility:
        return []
    if visibility == 0:
        return [(0.0, math.pi / 2)]
    lower = max(0.0, (lower - floor) / visibility)
    upper = min(1.0, (upper - floor) / visibility)
    a, b = math.asin(math.sqrt(lower)), math.asin(math.sqrt(upper))
    d = 2 * power + 1
    intervals = []
    for k in range(power + 1):
        for lo, hi in (
            (k * math.pi + a, k * math.pi + b),
            ((k + 1) * math.pi - b, (k + 1) * math.pi - a),
        ):
            lo, hi = max(0, lo / d), min(math.pi / 2, hi / d)
            if lo <= hi:
                intervals.append((lo, hi))
    return _merge(intervals)


@dataclass(frozen=True)
class AmplitudeEstimationPlan:
    """Independent-depth likelihood estimation with simultaneous KL confidence sets.

    visibility specifies a calibrated model P(1|m)=(1-v)/2+v*sin²((2m+1)*theta).
    Confidence assumes that model and independent binomial shots at fixed depths.
    """

    powers: tuple = (0, 1, 2, 3, 5, 7, 10, 15, 22, 32)
    shots: object = 128
    confidence: float = 0.95
    visibility: object = 1
    max_memory_mb: float = 128
    max_work: int = 100_000_000

    def __post_init__(self):
        powers = tuple(integer(m, "Grover power") for m in self.powers)
        if (
            not powers
            or powers[0] != 0
            or any(a >= b for a, b in zip(powers, powers[1:], strict=False))
        ):
            raise ValidationError("Grover depths must increase strictly, starting at zero.")
        raw = np.asarray(self.shots)
        if raw.dtype.kind not in "iu" or raw.ndim > 1:
            raise ValidationError("Shots must be a positive integer or integer depth vector.")
        shots = tuple(integer(int(n), "shots", 1) for n in np.broadcast_to(raw, (len(powers),)))
        visibility = np.broadcast_to(np.asarray(self.visibility, dtype=float), (len(powers),))
        confidence, memory = real(self.confidence), memory_budget(self.max_memory_mb)
        if (
            not 0 < confidence < 1
            or memory <= 0
            or not np.all(np.isfinite(visibility))
            or np.any((visibility < 0) | (visibility > 1))
        ):
            raise ValidationError("Invalid confidence, visibility or memory budget.")
        object.__setattr__(self, "powers", powers)
        object.__setattr__(self, "shots", shots)
        object.__setattr__(self, "visibility", tuple(float(v) for v in visibility))
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "max_memory_mb", memory)
        object.__setattr__(self, "max_work", integer(self.max_work, "max_work", 1))
        self.resource_profile()

    @classmethod
    def geometric(cls, max_power=32, *, ratio=1.45, shots=128, **options):
        maximum, ratio = integer(max_power, "max_power"), real(ratio)
        if ratio <= 1:
            raise ValidationError("A geometric schedule needs ratio > 1.")
        powers = [0]
        while powers[-1] < maximum:
            powers.append(min(maximum, max(powers[-1] + 1, round(max(1, powers[-1]) * ratio))))
            if len(powers) > 4096:
                raise ResourceError("Geometric schedule exceeds the depth-level budget.")
        return cls(tuple(powers), shots=shots, **options)

    def resource_profile(self, oracle=None):
        d = 2 * self.powers[-1] + 1
        if d > self.max_work:
            raise ResourceError("Amplitude depth exceeds the inference arithmetic budget.")
        information = sum(
            n * (2 * m + 1) ** 2 for m, n in zip(self.powers, self.shots, strict=True)
        )
        grid = max(1025, 64 * d + 1, math.ceil(math.pi * math.sqrt(information) / 2) + 1)
        work = grid * len(self.powers) * 128
        if (
            64 * grid * len(self.powers) > memory_budget(self.max_memory_mb) * 2**20
            or work > self.max_work
        ):
            raise ResourceError("Amplitude inference exceeds memory or arithmetic budgets.")
        profile = dict(
            powers=self.powers,
            shots=self.shots,
            total_shots=sum(self.shots),
            oracle_queries=sum(
                n * (2 * m + 1) for m, n in zip(self.powers, self.shots, strict=True)
            ),
            deepest_oracle_queries=d,
            likelihood_grid_points=grid,
            inference_workspace_bytes=64 * grid * len(self.powers),
        )
        if oracle is not None:
            if not isinstance(oracle, AmplitudeOracle):
                raise ValidationError("Supply an AmplitudeOracle.")
            circuits = [oracle.resource_profile(m) for m in self.powers]
            profile.update(
                executed_operations=sum(
                    n * c["operations"] for n, c in zip(self.shots, circuits, strict=True)
                ),
                peak_qubits=max(c["preparation_qubits"] + c["work_qubits"] for c in circuits),
                operation_basis="portable_IR_before_hardware_transpilation",
                excludes="routing_fault_tolerance_queue_calibration_and_process_RSS",
            )
            if 512 * sum(c["operations"] for c in circuits) > oracle.max_memory_mb * 2**20:
                raise ResourceError("Amplitude circuit batch exceeds descriptor memory budget.")
        return profile

    def _likelihood(self, theta, successes):
        d, v = 2 * np.asarray(self.powers) + 1, np.asarray(self.visibility)
        q = (1 - v) / 2 + v * np.sin(np.asarray(theta)[..., None] * d) ** 2
        q = np.clip(q, np.finfo(float).tiny, 1 - np.finfo(float).eps)
        return get_compute_kernel().matmul(np.log(q), successes) + get_compute_kernel().matmul(
            np.log1p(-q), np.asarray(self.shots) - successes
        )

    def fit(self, successes):
        raw = np.asarray(successes)
        if (
            raw.shape != (len(self.powers),)
            or raw.dtype.kind not in "iu"
            or np.any(raw > self.shots)
            or np.any(raw < 0)
        ):
            raise ValidationError(
                "Success counts must be nonnegative integers within scheduled shots."
            )
        counts = raw.astype(float)
        threshold = math.log(2 * len(self.powers) / (1 - self.confidence))
        feasible = [(0.0, math.pi / 2)]
        for s, n, m, v in zip(raw, self.shots, self.powers, self.visibility, strict=True):
            feasible = _intersect(
                feasible, _aliases(*_binomial_interval(int(s), n, threshold), m, v)
            )
            if not feasible:
                raise BackendError(
                    "Counts have no common confidence set under the calibrated amplitude model."
                )
        if not any(self.visibility):
            return AmplitudeEstimate(
                0.5,
                ((0.0, 1.0),),
                self.confidence,
                self.powers,
                tuple(int(s) for s in raw),
                self.shots,
                -sum(self.shots) * math.log(2),
            )
        grid_size = self.resource_profile()["likelihood_grid_points"]
        grid = np.linspace(0, math.pi / 2, grid_size)
        likelihood = self._likelihood(grid, counts)
        peaks = np.flatnonzero(
            (likelihood >= np.r_[-np.inf, likelihood[:-1]])
            & (likelihood >= np.r_[likelihood[1:], -np.inf])
        )
        if likelihood.max() == likelihood.min():
            peaks = np.asarray([grid_size // 2])
        lo, hi = grid[np.maximum(0, peaks - 1)], grid[np.minimum(grid_size - 1, peaks + 1)]
        # Refine every detected mode, retaining competing aliases rather than
        # starting a local optimizer at the shallow-depth estimate.
        ratio = (math.sqrt(5) - 1) / 2
        for _ in range(48):
            a, b = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
            left = self._likelihood(a, counts) >= self._likelihood(b, counts)
            lo, hi = np.where(left, lo, a), np.where(left, b, hi)
        candidates = np.r_[(lo + hi) / 2, 0, math.pi / 2]
        values = self._likelihood(candidates, counts)
        best = int(np.argmax(values))
        intervals = tuple((float(math.sin(a) ** 2), float(math.sin(b) ** 2)) for a, b in feasible)
        return AmplitudeEstimate(
            float(math.sin(candidates[best]) ** 2),
            intervals,
            self.confidence,
            self.powers,
            tuple(int(s) for s in raw),
            self.shots,
            float(values[best]),
        )

    def circuits(self, oracle):
        self.resource_profile(oracle)
        return tuple(oracle.circuit(m) for m in self.powers)

    def fit_counts(self, records):
        records = tuple(records)
        if len(records) != len(self.powers):
            raise ValidationError("Provide one count record per scheduled depth.")
        successes = []
        for record, shots in zip(records, self.shots, strict=True):
            if not isinstance(record, Mapping) or set(record) - {"0", "1"}:
                raise ValidationError("Amplitude records must measure only the marked qubit.")
            values = {k: integer(v, "shot count") for k, v in record.items()}
            if sum(values.values()) != shots:
                raise ValidationError("Observed shots differ from the fixed schedule.")
            successes.append(values.get("1", 0))
        return self.fit(np.asarray(successes, dtype=np.int64))

    def run(self, oracle, *, backend="auto", seed=None, **options):
        from .batch import execute_batch

        circuits, records = self.circuits(oracle), [None] * len(self.powers)
        options.setdefault("max_memory_mb", oracle.max_memory_mb)
        seed = None if seed is None else integer(seed, "seed")
        for index, shots in enumerate(dict.fromkeys(self.shots)):
            indices = [i for i, n in enumerate(self.shots) if n == shots]
            results = execute_batch(
                [circuits[i] for i in indices],
                backend=backend,
                shots=shots,
                seed=None if seed is None else seed + index * len(circuits),
                **options,
            )
            for i, result in zip(indices, results, strict=True):
                records[i] = result.counts
        return self.fit_counts(records)
