"""Budgeted readout correction and statistical zero-noise extrapolation.

These estimators do not submit circuits, clip results to physical bounds, or
promise reduced total error. Calibration drift and model bias remain distinct
from the reported shot uncertainty.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import islice

import numpy as np

from ..compute import compute_operation, get_compute_kernel
from ..errors import NumericalError, ResourceError, ValidationError
from ..ir import integer
from ..ir.observables import Hamiltonian, PauliString
from ..ir.parameters import real


def _array(value, name, max_memory_mb):
    try:
        value = np.asarray(value)
        if value.dtype.kind not in "fiu":
            raise ValueError
        get_compute_kernel().admit(
            2 * value.size, max_memory_mb, itemsize=max(8, value.dtype.itemsize)
        )
        if not np.isfinite(value).all():
            raise ValueError
        return value.astype(float, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{name} must contain finite real numbers.") from exc


def _limit(value, name):
    value = real(value)
    if value < 1:
        raise ValidationError(f"{name} must be at least one.")
    return value


def _counts(counts, width=None):
    if not isinstance(counts, Mapping) or not counts:
        raise ValidationError("counts must be a nonempty bit-string/count mapping.")
    total = 0
    for key, count in counts.items():
        if not isinstance(key, str) or not key or set(key) - {"0", "1"}:
            raise ValidationError("Counts require nonempty binary strings in q0-first order.")
        width = len(key) if width is None else width
        if len(key) != width:
            raise ValidationError("Count widths differ from the readout calibration.")
        total += integer(count, "count")
    if not 0 < total <= np.iinfo(np.int64).max:
        raise ValidationError("Total shots must be positive and fit in a signed 64-bit integer.")
    return width, total


@dataclass(frozen=True)
class MitigatedEstimate:
    values: np.ndarray
    standard_errors: np.ndarray | None
    shots: int
    sampling_overhead: np.ndarray


@dataclass(frozen=True)
class ExtrapolationResult:
    values: np.ndarray
    standard_errors: np.ndarray | None
    weights: np.ndarray
    residuals: np.ndarray
    condition_numbers: np.ndarray
    sampling_overhead: np.ndarray


@dataclass(frozen=True)
class ReadoutMitigation:
    """Independent 2x2 assignment matrices A[measured, prepared], per wire.

    Estimates diagonal observables directly, without constructing a 2**n
    probability vector or assignment matrix. Shot errors condition on the
    calibration; they exclude calibration uncertainty and correlated readout.
    sampling_overhead bounds squared single-shot scores, relative to the
    corresponding unmitigated observable's coefficient L1 bound.
    """

    assignment: object
    max_condition: float = 1e6
    max_overhead: float = 1e6
    max_memory_mb: float = 64
    max_work: int = 100_000_000
    _weights: object = field(init=False, repr=False, compare=False)

    @compute_operation
    def __post_init__(self):
        kernel = get_compute_kernel()
        assignment = _array(self.assignment, "assignment", self.max_memory_mb)
        if assignment.ndim != 3 or assignment.shape[1:] != (2, 2) or not len(assignment):
            raise ValidationError("assignment must have shape (num_qubits, 2, 2).")
        memory = kernel.admit(32 * len(assignment), self.max_memory_mb, itemsize=8)
        condition = _limit(self.max_condition, "max_condition")
        overhead = _limit(self.max_overhead, "max_overhead")
        work = integer(self.max_work, "max_work", 1)
        kernel.admit(0, memory, work=32 * len(assignment), max_work=work)
        if (
            np.any(assignment < 0)
            or np.any(assignment > 1)
            or not np.allclose(assignment.sum(axis=1), 1, atol=1e-10, rtol=0)
        ):
            raise ValidationError("Assignment columns must be normalized probabilities.")
        assignment = assignment.copy()
        assignment /= assignment.sum(axis=1, keepdims=True)
        _, singular, _ = kernel.svd(assignment)
        if np.any(singular[:, -1] <= 0) or np.any(singular[:, -1] < singular[:, 0] / condition):
            raise NumericalError("Readout calibration is singular or exceeds max_condition.")
        rhs = np.broadcast_to(np.array([1.0, -1.0])[..., None], (len(assignment), 2, 1))
        weights = kernel.solve(assignment.swapaxes(-1, -2), rhs)[..., 0]
        assignment.flags.writeable = weights.flags.writeable = False
        for name, value in (
            ("assignment", assignment),
            ("_weights", weights),
            ("max_condition", condition),
            ("max_overhead", overhead),
            ("max_memory_mb", memory),
            ("max_work", work),
        ):
            object.__setattr__(self, name, value)

    @classmethod
    @compute_operation
    def from_counts(cls, prepared_zero, prepared_one, **options):
        """Calibrate marginals from all-zero/all-one preparations.

        Assumes trusted state preparation and independent readout. Two global
        preparations cannot characterize correlated errors or preparation bias.
        """
        width, n0 = _counts(prepared_zero)
        _, n1 = _counts(prepared_one, width)
        kernel = get_compute_kernel()
        kernel.admit(32 * width, options.get("max_memory_mb", 64), itemsize=8)
        kernel.admit(
            0,
            options.get("max_memory_mb", 64),
            work=(len(prepared_zero) + len(prepared_one)) * width,
            max_work=integer(options.get("max_work", 100_000_000), "max_work", 1),
        )
        assignment = np.empty((width, 2, 2))
        for prepared, counts, shots in ((0, prepared_zero, n0), (1, prepared_one, n1)):
            ones = np.zeros(width)
            for key, count in counts.items():
                ones += (np.frombuffer(key.encode("ascii"), dtype=np.uint8) - 48) * (count / shots)
            assignment[:, 1, prepared] = ones
            assignment[:, 0, prepared] = 1 - ones
        # Summing many fractions can stray outside [0,1] by machine roundoff.
        return cls(np.clip(assignment, 0, 1), **options)

    @compute_operation
    def estimate(self, counts, observables):
        """Correct I/Z Pauli strings and Hamiltonians from one measurement basis."""
        width, shots = _counts(counts, len(self.assignment))
        if isinstance(observables, (PauliString, Hamiltonian)):
            observables = (observables,)
        else:
            try:
                observables = tuple(observables)
            except TypeError as exc:
                raise ValidationError("Provide diagonal Pauli observables.") from exc
        if not observables:
            raise ValidationError("Provide at least one diagonal observable.")
        terms, overhead, work = [], [], width
        max_weights = np.max(np.abs(self._weights), axis=1)
        for observable in observables:
            group = observable.terms if isinstance(observable, Hamiltonian) else (observable,)
            parsed, score_bound, original_bound = [], 0.0, 0.0
            for term in group:
                if (
                    not isinstance(term, PauliString)
                    or len(term.paulis) != width
                    or set(term.paulis) - {"I", "Z"}
                ):
                    raise ValidationError("Readout correction requires width-matched I/Z terms.")
                support = tuple(i for i, p in enumerate(term.paulis) if p == "Z")
                with np.errstate(over="ignore", invalid="ignore"):
                    bound = (
                        abs(term.coefficient) * np.prod(max_weights[list(support)])
                        if term.coefficient
                        else 0.0
                    )
                score_bound += bound
                original_bound += abs(term.coefficient)
                parsed.append((term.coefficient, support))
                work += len(support) + 2
            amplification = score_bound / original_bound if original_bound else 1.0
            if not np.isfinite(score_bound) or amplification > np.sqrt(self.max_overhead):
                raise ResourceError("Readout correction exceeds max_overhead.")
            terms.append(parsed)
            overhead.append(amplification**2)
        kernel = get_compute_kernel()
        # Include local weights, score temporaries and bit-string encoding.
        static = self.assignment.nbytes + self._weights.nbytes + 64 * len(terms)
        per_row = 32 * width + 32 * len(terms) + 64
        budget = kernel.admit(
            static + per_row,
            self.max_memory_mb,
            itemsize=1,
            work=4 * len(counts) * work,
            max_work=self.max_work,
        )
        block_size = min(4096, max(1, (int(budget * 2**20) - static) // per_row))
        mean, m2, total = np.zeros(len(terms)), np.zeros(len(terms)), 0.0
        iterator = iter(counts.items())
        while block := list(islice(iterator, block_size)):
            bits = np.frombuffer("".join(key for key, _ in block).encode("ascii"), np.uint8)
            bits = bits.reshape(len(block), width) - 48
            weights = self._weights[np.arange(width), bits]
            frequency = np.array([count for _, count in block], dtype=float)
            count = frequency.sum()
            if not count:
                continue
            scores = np.zeros((len(block), len(terms)))
            for j, group in enumerate(terms):
                for coefficient, support in group:
                    product = np.full(len(block), coefficient)
                    for wire in support:
                        product *= weights[:, wire]
                    scores[:, j] += product
            local_mean = kernel.einsum("b,bm->m", frequency / count, scores)
            centered = scores - local_mean
            local_m2 = kernel.einsum("b,bm,bm->m", frequency, centered, centered)
            delta = local_mean - mean
            combined = total + count
            m2 += local_m2 + delta**2 * (total * count / combined)
            mean += delta * (count / combined)
            total = combined
        errors = np.sqrt(np.maximum(m2, 0) / shots / (shots - 1)) if shots > 1 else None
        return MitigatedEstimate(
            kernel.finite(mean),
            None if errors is None else kernel.finite(errors),
            shots,
            np.asarray(overhead),
        )


@compute_operation
def zero_noise_extrapolate(
    scales,
    values,
    *,
    order=1,
    standard_errors=None,
    covariance=None,
    max_condition=1e8,
    max_overhead=1e6,
    max_memory_mb=64,
    max_work=100_000_000,
):
    """Fit a Chebyshev polynomial in physical noise scale and evaluate at zero.

    values has shape (scales, ...). Independent standard errors have that shape
    or shape (scales,); covariance is a shared positive-definite (scales, scales)
    matrix instead. Returned errors propagate the supplied measurement errors,
    never polynomial bias. Without them, standard_errors is None. The function
    neither creates scaled circuits nor submits jobs; callers must justify the
    noise-scaling procedure and polynomial approximation.
    """
    kernel = get_compute_kernel()
    scales = _array(scales, "scales", max_memory_mb)
    values = _array(values, "values", max_memory_mb)
    order = integer(order, "order", 1)
    condition_limit = _limit(max_condition, "max_condition")
    overhead_limit = _limit(max_overhead, "max_overhead")
    if (
        scales.ndim != 1
        or len(scales) < order + 1
        or np.any(scales <= 0)
        or len(np.unique(scales)) != len(scales)
        or not values.ndim
        or values.shape[0] != len(scales)
        or not values.size
    ):
        raise ValidationError("Distinct positive scales must match values and exceed order.")
    n, dimensions = len(scales), order + 1
    shape, outputs = values.shape[1:], values.size // n
    kernel.admit(
        12 * outputs * n * dimensions + 8 * n * n,
        max_memory_mb,
        itemsize=8,
        work=outputs * n * dimensions**2 + n**3,
        max_work=integer(max_work, "max_work", 1),
    )
    if standard_errors is not None and covariance is not None:
        raise ValidationError("Supply standard_errors or covariance, not both.")
    low, high = scales.min(), scales.max()
    coordinates = 2 * ((scales - low) / (high - low)) - 1
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        target = np.polynomial.chebyshev.chebvander(-1 - 2 * low / (high - low), order)[0]
    if not np.isfinite(target).all():
        raise NumericalError("Noise scales are too far from zero for stable extrapolation.")
    design = np.polynomial.chebyshev.chebvander(coordinates, order)
    y = values.reshape(n, outputs).T
    sigma = chol = None
    if standard_errors is not None:
        sigma = _array(standard_errors, "standard_errors", max_memory_mb)
        if sigma.shape == (n,):
            sigma = np.broadcast_to(sigma[:, None], (n, outputs))
        elif sigma.shape == values.shape:
            sigma = sigma.reshape(n, outputs)
        else:
            raise ValidationError("standard_errors must match scales or values shape.")
        if np.any(sigma <= 0):
            raise ValidationError("standard_errors must be strictly positive.")
        sigma = sigma.T
        divisor = sigma / sigma.max(axis=1, keepdims=True)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            matrix, data = design[None] / divisor[..., None], y / divisor
    elif covariance is not None:
        cov = _array(covariance, "covariance", max_memory_mb)
        if cov.shape != (n, n) or not np.allclose(cov, cov.T, atol=0, rtol=1e-12):
            raise ValidationError("covariance must be symmetric with shape (scales, scales).")
        scale = np.max(np.abs(cov))
        try:
            chol = np.linalg.cholesky(cov / scale) if scale > 0 else None
        except np.linalg.LinAlgError as exc:
            raise ValidationError("covariance must be positive definite.") from exc
        if chol is None:
            raise ValidationError("covariance must be positive definite.")
        matrix = np.broadcast_to(kernel.solve(chol, design), (outputs, n, dimensions))
        data = kernel.solve(chol, y.T).T
    else:
        matrix = np.broadcast_to(design, (outputs, n, dimensions))
        data = y
    if not np.isfinite(matrix).all() or not np.isfinite(data).all():
        raise NumericalError("Measurement uncertainties span an unstable numerical range.")
    u, singular, vh = kernel.svd(matrix)
    if np.any(singular[:, -1] <= 0) or np.any(singular[:, -1] < singular[:, 0] / condition_limit):
        raise NumericalError("Noise extrapolation is rank deficient or exceeds max_condition.")
    condition = singular[:, 0] / singular[:, -1]
    dual = kernel.einsum("rdi,i->rd", vh, target) / singular
    weights = kernel.einsum("rsd,rd->rs", u, dual)
    if sigma is not None:
        weights /= divisor
    elif chol is not None:
        weights = kernel.solve(chol.T, weights.T).T
    l1 = np.sum(np.abs(weights), axis=1)
    if not np.isfinite(l1).all() or np.any(l1 > np.sqrt(overhead_limit)):
        raise ResourceError("Noise extrapolation exceeds max_overhead.")
    coefficients = kernel.einsum("rsd,rs->rd", u, data) / singular
    coefficients = kernel.einsum("rdi,rd->ri", vh, coefficients)
    residuals = y - kernel.einsum("si,ri->rs", design, coefficients)
    estimate = kernel.einsum("rs,rs->r", weights, y)
    errors = None
    if sigma is not None:
        errors = kernel.norm(weights * sigma, axis=1)
    elif chol is not None:
        errors = kernel.norm(kernel.matmul(weights, chol), axis=1) * np.sqrt(scale)
    return ExtrapolationResult(
        kernel.finite(estimate).reshape(shape),
        None if errors is None else kernel.finite(errors).reshape(shape),
        kernel.finite(weights.T).reshape(values.shape),
        kernel.finite(residuals.T).reshape(values.shape),
        condition.reshape(shape),
        l1.reshape(shape) ** 2,
    )
