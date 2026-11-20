"""Finite-support EVaR and its least-favourable scenario distribution."""

import math
from typing import NamedTuple

from ..errors import ConvergenceError, ValidationError
from ..ir.parameters import real
from ._array import Ops, admission, machine_epsilon, rows


class EntropicRiskResult(NamedTuple):
    value: object
    worst_probabilities: object
    relative_entropy: object


class EntropicAttribution(NamedTuple):
    risk: EntropicRiskResult
    marginal: object
    contributions: object
    deviation_contributions: object


def _tilt(penalty, p, log_p, ops):
    """Exponential tilt around the largest supported loss; zero mass stays zero."""
    xp = ops.xp
    logits = log_p - penalty
    shift = ops.max(logits, keepdims=True)
    weights = xp.exp(logits - shift)
    total = ops.sum(weights, keepdims=True)
    q = weights / total
    log_z = shift + xp.log(total)
    # Around the nominal distribution Z is nearly one. expm1/log1p preserve
    # its small deviation even when exp(-penalty) would round to one.
    delta = ops.sum(p * xp.expm1(-penalty), keepdims=True)
    close = delta > -0.5
    log_z = xp.where(close, xp.log1p(xp.where(close, delta, 0)), log_z)
    ratio = -penalty - log_z
    # Evaluate p*(exp(r)*r-expm1(r)) without cancellation near r=0.
    small = xp.abs(ratio) < 0.001
    r = xp.where(small, ratio, 0)
    series = p * r**2 * (0.5 + r * (1 / 3 + r * (1 / 8 + r / 30)))
    entropy = ops.sum(xp.where(small, series, q * ratio - (q - p)), keepdims=True)
    return q, log_z, entropy


def entropic_value_at_risk(losses, *, probabilities=None, confidence=0.95, max_memory_mb=256):
    """Maximize mean loss over KL(q || p) <= -log(1-confidence).

    The final axis indexes finite scenarios. Return risk, the maximizing
    distribution in input order, and its KL divergence in nats. At a saturated
    maximum, split mass among tied maxima in nominal proportions. Interior
    sensitivities use the implicit scalar optimality equation.
    """
    confidence = real(confidence)
    if not 0 <= confidence < 1:
        raise ValidationError("EVaR confidence must be in [0,1).")
    ops = Ops(losses, probabilities)
    xp, x = ops.xp, ops.real(losses)
    if not x.ndim or not x.shape[-1]:
        raise ValidationError("EVaR needs a nonempty scenario axis.")
    size = x.shape[-1]
    admission(12 * rows(x.shape), max_memory_mb)
    p = ops.probabilities(probabilities, size)
    prefix = ops.broadcast_shape(x.shape[:-1], p.shape[:-1])
    shape = prefix + (size,)
    admission(40 * rows(shape), max_memory_mb, work=64 * rows(shape))
    x, p = ops.broadcast(x, shape), ops.broadcast(p, shape)
    support = p > 0
    x = xp.where(support, x, 0)
    eps = machine_epsilon(x, ops)
    rho = -math.log1p(-confidence)
    if size == 1 or rho == 0:
        mean = ops.sum(p * x)
        return EntropicRiskResult(mean, p, xp.zeros_like(mean))

    peak = ops.max(xp.where(support, x, -math.inf), keepdims=True)
    top = support & (x == peak)
    mass = ops.sum(xp.where(top, p, 0), keepdims=True)
    top_entropy = ops.maximum(-xp.log(mass), 0)
    interior = top_entropy > rho
    top_q = xp.where(top, p, 0) / xp.where(interior, 1, mass)
    reference = ops.stop(peak)
    # Select the nominal-proportional subgradient at tied saturated maxima.
    top_value = reference + ops.sum(
        top_q * (xp.where(top, x, reference) - reference), keepdims=True
    )
    if ops.name == "numpy" and not xp.any(interior):
        return EntropicRiskResult(top_value[..., 0], top_q, top_entropy[..., 0])
    log_p = xp.where(support, xp.log(xp.where(support, p, 1)), -math.inf)
    positive = support & ~top
    # Scale each gap separately. One extreme low loss must not coalesce
    # distinct small losses near the maximum or overflow their differences.
    magnitude = xp.maximum(xp.abs(x), xp.abs(peak))
    _, exponent = xp.frexp(magnitude)
    scale = ops.stop(xp.where(magnitude > 0, xp.ldexp(xp.ones_like(magnitude), exponent - 1), 1))
    gap = (peak / scale) - (x / scale)
    log_scale = xp.log(scale)
    log_distance = xp.log(xp.where(positive, gap, 1)) + log_scale
    log_gap = ops.stop(log_distance)
    smallest = ops.min(xp.where(positive, log_gap, math.inf), keepdims=True)
    largest = ops.max(xp.where(positive, log_gap, -math.inf), keepdims=True)
    low = xp.where(interior, smallest, 0) - xp.log(80 + ops.stop(top_entropy))
    # KL at temperature T is at most range(loss)^2 / (8*T^2).
    high = xp.where(interior, largest, 0) - 0.5 * (math.log(8) + math.log(rho))
    fixed_p, fixed_log_p = ops.stop(p), ops.stop(log_p)

    def bisect(_, bounds):
        lo, hi = bounds
        mid = (lo + hi) * 0.5
        penalty = xp.where(
            positive, xp.exp(xp.minimum(log_gap - mid, ops.array(math.log(2048)))), 0
        )
        _, _, entropy = _tilt(penalty, fixed_p, fixed_log_p, ops)
        return xp.where(entropy > rho, mid, lo), xp.where(entropy > rho, hi, mid)

    iterations = 60 if eps < 1e-10 else 32
    if ops.name == "jax":
        import jax

        low, high = jax.lax.fori_loop(0, iterations, bisect, (low, high))
    else:
        for i in range(iterations):
            low, high = bisect(i, (low, high))
    log_temp = ops.stop(xp.where(interior, (low + high) * 0.5, 0))
    # For equal maxima the penalty is zero but its loss derivative is live.
    # Use each pair's scale for that derivative without forming 1/T.
    at_top = interior & top
    log_ratio = xp.where(at_top, log_scale - log_temp, 0)
    ratio = xp.exp(xp.minimum(log_ratio, ops.array(708.0 if eps < 1e-10 else 87.0)))
    penalty = xp.where(
        positive,
        xp.exp(xp.minimum(log_distance - log_temp, ops.array(math.log(2048)))),
        xp.where(at_top, gap, 0) * ratio,
    )
    # d KL / d log(T) = -Var_q(penalty). Two local Newton corrections
    # recover first and second implicit sensitivities of the tilted weights.
    # The converged forward temperature is unchanged by either correction.
    for _ in range(2):
        q, _, entropy = _tilt(penalty, p, log_p, ops)
        variance = ops.sum(q * (penalty - ops.sum(q * penalty, keepdims=True)) ** 2, keepdims=True)
        correction = (entropy - ops.stop(entropy)) / ops.stop(
            xp.where(interior & (variance > 0), variance, 1)
        )
        penalty = penalty * xp.exp(-xp.where(interior, correction, 0))
    q, _, entropy = _tilt(penalty, p, log_p, ops)
    q = xp.where(interior, q, top_q)
    value = xp.where(interior, ops.sum(q * x, keepdims=True), top_value)
    entropy = xp.where(interior, entropy, top_entropy)
    value = ops.check(
        value,
        xp.isfinite(value) & (~interior | (xp.abs(entropy - rho) <= 128 * eps * (1 + rho))),
        "EVaR dual did not reach its entropy tolerance.",
        error=ConvergenceError,
    )
    return EntropicRiskResult(value[..., 0], q, entropy[..., 0])
