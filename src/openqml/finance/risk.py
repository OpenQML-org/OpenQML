"""Weighted factor risk, exact fractional-tail CVaR and proper scenario scores."""

import math
from typing import NamedTuple

import numpy as np

from ..errors import ValidationError, _require_instance
from ..ir.parameters import real
from ._array import Ops, admission, rows


class FactorRisk(NamedTuple):
    mean: object
    factors: object
    factor_weights: object
    diagonal: object


def factor_risk(
    returns, probabilities=None, *, shrinkage=0.1, variance_floor=1e-8, max_memory_mb=256
):
    """PSD shrinkage covariance as diag(d) + F.T @ diag(p) @ F.

    Return samples have shape [...,S,N]. No dense N-by-N covariance is built.
    Keeping probabilities separate avoids sqrt(p) derivatives at zero mass.
    """
    ops = Ops(probabilities, returns)
    x = ops.real(returns)
    if x.ndim < 2 or min(x.shape[-2:]) < 1:
        raise ValidationError("Risk estimation needs returns[...,scenarios,assets].")
    shrinkage, floor = real(shrinkage), real(variance_floor)
    if not 0 <= shrinkage <= 1 or floor < 0:
        raise ValidationError("Shrinkage must lie in [0,1] and variance_floor be nonnegative.")
    p = ops.probabilities(probabilities, x.shape[-2])
    shape = ops.broadcast_shape(x.shape[:-2], p.shape[:-1])
    admission(8 * rows(shape) * x.shape[-2] * x.shape[-1], max_memory_mb)
    x = ops.broadcast(x, shape + x.shape[-2:])
    p = ops.broadcast(p, shape + (x.shape[-2],))
    mean = ops.sum(p[..., :, None] * x, axis=-2)
    centered = x - mean[..., None, :]
    variance = ops.sum(p[..., :, None] * centered * centered, axis=-2)
    return FactorRisk(mean, centered, (1 - shrinkage) * p, shrinkage * variance + floor)


def prepare_risk(risk, ops, shape):
    _require_instance(risk, FactorRisk, "risk")
    if not shape or shape[-1] < 1:
        raise ValidationError("Financial vectors require a nonempty asset axis.")
    mean, factors, weights, diagonal = (ops.real(v) for v in risk)
    n = shape[-1]
    if (
        factors.ndim < 2
        or factors.shape[-1] != n
        or weights.shape[-1:] != (factors.shape[-2],)
        or diagonal.shape[-1:] != (n,)
        or mean.shape[-1:] != (n,)
    ):
        raise ValidationError("Factor risk dimensions differ from the asset universe.")
    prefix = ops.broadcast_shape(
        shape[:-1], mean.shape[:-1], factors.shape[:-2], weights.shape[:-1], diagonal.shape[:-1]
    )
    mean = ops.broadcast(mean, prefix + (n,))
    factors = ops.broadcast(factors, prefix + factors.shape[-2:])
    weights = ops.broadcast(weights, prefix + weights.shape[-1:])
    diagonal = ops.broadcast(diagonal, prefix + (n,))
    weights = ops.check(weights, weights >= 0, "Factor weights must be nonnegative.")
    diagonal = ops.check(diagonal, diagonal >= 0, "Specific variances must be nonnegative.")
    return FactorRisk(mean, factors, weights, diagonal)


def covariance_matvec(risk, vector):
    _require_instance(risk, FactorRisk, "risk")
    ops = Ops(vector, *risk)
    vector = ops.real(vector)
    risk = prepare_risk(risk, ops, vector.shape)
    projected = ops.einsum("...rn,...n->...r", risk.factors, vector)
    return risk.diagonal * vector + ops.einsum(
        "...rn,...r->...n", risk.factors, risk.factor_weights * projected
    )


def portfolio_variance(weights, risk):
    _require_instance(risk, FactorRisk, "risk")
    ops = Ops(weights, *risk)
    weights = ops.real(weights)
    return ops.sum(weights * covariance_matvec(risk, weights))


def dense_covariance(risk, *, max_memory_mb=256):
    _require_instance(risk, FactorRisk, "risk")
    ops = Ops(*risk)
    risk = prepare_risk(risk, ops, ops.real(risk.mean).shape)
    n = risk.mean.shape[-1]
    admission(4 * rows(risk.mean.shape[:-1]) * (n * n + risk.factors.shape[-2] * n), max_memory_mb)
    matrix = ops.einsum("...ri,...r,...rj->...ij", risk.factors, risk.factor_weights, risk.factors)
    return matrix + ops.array(np.eye(n)) * risk.diagonal[..., None, :]


def expected_shortfall(losses, *, confidence=0.95, probabilities=None):
    """Upper-tail CVaR including fractional boundary mass; differentiable a.e."""
    confidence = real(confidence)
    if not 0 <= confidence < 1:
        raise ValidationError("confidence must lie in [0,1).")
    ops = Ops(losses, probabilities)
    losses = ops.real(losses)
    if losses.ndim < 1 or losses.shape[-1] < 1:
        raise ValidationError("Provide a nonempty scenario axis.")
    if probabilities is None:
        n = losses.shape[-1]
        mass = (1 - confidence) * n
        k = min(n, max(1, math.ceil(mass)))
        if confidence == 0:
            return ops.sum(losses) / n
        if k == 1:
            return ops.max(losses)
        if ops.name == "torch":
            tail = ops.xp.topk(losses, k, dim=-1, sorted=False).values
        elif ops.name == "jax":
            from jax import lax

            tail = lax.top_k(losses, k)[0]
        else:
            tail = np.partition(losses, n - k, axis=-1)[..., -k:]
        # Remove one selected boundary atom before summing the interior. This
        # avoids cancellation between large losses at fractional tail boundaries.
        index = ops.xp.argmin(tail, dim=-1) if ops.name == "torch" else ops.xp.argmin(tail, axis=-1)
        boundary = ops.gather(tail, index[..., None])[..., 0]
        positions = (
            ops.xp.arange(k, device=ops.like.device) if ops.name == "torch" else ops.xp.arange(k)
        )
        interior = ops.xp.where(positions == index[..., None], 0, tail)
        return (ops.sum(interior) + (mass - (k - 1)) * boundary) / mass
    p = ops.probabilities(probabilities, losses.shape[-1])
    shape = ops.broadcast_shape(losses.shape, p.shape)
    losses, p = ops.broadcast(losses, shape), ops.broadcast(p, shape)
    if confidence == 0:
        return ops.sum(p * losses)
    order = ops.argsort(losses)
    ordered, mass = ops.gather(losses, order), ops.gather(p, order)
    if ops.name == "torch":
        ordered, mass = (ops.xp.flip(v, dims=(-1,)) for v in (ordered, mass))
    else:
        ordered, mass = (ops.xp.flip(v, axis=-1) for v in (ordered, mass))
    prefix = ops.cat((ops.xp.zeros_like(mass[..., :1]), ops.cumsum(mass)[..., :-1]), axis=-1)
    admitted = ops.xp.minimum(mass, ops.xp.maximum((1 - confidence) - prefix, ops.array(0)))
    return ops.sum(admitted * ordered) / (1 - confidence)


def wasserstein_cvar(
    weights,
    scenarios,
    *,
    confidence=0.95,
    probabilities=None,
    radius=0,
    transport_metric=None,
    max_memory_mb=256,
    cg_iterations=128,
    tolerance=1e-6,
):
    """W1 robust-CVaR upper bound with a Euclidean or anisotropic ground norm.

    The margin is radius*dual_norm(w)/(1-confidence), Euclidean by default.
    The norm's sharpness assumes an
    unbounded return space; it is conservative for constrained supports.
    Radius calibration and distribution-shift coverage are caller responsibilities.
    """
    from .transport import _metric_values, transport_dual_norm

    confidence = real(confidence)
    if not 0 <= confidence < 1:
        raise ValidationError("Invalid CVaR confidence.")
    ops = Ops(weights, scenarios, probabilities, radius, *_metric_values(transport_metric))
    w, scenarios = ops.real(weights), ops.real(scenarios)
    if w.ndim < 1 or scenarios.ndim < 2 or scenarios.shape[-1] != w.shape[-1]:
        raise ValidationError("Scenario and portfolio asset dimensions must match.")
    radius = ops.real(radius)
    radius = ops.check(radius, radius >= 0, "Wasserstein radii must be nonnegative.")
    dual = transport_dual_norm(
        w,
        transport_metric,
        max_memory_mb=max_memory_mb,
        cg_iterations=cg_iterations,
        tolerance=tolerance,
    )
    p = None if probabilities is None else ops.real(probabilities)
    if p is not None and p.shape[-1:] != (scenarios.shape[-2],):
        raise ValidationError("Probability width differs from the scenario count.")
    prefix = ops.broadcast_shape(
        w.shape[:-1],
        scenarios.shape[:-2],
        radius.shape,
        dual.shape,
        () if p is None else p.shape[:-1],
    )
    admission(
        rows(prefix)
        * (
            4 * scenarios.shape[-2] * scenarios.shape[-1]
            + 12 * scenarios.shape[-2]
            + 8 * w.shape[-1]
        ),
        max_memory_mb,
    )
    losses = -ops.einsum("...sn,...n->...s", scenarios, w)
    return expected_shortfall(losses, confidence=confidence, probabilities=p) + radius * dual / (
        1 - confidence
    )


def quantile_loss(predictions, targets, quantiles, *, mask=None):
    ops = Ops(predictions, targets)
    predictions, targets = ops.real(predictions), ops.real(targets, finite=False)
    levels = np.asarray(quantiles, dtype=float)
    if (
        levels.ndim != 1
        or not len(levels)
        or not np.all((levels > 0) & (levels < 1))
        or np.any(np.diff(levels) <= 0)
        or predictions.shape[:-1] != targets.shape
        or predictions.shape[-1] != len(levels)
    ):
        raise ValidationError("Ordered quantiles and targets must match predictions[...,assets,Q].")
    valid = ops.xp.ones_like(targets) > 0 if mask is None else ops.array(mask, boolean=True)
    if valid.shape != targets.shape:
        raise ValidationError("Quantile masks must match target asset observations.")
    targets = ops.xp.where(valid, targets, 0)
    targets = ops.check(
        targets, ops.xp.isfinite(targets), "Active financial targets must be finite."
    )
    error = targets[..., None] - predictions
    level = ops.array(levels)
    values = ops.xp.maximum(level * error, (level - 1) * error)
    if mask is None:
        return ops.sum(values, axis=(-1, -2)) / (targets.shape[-1] * len(levels))
    values = ops.xp.where(valid[..., None], values, 0)
    return ops.sum(values, axis=(-1, -2)) / ops.maximum(
        ops.sum(valid.to(values.dtype) if ops.name == "torch" else valid.astype(values.dtype))
        * len(levels),
        1,
    )
