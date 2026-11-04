"""Exact sorted-breakpoint proximal maps for capped, cost-aware portfolios."""

import numpy as np

from ..errors import ValidationError
from ._array import Ops, admission, rows


def _prox(values, center, caps, penalty, budget, ops):
    """Solve .5||w-v||² + sum(a*|w-c|), 0<=w<=caps, sum(w)=budget.

    Clipped soft-thresholding is a sum of two bounded ramps per asset. Sorting
    their four breakpoints finds the equality multiplier in O(N log N) work
    and linear space; no iterative bisection or asset-by-asset active set.
    """
    xp = ops.xp
    values = values - ops.sum(values, keepdims=True) / values.shape[-1]
    center = ops.clip(center, 0, caps)
    heights = ops.cat((center, caps - center), axis=-1)
    ends = ops.cat((values + penalty, values - penalty - center), axis=-1)
    starts = ends - heights
    knots = ops.cat((starts, ends), axis=-1)
    changes = ops.cat((-xp.ones_like(starts), xp.ones_like(ends)), axis=-1)
    order = ops.argsort(knots)
    knots, changes = ops.gather(knots, order), ops.gather(changes, order)
    slope = ops.cumsum(changes)
    intercept = ops.sum(caps, keepdims=True) - ops.cumsum(changes * knots)
    at_knot = intercept + slope * knots
    # Floating-point cancellation at the last knot cannot leave gather out of
    # range; the candidate's mass is checked by the caller/optimizer as well.
    eps = np.finfo(np.float64 if "64" in str(values.dtype) else np.float32).eps
    roundoff = 8 * eps * knots.shape[-1] * (1 + ops.max(xp.abs(knots)))
    reached = at_knot <= (budget + roundoff)[..., None]
    count = (
        xp.argmax(reached.to(xp.int64), dim=-1)
        if ops.name == "torch"
        else xp.argmax(reached, axis=-1)
    )
    previous = ops.clip(count - 1, 0, knots.shape[-1] - 1)
    previous = previous.to(xp.int64) if ops.name == "torch" else previous.astype("int32")
    m = ops.gather(slope, previous[..., None])[..., 0]
    b = ops.gather(intercept, previous[..., None])[..., 0]
    endpoint = ops.gather(knots, count[..., None])[..., 0]
    multiplier = xp.where(m < 0, (budget - b) / xp.where(m < 0, m, -1), endpoint)
    difference = values - multiplier[..., None] - center
    soft = xp.sign(difference) * xp.maximum(xp.abs(difference) - penalty, ops.array(0))
    result = ops.clip(center + soft, 0, caps)
    result = xp.where((budget <= 0)[..., None], 0, result)
    return xp.where((budget >= ops.sum(caps))[..., None], caps, result)


def cost_aware_projection(values, *, previous=None, caps=1, penalty=0, budget=1, max_memory_mb=256):
    ops = Ops(values)
    values = ops.real(values)
    if values.ndim < 1 or values.shape[-1] < 1:
        raise ValidationError("Portfolio vectors require a nonempty asset axis.")
    previous = ops.real(0 if previous is None else previous)
    caps, penalty, budget = (ops.real(v) for v in (caps, penalty, budget))
    shape = np.broadcast_shapes(values.shape, previous.shape, caps.shape, penalty.shape)
    prefix = np.broadcast_shapes(shape[:-1], budget.shape)
    shape = prefix + (shape[-1],)
    admission(40 * rows(shape), max_memory_mb)
    values, previous, caps, penalty = (
        ops.broadcast(v, shape) for v in (values, previous, caps, penalty)
    )
    budget = ops.broadcast(budget, prefix)
    caps = ops.check(caps, caps >= 0, "Position caps must be nonnegative.")
    penalty = ops.check(penalty, penalty >= 0, "Trading penalties must be nonnegative.")
    eps = np.finfo(np.float64 if "64" in str(values.dtype) else np.float32).eps
    capacity = ops.sum(caps)
    budget = ops.check(
        budget,
        (budget >= 0) & (budget <= capacity + 8 * eps * values.shape[-1]),
        "Portfolio budget exceeds available position capacity.",
    )
    return _prox(values, previous, caps, penalty, ops.xp.minimum(budget, capacity), ops)
