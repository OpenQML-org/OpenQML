"""Sorted native expectile roots and unrestricted-support W1 portfolio risk."""

from ..errors import NumericalError, ValidationError
from ..ir.parameters import real
from ._array import Ops, admission, rows
from .transport import _metric_values, transport_dual_norm


def _expectile_root(losses, level, probabilities, margin, max_memory_mb):
    level = real(level)
    if not 0.5 <= level < 1:
        raise ValidationError("Coherent expectile levels must be in [0.5,1).")
    ops = Ops(losses, probabilities, margin)
    losses, margin = ops.real(losses), ops.real(margin)
    if losses.ndim < 1 or losses.shape[-1] < 1:
        raise ValidationError("Expectiles require a nonempty scenario axis.")
    margin = ops.check(margin, margin >= 0, "Expectile transport margins must be nonnegative.")
    size = losses.shape[-1]
    admission(40 * rows(losses.shape), max_memory_mb)
    p = ops.probabilities(probabilities, size)
    prefix = ops.broadcast_shape(losses.shape[:-1], p.shape[:-1], margin.shape)
    admission(40 * rows(prefix) * size, max_memory_mb)
    losses, p = (ops.broadcast(v, prefix + (size,)) for v in (losses, p))
    margin = ops.broadcast(margin, prefix)
    order = ops.argsort(losses)
    ordered, mass = ops.gather(losses, order), ops.gather(p, order)

    def flip(v):
        return ops.xp.flip(v, dims=(-1,)) if ops.name == "torch" else ops.xp.flip(v, axis=-1)

    def suffix(v):
        return ops.cat((flip(ops.cumsum(flip(v)))[..., 1:], ops.xp.zeros_like(v[..., :1])), axis=-1)

    below, above = ops.cumsum(mass), suffix(mass)
    lower_sum, upper_sum = ops.cumsum(mass * ordered), suffix(mass * ordered)
    denominator = (1 - level) * below + level * above
    # Normalize the coefficients first: multiplying a subnormal loss by
    # 1-level before dividing would lose it. No global loss scale is needed,
    # so zero-mass extreme atoms cannot underflow the contributing observations.
    lower_weight, upper_weight = (1 - level) / denominator, level / denominator
    candidates = (
        lower_weight * lower_sum + upper_weight * upper_sum + upper_weight * margin[..., None]
    )
    # The increasing piecewise-affine root lies above min(loss) for level>=1/2.
    # The first interval whose upper endpoint admits its candidate is the root.
    eligible = ops.cat(
        (candidates[..., :-1] <= ordered[..., 1:], ops.xp.ones_like(ordered[..., :1], dtype=bool)),
        axis=-1,
    )
    index = (
        eligible.to(ops.xp.int64).argmax(dim=-1, keepdim=True)
        if ops.name == "torch"
        else ops.xp.argmax(eligible, axis=-1)[..., None]
    )
    root = ops.gather(candidates, index)[..., 0]
    return ops.check(
        root,
        ops.xp.isfinite(root),
        "Expectile risk exceeds finite precision.",
        error=NumericalError,
    )


def expectile(losses, *, level=0.95, probabilities=None, max_memory_mb=256):
    """Coherent weighted expectile, with exact sorted roots and native derivatives.

    The level is an asymmetric-loss weight, not a quantile confidence or tail
    probability. Derivatives are piecewise smooth away from active-set changes.
    """
    return _expectile_root(losses, level, probabilities, 0, max_memory_mb)


def wasserstein_expectile(
    weights,
    scenarios,
    *,
    level=0.95,
    probabilities=None,
    radius=0,
    transport_metric=None,
    max_memory_mb=256,
    cg_iterations=128,
    tolerance=1e-6,
):
    """Worst-case loss expectile over an unrestricted-support W1 ambiguity ball.

    Solve level*E[(L-q)+] -(1-level)*E[(q-L)+] + level*radius*dual_norm(w)=0.
    This bound is conservative for constrained supports or additional moments.
    """
    ops = Ops(weights, scenarios, probabilities, radius, *_metric_values(transport_metric))
    w, r, radius = ops.real(weights), ops.real(scenarios), ops.real(radius)
    if w.ndim < 1 or r.ndim < 2 or r.shape[-1] != w.shape[-1] or min(r.shape[-2:]) < 1:
        raise ValidationError("Portfolio and scenario asset axes must match and be nonempty.")
    radius = ops.check(radius, radius >= 0, "Wasserstein radii must be nonnegative.")
    p = ops.probabilities(probabilities, r.shape[-2])
    dual = transport_dual_norm(
        w,
        transport_metric,
        max_memory_mb=max_memory_mb,
        cg_iterations=cg_iterations,
        tolerance=tolerance,
    )
    prefix = ops.broadcast_shape(w.shape[:-1], r.shape[:-2], p.shape[:-1], radius.shape, dual.shape)
    admission(rows(prefix) * (4 * rows(r.shape[-2:]) + 40 * r.shape[-2]), max_memory_mb)
    losses = -ops.einsum("...sn,...n->...s", r, w)
    return _expectile_root(losses, level, p, radius * dual, max_memory_mb)
