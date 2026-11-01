"""Native matrix-free pullback geometry helpers; no imported ML runtime."""

from typing import NamedTuple

from .errors import ValidationError
from .ir import integer
from .parameters import real


class NativeNaturalGradientResult(NamedTuple):
    direction: dict
    converged: object
    iterations: object
    metric_products: int
    residual_norm: object
    damping: float


def solve_metric(matvec, gradient, ops, *, damping=1e-3, tolerance=1e-6, max_iterations=32):
    """Fixed-budget, masked CG; no device-to-host synchronization in iterations.

    All matvecs are executed even after logical convergence. The fresh residual
    reports the true damped-system error, not a recursive residual estimate.
    """
    damping, tolerance = real(damping), real(tolerance)
    if damping <= 0 or tolerance <= 0:
        raise ValidationError("Use positive damping and tolerance.")
    integer(max_iterations, "max_iterations", 1)
    if not gradient:
        raise ValidationError("Native metric solve requires a nonempty parameter gradient.")
    keys = tuple(gradient)

    def dot(a, b):
        return sum((a[k] * b[k]).sum() for k in keys)

    def apply(v):
        product = matvec(v)
        return {k: product[k] + damping * v[k] for k in keys}

    direction = {k: gradient[k] * 0 for k in keys}
    residual = dict(gradient)
    search = dict(residual)
    squared = dot(residual, residual)
    threshold = tolerance * ops.xp.maximum(ops.xp.sqrt(squared), ops.array(1, real_value=True))
    active = squared > threshold * threshold
    iterations = ops.array(0, real_value=True)

    def iteration(_, state):
        direction, residual, search, squared, active, iterations = state
        product = apply(search)
        curvature = dot(search, product)
        valid = active & ops.xp.isfinite(curvature) & (curvature > 0)
        denominator = ops.xp.where(valid, curvature, 1.0)
        step = ops.xp.where(valid, squared / denominator, 0.0)
        direction = {k: direction[k] + step * search[k] for k in keys}
        residual = {k: residual[k] - step * product[k] for k in keys}
        updated = dot(residual, residual)
        beta = ops.xp.where(valid, updated / ops.xp.where(squared > 0, squared, 1.0), 0.0)
        search = {k: residual[k] + beta * search[k] for k in keys}
        iterations = iterations + active
        active = valid & (updated > threshold * threshold)
        squared = updated
        return direction, residual, search, squared, active, iterations

    state = direction, residual, search, squared, active, iterations
    if ops.backend == "jax":
        from jax import lax

        state = lax.fori_loop(0, max_iterations, iteration, state)
    else:
        for i in range(max_iterations):
            state = iteration(i, state)
    direction, _, _, _, _, iterations = state
    product = apply(direction)
    error = {k: gradient[k] - product[k] for k in keys}
    norm = ops.xp.sqrt(dot(error, error))
    return NativeNaturalGradientResult(
        direction,
        ops.xp.isfinite(norm) & (norm <= threshold),
        iterations.to(dtype=ops.xp.int64)
        if ops.backend == "torch"
        else iterations.astype(ops.xp.int32),
        max_iterations + 1,
        norm,
        damping,
    )
