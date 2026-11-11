"""Native matrix-free pullback geometry helpers; no imported ML runtime."""

from typing import NamedTuple

from ..compute import _torch_compiling
from ..compute._norms import global_norm, stop
from ..errors import ConvergenceError, ValidationError
from ..ir import integer
from ..ir.parameters import real


class NativeNaturalGradientResult(NamedTuple):
    direction: dict
    converged: object
    iterations: object
    metric_products: object
    residual_norm: object
    damping: float


def solve_metric(
    matvec,
    gradient,
    ops,
    *,
    damping=1e-3,
    tolerance=1e-6,
    max_iterations=32,
    require_convergence=True,
):
    """Bounded CG with a fresh damped-system residual certificate.

    Eager calls stop at convergence; JAX skips inactive products in its graph.
    Compiled Torch retains masked iterations to preserve differentiation.
    """
    damping, tolerance = real(damping), real(tolerance)
    if damping <= 0 or tolerance <= 0:
        raise ValidationError("Use positive damping and tolerance.")
    integer(max_iterations, "max_iterations", 1)
    if type(require_convergence) is not bool:
        raise ValidationError("require_convergence must be a boolean.")
    if not gradient:
        raise ValidationError("Native metric solve requires a nonempty parameter gradient.")
    keys = tuple(gradient)
    scale = stop(global_norm(gradient.values(), ops.xp, ops.backend), ops.backend)
    scale = ops.xp.where((scale > 0) & ops.xp.isfinite(scale), scale, 1)
    gradient = {key: value / scale for key, value in gradient.items()}

    def dot(a, b):
        return sum((a[k] * b[k]).sum() for k in keys)

    def apply(v):
        product = matvec(v)
        return {k: product[k] + damping * v[k] for k in keys}

    direction = {k: gradient[k] * 0 for k in keys}
    residual = dict(gradient)
    search = dict(residual)
    squared = dot(residual, residual)
    threshold = ops.xp.maximum(tolerance * ops.xp.sqrt(squared), tolerance / scale)
    active = ops.xp.sqrt(squared) > threshold
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
        active = valid & (ops.xp.sqrt(updated) > threshold)
        squared = updated
        return direction, residual, search, squared, active, iterations

    state = direction, residual, search, squared, active, iterations
    if ops.backend == "jax":
        from jax import lax

        state = lax.fori_loop(
            0,
            max_iterations,
            lambda i, s: lax.cond(s[4], lambda: iteration(i, s), lambda: s),
            state,
        )
        products = (state[-1] + 1).astype(ops.xp.int32)
    else:
        compiling = ops.backend == "torch" and _torch_compiling()
        products = 1
        for i in range(max_iterations):
            if not compiling and not bool(state[4]):
                break
            state = iteration(i, state)
            products += 1
    direction, _, _, _, _, iterations = state
    product = apply(direction)
    error = {k: gradient[k] - product[k] for k in keys}
    relative_norm = global_norm(error.values(), ops.xp, ops.backend)
    norm = relative_norm * scale
    converged = ops.xp.isfinite(norm) & (relative_norm <= threshold)
    direction = {key: value * scale for key, value in direction.items()}
    if require_convergence:
        direction = {
            key: ops.check(
                value,
                converged,
                "Natural-gradient solve failed its true damped-system residual tolerance.",
                error=ConvergenceError,
            )
            for key, value in direction.items()
        }
    return NativeNaturalGradientResult(
        direction,
        converged,
        iterations.to(dtype=ops.xp.int64)
        if ops.backend == "torch"
        else iterations.astype(ops.xp.int32),
        products,
        norm,
        damping,
    )
