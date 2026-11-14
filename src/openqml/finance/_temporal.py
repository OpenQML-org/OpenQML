"""Matrix-free constrained horizon Hessians and implicit temporal solves."""

from ..compute.kernel import _torch_compiling
from ..errors import ConvergenceError
from ._array import machine_epsilon


def temporal(value, ops):
    zero = ops.xp.zeros_like(value[..., :1, :])
    difference = value - ops.cat((zero, value[..., :-1, :]), axis=-2)
    return difference - ops.cat((difference[..., 1:, :], zero), axis=-2)


def hessian(diagonal, factors, mass, turnover, value, ops):
    dual = ops.einsum("...hrn,...hn->...hr", factors, value)
    return (
        diagonal * value
        + ops.einsum("...hrn,...hr->...hn", factors, mass * dual)
        + turnover[..., None, None] * temporal(value, ops)
    )


def project(value, free, ops):
    count = ops.sum(free, keepdims=True)
    average = ops.sum(free * value, keepdims=True) / ops.xp.where(count > 0, count, 1)
    return free * (value - average)


def product(diagonal, factors, mass, turnover, free, value, ops):
    tangent = project(value, free, ops)
    return (
        project(hessian(diagonal, factors, mass, turnover, tangent, ops), free, ops)
        + value
        - tangent
    )


def _cg(
    diagonal, factors, mass, turnover, free, rhs, iterations, tolerance, ops, *, capturing=False
):
    xp = ops.xp
    eps = machine_epsilon(rhs, ops)
    tolerance = max(tolerance, 16 * eps)
    scale = ops.stop(ops.max(xp.abs(rhs), axis=(-2, -1), keepdims=True))
    scale = xp.where(scale > 0, scale, 1)
    target = rhs / scale
    reference = ops.norm(target, axis=(-2, -1))
    zero = xp.zeros_like(target)
    squared = ops.sum(target * target, axis=(-2, -1))
    direction = target.clone() if ops.name == "torch" else target
    initial = zero, target, direction, squared, xp.isfinite(squared)

    def advance(state, index):
        value, residual, direction, squared, healthy = state
        active = (ops.norm(residual, axis=(-2, -1)) > tolerance * reference) & healthy
        applied = product(diagonal, factors, mass, turnover, free, direction, ops)
        curvature = ops.sum(direction * applied, axis=(-2, -1))
        valid = xp.isfinite(curvature) & (curvature > 0) & xp.isfinite(squared)
        healthy = healthy & (~active | valid)
        active = active & valid
        alpha = xp.where(active, squared / xp.where(active, curvature, 1), 0)
        value = value + alpha[..., None, None] * direction
        residual = residual - alpha[..., None, None] * applied
        updated = ops.sum(residual * residual, axis=(-2, -1))
        beta = xp.where(active, updated / xp.where(squared > 0, squared, 1), 0)
        direction = residual + beta[..., None, None] * direction
        return value, residual, direction, updated, healthy

    if ops.name == "jax":
        import jax

        def condition(carry):
            i, state = carry
            return (i < iterations) & xp.any(
                (ops.norm(state[1], axis=(-2, -1)) > tolerance * reference) & state[-1]
            )

        value = jax.lax.while_loop(
            condition, lambda c: (c[0] + 1, advance(c[1], c[0])), (0, initial)
        )[1][0]
    elif ops.name == "torch" and _torch_compiling() and hasattr(xp, "while_loop"):

        def condition(i, value, residual, direction, squared, healthy):
            return (i < iterations) & xp.any(
                (ops.norm(residual, axis=(-2, -1)) > tolerance * reference) & healthy
            )

        def step(i, *state):
            return (i + 1, *advance(state, i))

        counter = xp.zeros((), dtype=xp.int64, device=rhs.device)
        value = xp.while_loop(condition, step, (counter, *initial))[1]
    else:
        state = initial
        for i in range(iterations):
            state = advance(state, i)
            if i % 8 == 7 and not capturing and not (ops.name == "torch" and _torch_compiling()):
                if bool(xp.all(ops.norm(state[1], axis=(-2, -1)) <= tolerance * reference)):
                    break
        value = state[0]
    residual = ops.norm(
        target - product(diagonal, factors, mass, turnover, free, value, ops), axis=(-2, -1)
    )
    valid = xp.isfinite(residual) & (residual <= tolerance * reference)
    result = value * scale
    if capturing:
        xp._assert_async(
            xp.all(valid), "[OpenQML:ConvergenceError] Temporal solve failed its true residual."
        )
        return xp.where(valid[..., None, None], result, float("nan"))
    return ops.check(
        result,
        valid[..., None, None],
        "Temporal solve failed its true residual; increase cg_iterations or precision.",
        error=ConvergenceError,
    )


def solve(diagonal, factors, mass, turnover, free, rhs, iterations, tolerance, ops):
    if ops.name == "torch":
        from ._torch_temporal import solve as torch_solve

        return torch_solve(diagonal, factors, mass, turnover, free, rhs, iterations, tolerance)
    if ops.name == "jax":
        import jax

        value = jax.lax.custom_linear_solve(
            lambda x: product(diagonal, factors, mass, turnover, free, x, ops),
            rhs,
            solve=lambda _, x: _cg(
                diagonal, factors, mass, turnover, free, x, iterations, tolerance, ops
            ),
            symmetric=True,
        )
        return ops.check(
            value,
            ops.xp.isfinite(value),
            "Temporal solve failed its true residual; increase cg_iterations or precision.",
            error=ConvergenceError,
        )
    return _cg(diagonal, factors, mass, turnover, free, rhs, iterations, tolerance, ops)
