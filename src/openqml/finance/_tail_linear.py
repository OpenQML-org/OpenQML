"""Projected matrix-free Hessians for joint path losses and rebalancing."""

from ..errors import ConvergenceError
from ._array import machine_epsilon
from ._temporal import temporal


def project(value, horizon, assets, ops):
    weights = value[..., :-1].reshape(value.shape[:-1] + (horizon, assets))
    weights = weights - ops.sum(weights, keepdims=True) / assets
    return ops.cat((weights.reshape(value.shape[:-1] + (-1,)), value[..., -1:]), axis=-1)


def changes(value, horizon, assets, ops):
    weights = value[..., :-1].reshape(value.shape[:-1] + (horizon, assets))
    result = temporal(weights, ops).reshape(value.shape[:-1] + (-1,))
    return ops.cat((result, ops.xp.zeros_like(value[..., -1:])), axis=-1)


def product(diagonal, factors, mass, turnover, value, horizon, assets, ops):
    tangent = project(value, horizon, assets, ops)
    dual = ops.einsum("...rd,...d->...r", factors, tangent)
    applied = diagonal * tangent + ops.einsum("...rd,...r->...d", factors, mass * dual)
    applied = applied + turnover[..., None] * changes(tangent, horizon, assets, ops)
    return project(applied, horizon, assets, ops) + value - tangent


def cg(operator, rhs, iterations, tolerance, ops):
    xp = ops.xp
    tolerance = max(tolerance, 128 * machine_epsilon(rhs, ops))
    scale = ops.stop(ops.max(xp.abs(rhs), keepdims=True))
    scale = xp.where(scale > 0, scale, 1)
    target = rhs / scale
    reference = ops.norm(target)
    squared = ops.sum(target * target)
    initial = xp.zeros_like(rhs), target, target, squared

    def advance(state, index):
        value, residual, direction, squared = state
        active = ops.norm(residual) > tolerance * reference
        applied = operator(direction)
        curvature = ops.sum(direction * applied)
        valid = active & (curvature > 0) & xp.isfinite(curvature)
        alpha = xp.where(valid, squared / xp.where(valid, curvature, 1), 0)
        value = value + alpha[..., None] * direction
        residual = residual - alpha[..., None] * applied
        if ops.name == "jax":
            import jax

            residual = jax.lax.cond(
                (index + 1) % 8 == 0, lambda: target - operator(value), lambda: residual
            )
        elif (index + 1) % 8 == 0:
            residual = target - operator(value)
        updated = ops.sum(residual * residual)
        beta = xp.where(valid, updated / xp.where(squared > 0, squared, 1), 0)
        return value, residual, residual + beta[..., None] * direction, updated

    if ops.name == "jax":
        import jax

        def condition(carry):
            i, state = carry
            return (i < iterations) & xp.any(ops.norm(state[1]) > tolerance * reference)

        result = jax.lax.while_loop(
            condition, lambda c: (c[0] + 1, advance(c[1], c[0])), (0, initial)
        )[1][0]
    else:
        state = initial
        for i in range(iterations):
            state = advance(state, i)
            if i % 8 == 7 and bool(xp.all(ops.norm(state[1]) <= tolerance * reference)):
                break
        result = state[0]
    residual = ops.norm(target - operator(result))
    valid = xp.isfinite(residual) & (residual <= tolerance * reference)
    return ops.check(
        result * scale,
        valid[..., None],
        "Path-tail Hessian solve failed its true residual; increase CG budget or precision.",
        error=ConvergenceError,
    )


def solve(diagonal, factors, mass, turnover, rhs, horizon, assets, plan, ops):
    if ops.name == "torch":
        from ._torch_tail import linear_solve

        return linear_solve(diagonal, factors, mass, turnover, rhs, horizon, assets, plan)

    def operator(x):
        return product(diagonal, factors, mass, turnover, x, horizon, assets, ops)

    if ops.name == "jax":
        import jax

        return jax.lax.custom_linear_solve(
            operator,
            rhs,
            solve=lambda _, x: cg(operator, x, plan.cg_iterations, plan.tolerance, ops),
            symmetric=True,
        )
    return cg(operator, rhs, plan.cg_iterations, plan.tolerance, ops)
