"""Matrix-free SPD solves with implicit Torch/JAX derivatives."""

import numpy as np

from ..errors import ConvergenceError


def matmat(diagonal, factors, mass, value, ops):
    projected = ops.matmul(factors, value)
    return diagonal[..., :, None] * value + ops.matmul(
        factors.swapaxes(-1, -2), mass[..., :, None] * projected
    )


def _woodbury(diagonal, factors, mass, rhs, ops):
    """Solve in factor space with finite derivatives at zero scenario mass."""
    inverse = 1 / diagonal
    # Balance the numerical solve before forming a Gram matrix. Implicit
    # derivatives still use the original operator with explicit scenario mass.
    weighted = factors * ops.xp.sqrt(ops.xp.abs(mass))[..., :, None]
    signs = ops.xp.sign(mass)
    scaled = weighted * inverse[..., None, :]
    core = ops.matmul(scaled, weighted.swapaxes(-1, -2)) * signs[..., None, :]
    core = core + ops.array(np.eye(factors.shape[-2]))

    def apply(value):
        direct = inverse[..., :, None] * value
        dual = ops.solve(core, ops.matmul(weighted, direct))
        return direct - ops.matmul(scaled.swapaxes(-1, -2), signs[..., :, None] * dual)

    result = apply(rhs)
    # Reuse the small factor system for a residual correction after the
    # subtraction in Woodbury, improving cancellation-sensitive float32 cases.
    return result + apply(rhs - matmat(diagonal, factors, mass, result, ops))


def _solve(diagonal, factors, mass, rhs, iterations, tolerance, ops, *, return_info=False):
    rank, assets = factors.shape[-2:]
    if rank == 0:
        result = rhs / diagonal[..., :, None]
    elif rank <= 64 and 4 * rank <= assets:
        result = _woodbury(diagonal, factors, mass, rhs, ops)
    else:
        result = _pcg(diagonal, factors, mass, rhs, iterations, tolerance, ops)
    eps = np.finfo(np.float64 if "64" in str(rhs.dtype) else np.float32).eps
    residual = ops.norm(rhs - matmat(diagonal, factors, mass, result, ops), axis=-2)
    reference = ops.norm(rhs, axis=-2)
    valid = ops.xp.isfinite(residual) & (residual <= max(tolerance, 8 * eps) * reference)
    if return_info:
        return ops.xp.where(valid[..., None, :], result, float("nan")), valid
    return ops.check(
        result,
        valid[..., None, :],
        f"Factor solve failed its true residual with iteration budget {iterations} "
        f"and relative tolerance {tolerance:g}.",
        error=ConvergenceError,
    )


def _pcg(diagonal, factors, mass, rhs, iterations, tolerance, ops):
    xp = ops.xp
    rhs_scale = ops.stop(ops.max(xp.abs(rhs), axis=-2, keepdims=True))
    rhs_scale = xp.where(rhs_scale > 0, rhs_scale, 1)
    rhs = rhs / rhs_scale
    eps = np.finfo(np.float64 if "64" in str(rhs.dtype) else np.float32).eps
    tolerance = max(tolerance, 8 * eps)
    weighted = factors * xp.sqrt(xp.abs(mass))[..., :, None]
    preconditioner = diagonal + ops.sum(weighted * weighted * xp.sign(mass)[..., :, None], axis=-2)
    preconditioner = ops.check(
        preconditioner,
        xp.isfinite(preconditioner) & (preconditioner > 0),
        "Factor solve requires a finite positive preconditioner.",
        error=ConvergenceError,
    )
    zero = xp.zeros_like(rhs)
    z = rhs / preconditioner[..., :, None]
    scale = ops.norm(rhs, axis=-2)
    rz = ops.sum(rhs * z, axis=-2)
    initial = (zero, rhs, z, rz, xp.isfinite(rz) & (rz >= 0))

    def step(state, index):
        x, residual, direction, product, healthy = state
        active = (ops.norm(residual, axis=-2) > tolerance * scale) & healthy
        applied = matmat(diagonal, factors, mass, direction, ops)
        denominator = ops.sum(direction * applied, axis=-2)
        valid = xp.isfinite(denominator) & (denominator > 0) & xp.isfinite(product) & (product >= 0)
        healthy = healthy & (~active | valid)
        active = active & valid
        alpha = xp.where(active, product / xp.where(active, denominator, 1), 0)
        x = x + direction * alpha[..., None, :]
        residual = residual - applied * alpha[..., None, :]
        if ops.name == "jax":
            import jax

            residual = jax.lax.cond(
                (index + 1) % 8 == 0,
                lambda: rhs - matmat(diagonal, factors, mass, x, ops),
                lambda: residual,
            )
        elif ops.name != "torch" or not getattr(xp.compiler, "is_compiling", lambda: False)():
            if (index + 1) % 8 == 0:
                residual = rhs - matmat(diagonal, factors, mass, x, ops)
        z = residual / preconditioner[..., :, None]
        next_product = ops.sum(residual * z, axis=-2)
        beta = xp.where(active, next_product / xp.where(product != 0, product, 1), 0)
        return x, residual, z + direction * beta[..., None, :], next_product, healthy

    if ops.name == "jax":
        import jax

        def condition(carry):
            index, state = carry
            return (index < iterations) & xp.any(
                (ops.norm(state[1], axis=-2) > tolerance * scale) & state[4]
            )

        result = jax.lax.while_loop(
            condition, lambda c: (c[0] + 1, step(c[1], c[0])), (0, initial)
        )[1][0]
        return result * rhs_scale
    if (
        ops.name == "torch"
        and hasattr(xp, "while_loop")
        and getattr(xp.compiler, "is_compiling", lambda: False)()
    ):

        def condition(index, x, residual, direction, product, healthy):
            return (index < iterations) & xp.any(
                (ops.norm(residual, axis=-2) > tolerance * scale) & healthy
            )

        def advance(index, x, residual, direction, product, healthy):
            return (index + 1, *step((x, residual, direction, product, healthy), index))

        counter = xp.zeros((), dtype=xp.int64, device=rhs.device)
        return xp.while_loop(condition, advance, (counter, *initial))[1] * rhs_scale
    state = initial
    for i in range(iterations):
        state = step(state, i)
        if i % 8 == 7 and not (
            ops.name == "torch" and getattr(xp.compiler, "is_compiling", lambda: False)()
        ):
            if bool(xp.all(ops.norm(state[1], axis=-2) <= tolerance * scale)):
                break
    return state[0] * rhs_scale


def linear_solve(diagonal, factors, mass, rhs, iterations, tolerance, ops):
    if ops.name == "torch":
        from ._torch_linear import solve

        return solve(diagonal, factors, mass, rhs, iterations, tolerance)
    if ops.name == "jax":
        import jax

        def solve(_, value):
            return _solve(
                diagonal, factors, mass, value, iterations, tolerance, ops, return_info=True
            )

        result, converged = jax.lax.custom_linear_solve(
            lambda value: matmat(diagonal, factors, mass, value, ops),
            rhs,
            solve=solve,
            transpose_solve=solve,
            symmetric=True,
            has_aux=True,
        )
        return ops.check(
            result,
            converged[..., None, :],
            f"Factor solve failed its true residual with iteration budget {iterations} "
            f"and relative tolerance {tolerance:g}.",
            error=ConvergenceError,
        )
    return _solve(diagonal, factors, mass, rhs, iterations, tolerance, ops)
