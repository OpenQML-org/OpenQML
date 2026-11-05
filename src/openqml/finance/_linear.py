"""Matrix-free SPD solves with implicit Torch/JAX derivatives."""

import numpy as np


def matmat(diagonal, factors, mass, value, ops):
    projected = ops.xp.einsum("...rn,...nc->...rc", factors, value)
    return diagonal[..., :, None] * value + ops.xp.einsum(
        "...rn,...rc->...nc", factors, mass[..., :, None] * projected
    )


def _woodbury(diagonal, factors, mass, rhs, ops):
    """Solve in factor space with finite derivatives at zero scenario mass."""
    inverse = 1 / diagonal
    # Balance the numerical solve before forming a Gram matrix. Implicit
    # derivatives still use the original operator with explicit scenario mass.
    weighted = factors * ops.xp.sqrt(ops.xp.abs(mass))[..., :, None]
    signs = ops.xp.sign(mass)
    scaled = weighted * inverse[..., None, :]
    core = (scaled @ weighted.swapaxes(-1, -2)) * signs[..., None, :]
    core = core + ops.array(np.eye(factors.shape[-2]))

    def apply(value):
        direct = inverse[..., :, None] * value
        dual = ops.xp.linalg.solve(core, weighted @ direct)
        return direct - scaled.swapaxes(-1, -2) @ (signs[..., :, None] * dual)

    result = apply(rhs)
    # Reuse the small factor system for a residual correction after the
    # subtraction in Woodbury, improving cancellation-sensitive float32 cases.
    return result + apply(rhs - matmat(diagonal, factors, mass, result, ops))


def _solve(diagonal, factors, mass, rhs, iterations, tolerance, ops):
    rank, assets = factors.shape[-2:]
    if rank == 0:
        return rhs / diagonal[..., :, None]
    if rank <= 64 and 4 * rank <= assets:
        return _woodbury(diagonal, factors, mass, rhs, ops)
    return _pcg(diagonal, factors, mass, rhs, iterations, tolerance, ops)


def _pcg(diagonal, factors, mass, rhs, iterations, tolerance, ops):
    xp = ops.xp
    rhs_scale = ops.stop(ops.max(xp.abs(rhs), axis=-2, keepdims=True))
    rhs_scale = xp.where(rhs_scale > 0, rhs_scale, 1)
    rhs = rhs / rhs_scale
    eps = np.finfo(np.float64 if "64" in str(rhs.dtype) else np.float32).eps
    tolerance = max(tolerance, 8 * eps)
    weighted = factors * xp.sqrt(xp.abs(mass))[..., :, None]
    preconditioner = diagonal + xp.einsum("...rn,...r->...n", weighted * weighted, xp.sign(mass))
    zero = xp.zeros_like(rhs)
    z = rhs / preconditioner[..., :, None]
    scale = ops.norm(rhs, axis=-2)
    rz = ops.sum(rhs * z, axis=-2)
    initial = (zero, rhs, z, rz)

    def step(state):
        x, residual, direction, product = state
        active = ops.norm(residual, axis=-2) > tolerance * scale
        applied = matmat(diagonal, factors, mass, direction, ops)
        denominator = ops.sum(direction * applied, axis=-2)
        alpha = xp.where(active, product / xp.where(active, denominator, 1), 0)
        x = x + direction * alpha[..., None, :]
        residual = residual - applied * alpha[..., None, :]
        z = residual / preconditioner[..., :, None]
        next_product = ops.sum(residual * z, axis=-2)
        beta = xp.where(active, next_product / xp.where(product != 0, product, 1), 0)
        return x, residual, z + direction * beta[..., None, :], next_product

    if ops.name == "jax":
        import jax

        def condition(carry):
            index, state = carry
            return (index < iterations) & xp.any(ops.norm(state[1], axis=-2) > tolerance * scale)

        result = jax.lax.while_loop(condition, lambda c: (c[0] + 1, step(c[1])), (0, initial))[1][0]
        return result * rhs_scale
    if (
        ops.name == "torch"
        and hasattr(xp, "while_loop")
        and getattr(xp.compiler, "is_compiling", lambda: False)()
    ):

        def condition(index, x, residual, direction, product):
            return (index < iterations) & xp.any(ops.norm(residual, axis=-2) > tolerance * scale)

        def advance(index, x, residual, direction, product):
            return (index + 1, *step((x, residual, direction, product)))

        counter = xp.zeros((), dtype=xp.int64, device=rhs.device)
        return xp.while_loop(condition, advance, (counter, *initial))[1] * rhs_scale
    state = initial
    for i in range(iterations):
        state = step(state)
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
            return _solve(diagonal, factors, mass, value, iterations, tolerance, ops)

        return jax.lax.custom_linear_solve(
            lambda value: matmat(diagonal, factors, mass, value, ops),
            rhs,
            solve=solve,
            transpose_solve=solve,
            symmetric=True,
        )
    return _solve(diagonal, factors, mass, rhs, iterations, tolerance, ops)
