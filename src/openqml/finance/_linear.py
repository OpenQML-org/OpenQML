"""Matrix-free SPD solves with implicit Torch/JAX derivatives."""

import numpy as np

from ._array import Ops

_TORCH_SOLVE = None


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
    eps = np.finfo(np.float64 if "64" in str(rhs.dtype) else np.float32).eps
    tolerance = max(tolerance, 8 * eps)
    weighted = factors * xp.sqrt(xp.abs(mass))[..., :, None]
    preconditioner = diagonal + xp.einsum(
        "...rn,...r->...n", weighted * weighted, xp.sign(mass)
    )
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

        return jax.lax.while_loop(condition, lambda c: (c[0] + 1, step(c[1])), (0, initial))[1][0]
    state = initial
    for i in range(iterations):
        state = step(state)
        if i % 8 == 7 and not (
            ops.name == "torch" and getattr(xp.compiler, "is_compiling", lambda: False)()
        ):
            if bool(xp.all(ops.norm(state[1], axis=-2) <= tolerance * scale)):
                break
    return state[0]


def _torch_class():
    global _TORCH_SOLVE
    if _TORCH_SOLVE is not None:
        return _TORCH_SOLVE
    import torch

    class ImplicitFactorSolve(torch.autograd.Function):
        @staticmethod
        def forward(diagonal, factors, mass, rhs, iterations, tolerance):
            return _solve(diagonal, factors, mass, rhs, iterations, tolerance, Ops(diagonal))

        @staticmethod
        def setup_context(ctx, inputs, output):
            diagonal, factors, mass, _, ctx.iterations, ctx.tolerance = inputs
            ctx.save_for_backward(diagonal, factors, mass, output)
            ctx.save_for_forward(diagonal, factors, mass, output)

        @staticmethod
        def backward(ctx, gradient):
            diagonal, factors, mass, solution = ctx.saved_tensors
            ops = Ops(diagonal)
            adjoint = linear_solve(
                diagonal, factors, mass, gradient, ctx.iterations, ctx.tolerance, ops
            )
            left = torch.einsum("...rn,...nc->...rc", factors, adjoint)
            right = torch.einsum("...rn,...nc->...rc", factors, solution)
            grad_diagonal = -torch.sum(adjoint * solution, dim=-1)
            grad_mass = -torch.sum(left * right, dim=-1)
            grad_factors = -mass[..., :, None] * (
                torch.einsum("...rc,...nc->...rn", left, solution)
                + torch.einsum("...rc,...nc->...rn", right, adjoint)
            )
            return grad_diagonal, grad_factors, grad_mass, adjoint, None, None

        @staticmethod
        def jvp(ctx, d_diagonal, d_factors, d_mass, d_rhs, _iterations, _tolerance):
            diagonal, factors, mass, solution = ctx.saved_tensors
            d_diagonal = torch.zeros_like(diagonal) if d_diagonal is None else d_diagonal
            d_factors = torch.zeros_like(factors) if d_factors is None else d_factors
            d_mass = torch.zeros_like(mass) if d_mass is None else d_mass
            d_rhs = torch.zeros_like(solution) if d_rhs is None else d_rhs
            projected = torch.einsum("...rn,...nc->...rc", factors, solution)
            tangent = d_diagonal[..., :, None] * solution
            tangent += torch.einsum("...rn,...rc->...nc", d_factors, mass[..., :, None] * projected)
            tangent += torch.einsum("...rn,...rc->...nc", factors, d_mass[..., :, None] * projected)
            tangent += torch.einsum(
                "...rn,...rc->...nc",
                factors,
                mass[..., :, None] * torch.einsum("...rn,...nc->...rc", d_factors, solution),
            )
            return linear_solve(
                diagonal,
                factors,
                mass,
                d_rhs - tangent,
                ctx.iterations,
                ctx.tolerance,
                Ops(diagonal),
            )

    _TORCH_SOLVE = ImplicitFactorSolve
    return _TORCH_SOLVE


def linear_solve(diagonal, factors, mass, rhs, iterations, tolerance, ops):
    if ops.name == "torch":
        return _torch_class().apply(diagonal, factors, mass, rhs, iterations, tolerance)
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
