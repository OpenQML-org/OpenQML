"""Lazy Torch implicit solves with distinct eager-JVP and compiled reverse boundaries."""

import torch

from ._array import Ops
from ._linear import _solve, linear_solve


class _ImplicitFactorSolve(torch.autograd.Function):
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


class _ForwardFactorSolve(_ImplicitFactorSolve):
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


def solve(diagonal, factors, mass, rhs, iterations, tolerance):
    compiling = getattr(torch.compiler, "is_compiling", None) or torch._dynamo.is_compiling
    boundary = _ImplicitFactorSolve if compiling() else _ForwardFactorSolve
    return boundary.apply(diagonal, factors, mass, rhs, iterations, tolerance)
