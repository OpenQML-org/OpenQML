"""Torch implicit horizon solves without an optimization-iteration tape."""

import torch

from ..compute.kernel import _torch_compiling
from ._array import Ops
from ._temporal import _cg, hessian, project, temporal
from ._temporal import solve as native_solve


def _implicit_backward(inputs, gradient, iterations, tolerance, *, compiled):
    diagonal, factors, mass, turnover, free, solution = inputs
    ops = Ops(diagonal)
    if compiled:
        dual = _CompiledTemporal.apply(
            diagonal, factors, mass, turnover, free, gradient, iterations, tolerance
        )
    else:
        dual = native_solve(
            diagonal,
            factors,
            mass,
            turnover,
            free,
            gradient,
            iterations,
            tolerance,
            ops,
        )
    left, right = project(dual, free, ops), project(solution, free, ops)
    fl = torch.einsum("...hrn,...hn->...hr", factors, left)
    fr = torch.einsum("...hrn,...hn->...hr", factors, right)
    grad_factors = -mass[..., None] * (
        fl[..., None] * right[..., None, :] + fr[..., None] * left[..., None, :]
    )
    grad_turnover = -(left * temporal(right, ops)).sum(dim=(-2, -1))
    return -left * right, grad_factors, -fl * fr, grad_turnover, None, dual, None, None


class _TemporalSolve(torch.autograd.Function):
    @staticmethod
    def forward(diagonal, factors, mass, turnover, free, rhs, iterations, tolerance):
        return _cg(
            diagonal, factors, mass, turnover, free, rhs, iterations, tolerance, Ops(diagonal)
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        diagonal, factors, mass, turnover, free, _, ctx.iterations, ctx.tolerance = inputs
        ctx.save_for_backward(diagonal, factors, mass, turnover, free, output)
        ctx.save_for_forward(diagonal, factors, mass, turnover, free, output)

    @staticmethod
    def backward(ctx, gradient):
        return _implicit_backward(
            ctx.saved_tensors, gradient, ctx.iterations, ctx.tolerance, compiled=False
        )


class _ForwardTemporal(_TemporalSolve):
    @staticmethod
    def jvp(ctx, d_diagonal, d_factors, d_mass, d_turnover, _free, d_rhs, _iterations, _tolerance):
        diagonal, factors, mass, turnover, free, solution = ctx.saved_tensors
        ops = Ops(diagonal)
        right = project(solution, free, ops)
        d_rhs = torch.zeros_like(solution) if d_rhs is None else d_rhs
        d_diagonal = torch.zeros_like(diagonal) if d_diagonal is None else d_diagonal
        d_factors = torch.zeros_like(factors) if d_factors is None else d_factors
        d_mass = torch.zeros_like(mass) if d_mass is None else d_mass
        d_turnover = torch.zeros_like(turnover) if d_turnover is None else d_turnover
        tangent = hessian(d_diagonal, factors, d_mass, d_turnover, right, ops)
        tangent = tangent + torch.einsum(
            "...hrn,...hr->...hn",
            d_factors,
            mass * torch.einsum("...hrn,...hn->...hr", factors, right),
        )
        tangent = tangent + torch.einsum(
            "...hrn,...hr->...hn",
            factors,
            mass * torch.einsum("...hrn,...hn->...hr", d_factors, right),
        )
        return native_solve(
            diagonal,
            factors,
            mass,
            turnover,
            free,
            d_rhs - project(tangent, free, ops),
            ctx.iterations,
            ctx.tolerance,
            ops,
        )


class _CompiledTemporal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, diagonal, factors, mass, turnover, free, rhs, iterations, tolerance):
        output = _cg(
            diagonal,
            factors,
            mass,
            turnover,
            free,
            rhs,
            iterations,
            tolerance,
            Ops(diagonal),
            capturing=True,
        )
        ctx.iterations, ctx.tolerance = iterations, tolerance
        ctx.save_for_backward(diagonal, factors, mass, turnover, free, output)
        return output

    @staticmethod
    def backward(ctx, gradient):
        return _implicit_backward(
            ctx.saved_tensors, gradient, ctx.iterations, ctx.tolerance, compiled=True
        )


def solve(diagonal, factors, mass, turnover, free, rhs, iterations, tolerance):
    boundary = _CompiledTemporal if _torch_compiling() else _ForwardTemporal
    return boundary.apply(diagonal, factors, mass, turnover, free, rhs, iterations, tolerance)
