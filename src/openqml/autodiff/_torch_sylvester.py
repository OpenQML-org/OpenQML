"""Lazy Torch SLD boundary with reverse, forward and higher derivatives."""

import torch

from ..sim.dense import _TensorOps
from ._sylvester import _spectral, adjoint, hermitian


def _implicit_backward(rho, solution, gradient, damping, *, compiled):
    dual = (
        _CompiledSLD.apply(rho, hermitian(gradient), damping)
        if compiled
        else solve(rho, hermitian(gradient), damping)
    )
    derivative = -hermitian(
        (torch.matmul(dual, adjoint(solution)) + torch.matmul(adjoint(solution), dual)) * 0.5
    )
    return derivative, dual, None


class _SLDSolve(torch.autograd.Function):
    @staticmethod
    def forward(rho, rhs, damping):
        return _spectral(rho, rhs, damping, _TensorOps("torch", rho.real))

    @staticmethod
    def setup_context(ctx, inputs, output):
        rho, _, ctx.damping = inputs
        ctx.save_for_backward(rho, output)
        ctx.save_for_forward(rho, output)

    @staticmethod
    def backward(ctx, gradient):
        rho, solution = ctx.saved_tensors
        return _implicit_backward(rho, solution, gradient, ctx.damping, compiled=False)


class _ForwardSLD(_SLDSolve):
    generate_vmap_rule = True

    @staticmethod
    def jvp(ctx, d_rho, d_rhs, _damping):
        rho, solution = ctx.saved_tensors
        d_rhs = torch.zeros_like(solution) if d_rhs is None else d_rhs
        if d_rho is not None:
            d_rhs = d_rhs - (torch.matmul(d_rho, solution) + torch.matmul(solution, d_rho)) * 0.5
        return solve(rho, hermitian(d_rhs), ctx.damping)


class _CompiledSLD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rho, rhs, damping):
        result = _spectral(rho, rhs, damping, _TensorOps("torch", rho.real), capturing=True)
        ctx.damping = damping
        ctx.save_for_backward(rho, result)
        return result

    @staticmethod
    def backward(ctx, gradient):
        rho, solution = ctx.saved_tensors
        return _implicit_backward(rho, solution, gradient, ctx.damping, compiled=True)


def solve(rho, rhs, damping, *, compiling=False):
    return (_CompiledSLD if compiling else _ForwardSLD).apply(rho, rhs, damping)
