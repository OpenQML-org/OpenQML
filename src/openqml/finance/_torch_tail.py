"""Implicit Torch derivatives of path-tail roots and their Hessian solves."""

import torch

from ._array import Ops
from ._tail_linear import cg, changes, product, project


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(diagonal, factors, mass, turnover, rhs, horizon, assets, plan):
        ops = Ops(diagonal)
        return cg(
            lambda x: product(diagonal, factors, mass, turnover, x, horizon, assets, ops),
            rhs,
            plan.cg_iterations,
            plan.tolerance,
            ops,
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        diagonal, factors, mass, turnover, _, ctx.horizon, ctx.assets, ctx.plan = inputs
        ctx.save_for_backward(diagonal, factors, mass, turnover, output)
        ctx.save_for_forward(diagonal, factors, mass, turnover, output)

    @staticmethod
    def backward(ctx, gradient):
        diagonal, factors, mass, turnover, output = ctx.saved_tensors
        h, n, ops = ctx.horizon, ctx.assets, Ops(diagonal)
        dual = linear_solve(diagonal, factors, mass, turnover, gradient, h, n, ctx.plan)
        left, right = project(dual, h, n, ops), project(output, h, n, ops)
        fl = torch.einsum("...rd,...d->...r", factors, left)
        fr = torch.einsum("...rd,...d->...r", factors, right)
        df = -mass[..., None] * (
            fl[..., None] * right[..., None, :] + fr[..., None] * left[..., None, :]
        )
        dt = -(left * changes(right, h, n, ops)).sum(dim=-1)
        return -left * right, df, -fl * fr, dt, dual, None, None, None

    @staticmethod
    def jvp(ctx, dd, df, dm, dt, db, _h, _n, _plan):
        diagonal, factors, mass, turnover, output = ctx.saved_tensors
        ops, h, n = Ops(diagonal), ctx.horizon, ctx.assets
        dd = torch.zeros_like(diagonal) if dd is None else dd
        df = torch.zeros_like(factors) if df is None else df
        dm = torch.zeros_like(mass) if dm is None else dm
        dt = torch.zeros_like(turnover) if dt is None else dt
        db = torch.zeros_like(output) if db is None else db
        right = project(output, h, n, ops)
        dual = torch.einsum("...rd,...d->...r", factors, right)
        variation = dd * right + dt[..., None] * changes(right, h, n, ops)
        variation = variation + torch.einsum("...rd,...r->...d", df, mass * dual)
        variation = variation + torch.einsum(
            "...rd,...r->...d",
            factors,
            dm * dual + mass * torch.einsum("...rd,...d->...r", df, right),
        )
        return linear_solve(
            diagonal,
            factors,
            mass,
            turnover,
            db - project(variation, h, n, ops),
            h,
            n,
            ctx.plan,
        )


def linear_solve(diagonal, factors, mass, turnover, rhs, horizon, assets, plan):
    return _Linear.apply(diagonal, factors, mass, turnover, rhs, horizon, assets, plan)


class _Root(torch.autograd.Function):
    @staticmethod
    def forward(plan, *parameters):
        from .tail import _newton

        return _newton(parameters, plan, Ops(parameters[0]))

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.plan = inputs[0]
        ctx.save_for_backward(*inputs[1:], output)
        ctx.save_for_forward(*inputs[1:], output)

    @staticmethod
    def backward(ctx, gradient):
        from .tail import _geometry, _stationarity

        *parameters, output = ctx.saved_tensors
        ops = Ops(parameters[0])
        h, n = parameters[0].shape[-2:]
        with torch.enable_grad():
            diagonal, factors, mass = _geometry(output, parameters, ctx.plan, ops)
            dual = linear_solve(diagonal, factors, mass, parameters[5], gradient, h, n, ctx.plan)
            _, pullback = torch.func.vjp(
                lambda *p: _stationarity(output, p, ctx.plan, ops), *parameters
            )
            derivatives = pullback(-dual)
        return None, *derivatives

    @staticmethod
    def jvp(ctx, _plan, *tangents):
        from .tail import _geometry, _stationarity

        *parameters, output = ctx.saved_tensors
        tangents = tuple(
            torch.zeros_like(x) if dx is None else dx
            for x, dx in zip(parameters, tangents, strict=True)
        )
        ops, (h, n) = Ops(parameters[0]), parameters[0].shape[-2:]
        variation = torch.func.jvp(
            lambda *p: _stationarity(output, p, ctx.plan, ops), tuple(parameters), tangents
        )[1]
        diagonal, factors, mass = _geometry(output, parameters, ctx.plan, ops)
        return linear_solve(diagonal, factors, mass, parameters[5], -variation, h, n, ctx.plan)


def root(parameters, plan):
    return _Root.apply(plan, *parameters)
