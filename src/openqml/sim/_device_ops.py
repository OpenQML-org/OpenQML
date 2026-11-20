"""Lazy dispatch and differentiation for OpenQML device kernels."""

from functools import lru_cache
from importlib import metadata, util

from packaging.version import Version


@lru_cache(maxsize=1)
def _has_triton():
    if util.find_spec("triton") is None:
        return False
    try:
        return Version(metadata.version("triton")) >= Version("3.4")
    except metadata.PackageNotFoundError:
        return False


def supports_device_kernel(state, matrix, torch):
    return (
        torch.version.cuda is not None
        and not torch._C._are_functorch_transforms_active()
        and torch.autograd.forward_ad.unpack_dual(state).tangent is None
        and torch.autograd.forward_ad.unpack_dual(matrix).tangent is None
        and _has_triton()
        and torch.cuda.get_device_capability(state.device)[0] >= 8
    )


def local_blocks(state, wires, width, ops):
    batch = tuple(state.shape[:-1])
    local = tuple(wires) + tuple(q for q in range(width) if q not in wires)
    axes = tuple(range(len(batch))) + tuple(len(batch) + q for q in local)
    return ops.transpose(state.reshape(batch + (2,) * width), axes).reshape(
        batch + (1 << len(wires), 1 << (width - len(wires)))
    )


def matrix_cotangent(state, cotangent, wires, width, ops):
    left = local_blocks(cotangent, wires, width, ops)
    right = local_blocks(state, wires, width, ops)
    return ops.matmul(left, ops.conj(right).swapaxes(-1, -2))


@lru_cache(maxsize=1)
def _differentiable_gate():
    import torch

    from ._triton_kernels import launch_gate
    from .dense import _apply_matrix_portable, _TensorOps

    class Gate(torch.autograd.Function):
        @staticmethod
        def forward(ctx, state, matrix, wires, width):
            ctx.save_for_backward(state, matrix)
            ctx.wires, ctx.width = wires, width
            return launch_gate(state, matrix, wires, width)

        @staticmethod
        def backward(ctx, gradient):
            state, matrix = ctx.saved_tensors
            wires, width = ctx.wires, ctx.width
            ops = _TensorOps("torch", state.real)
            dx = dm = None
            if ctx.needs_input_grad[0]:
                adjoint = matrix.mH
                if torch.is_grad_enabled():
                    dx = _apply_matrix_portable(gradient, adjoint, wires, width, ops)
                else:
                    dx = launch_gate(gradient, adjoint, wires, width)
            if ctx.needs_input_grad[1]:
                dm = matrix_cotangent(state, gradient, wires, width, ops).sum_to_size(matrix.shape)
            return dx, dm, None, None

    return Gate


def apply_device_gate(state, matrix, wires, width, torch):
    if torch.is_grad_enabled() and (state.requires_grad or matrix.requires_grad):
        return _differentiable_gate().apply(state, matrix, tuple(wires), width)
    from ._triton_kernels import launch_gate

    return launch_gate(state, matrix, wires, width)


@lru_cache(maxsize=1)
def _differentiable_contraction():
    import torch

    from ._triton_kernels import launch_contraction

    class Contraction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, left, right, plan, ops):
            ctx.save_for_backward(left, right)
            ctx.plan, ctx.ops = plan, ops
            return launch_contraction(plan, left, right)

        @staticmethod
        def backward(ctx, gradient):
            left, right = ctx.saved_tensors
            plan = ctx.plan
            a, b = plan.matrix_views(left, right, ctx.ops)
            dy = gradient.reshape(
                gradient.shape[: gradient.ndim - plan.output_rank] + (plan.m, plan.n)
            )

            def restore(value, original, axes):
                prefix = original.ndim - len(axes)
                rank = tuple(range(prefix)) + tuple(
                    prefix + axes.index(i) for i in range(len(axes))
                )
                return value.reshape(original.shape[:prefix] + (2,) * len(axes)).permute(rank)

            dl = dr = None
            if ctx.needs_input_grad[0]:
                dl = restore(ctx.ops.matmul(dy, b.mH).sum_to_size(a.shape), left, plan.left_axes)
            if ctx.needs_input_grad[1]:
                dr = restore(ctx.ops.matmul(a.mH, dy).sum_to_size(b.shape), right, plan.right_axes)
            return dl, dr, None, None

    return Contraction


def apply_device_contraction(plan, left, right, ops):
    if ops.xp.is_grad_enabled() and (left.requires_grad or right.requires_grad):
        return _differentiable_contraction().apply(left, right, plan, ops)
    from ._triton_kernels import launch_contraction

    return launch_contraction(plan, left, right)
