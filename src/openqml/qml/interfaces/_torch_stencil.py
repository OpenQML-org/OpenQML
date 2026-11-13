"""First-order Torch transforms using one explicit quantum stencil per primal."""

import numpy as np
import torch

from ._spec import broadcast_batch_shape
from ._torch_ad import _batch_values, _NumericalProduct, _numpy


class _CachedVJP(_NumericalProduct):
    generate_vmap_rule = True

    @staticmethod
    def forward(jacobian, cotangents, inputs, weights):
        product = torch.einsum("...mp,...m->...p", jacobian, cotangents)
        size = inputs.shape[-1]
        return (
            product[..., :size].sum_to_size(inputs.shape),
            product[..., size:].sum_to_size(weights.shape),
        )


class _CachedJVP(_NumericalProduct):
    generate_vmap_rule = True

    @staticmethod
    def forward(jacobian, inputs, weights, dx, dw):
        shape = broadcast_batch_shape(weights, inputs)
        direction = torch.cat(
            (dx.expand(shape + (dx.shape[-1],)), dw.expand(shape + (dw.shape[-1],))), dim=-1
        )
        return torch.einsum("...mp,...p->...m", jacobian, direction)


class BatchedShiftFunction(torch.autograd.Function):
    @staticmethod
    def forward(inputs, weights, layer):
        from ._spec import numeric_linearization

        values, jacobian = numeric_linearization(layer, _numpy(weights), _numpy(inputs))
        return tuple(
            torch.tensor(np.asarray(v), dtype=inputs.dtype, device=inputs.device)
            for v in (values, jacobian)
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        values, weights, _ = inputs
        _, jacobian = output
        ctx.mark_non_differentiable(jacobian)
        ctx.save_for_backward(values, weights, jacobian)
        ctx.save_for_forward(values, weights, jacobian)

    @staticmethod
    def backward(ctx, cotangents, _jacobian_cotangents):
        inputs, weights, jacobian = ctx.saved_tensors
        dx, dw = _CachedVJP.apply(jacobian, cotangents, inputs, weights)
        return dx, dw, None

    @staticmethod
    def jvp(ctx, dx, dw, _layer_tangent):
        inputs, weights, jacobian = ctx.saved_tensors
        dx = torch.zeros_like(inputs) if dx is None else dx
        dw = torch.zeros_like(weights) if dw is None else dw
        return _CachedJVP.apply(jacobian, inputs, weights, dx, dw), None

    @staticmethod
    def vmap(info, in_dims, inputs, weights, layer):
        values, _ = _batch_values(info.batch_size, in_dims[:2], (inputs, weights))
        return BatchedShiftFunction.apply(*values, layer), (0, 0)
