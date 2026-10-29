"""Composable first-order PyTorch transforms over external quantum backends.

Foreign NumPy execution stays in custom Function.forward boundaries. JVPs and
VJPs are streamed products; batching promotes one transform axis independently
of the model's existing broadcast dimensions. Higher derivatives require a
native tensor backend and fail explicitly here.

Contracts: https://docs.pytorch.org/docs/stable/notes/extending.func.html
"""

import numpy as np
import torch

from ..errors import ValidationError
from ._spec import broadcast_batch_shape

_HIGHER_ORDER = (
    "External quantum derivatives are first-order; select a native PyTorch backend "
    "for higher derivatives."
)


def _numpy(value):
    return value.detach().cpu().numpy()


def _batch_values(size, dimensions, values):
    shapes = [
        tuple(n for i, n in enumerate(value.shape) if i != dim)
        for value, dim in zip(values, dimensions, strict=True)
    ]
    rank = max(len(shape) - 1 for shape in shapes)
    result = []
    for value, dim, shape in zip(values, dimensions, shapes, strict=True):
        moved = value.unsqueeze(0) if dim is None else value.movedim(dim, 0)
        trailing = (1,) * (rank - len(shape) + 1) + shape
        result.append(
            moved.reshape((1 if dim is None else size,) + trailing).expand((size,) + trailing)
        )
    return result, shapes


class _NumericalProduct(torch.autograd.Function):
    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, *cotangents):
        raise ValidationError(_HIGHER_ORDER)

    @staticmethod
    def jvp(ctx, *tangents):
        raise ValidationError(_HIGHER_ORDER)


class _JVP(_NumericalProduct):
    @staticmethod
    def forward(inputs, weights, dx, dw, layer):
        from ._spec import numeric_jvp

        value = numeric_jvp(layer, _numpy(weights), _numpy(inputs), _numpy(dw), _numpy(dx))
        return torch.as_tensor(value, dtype=inputs.dtype, device=inputs.device)

    @staticmethod
    def vmap(info, in_dims, inputs, weights, dx, dw, layer):
        values, _ = _batch_values(info.batch_size, in_dims[:4], (inputs, weights, dx, dw))
        return _JVP.apply(*values, layer), 0


class _VJP(_NumericalProduct):
    @staticmethod
    def forward(inputs, weights, cotangents, layer, input_grad, weight_grad):
        from ._spec import numeric_vjp

        dw, dx = numeric_vjp(
            layer,
            _numpy(weights),
            _numpy(inputs),
            _numpy(cotangents),
            input_grad=input_grad,
            weight_grad=weight_grad,
        )
        return (
            torch.zeros_like(inputs)
            if dx is None
            else torch.as_tensor(dx, dtype=inputs.dtype, device=inputs.device),
            torch.zeros_like(weights)
            if dw is None
            else torch.as_tensor(dw, dtype=weights.dtype, device=weights.device),
        )

    @staticmethod
    def vmap(info, in_dims, inputs, weights, cotangents, layer, input_grad, weight_grad):
        values, shapes = _batch_values(info.batch_size, in_dims[:3], (inputs, weights, cotangents))
        dx, dw = _VJP.apply(*values, layer, input_grad, weight_grad)
        # Remove only synthetic padding; model broadcast reductions were already
        # performed by numeric_vjp while the transform axis remained independent.
        return (
            dx.reshape((info.batch_size,) + shapes[0]),
            dw.reshape((info.batch_size,) + shapes[1]),
        ), (0, 0)


class ExternalQuantumFunction(torch.autograd.Function):
    @staticmethod
    def forward(inputs, weights, layer):
        from ._spec import numeric_expectations

        output = numeric_expectations(layer, _numpy(weights), _numpy(inputs))
        shape = broadcast_batch_shape(weights, inputs) + (len(layer.observables),)
        return torch.as_tensor(
            np.asarray(output).reshape(shape), dtype=inputs.dtype, device=inputs.device
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        values, weights, layer = inputs
        ctx.layer = layer
        ctx.save_for_backward(values, weights)
        ctx.save_for_forward(values, weights)

    @staticmethod
    def jvp(ctx, input_tangent, weight_tangent, _layer_tangent):
        inputs, weights = ctx.saved_tensors
        dx = torch.zeros_like(inputs) if input_tangent is None else input_tangent
        dw = torch.zeros_like(weights) if weight_tangent is None else weight_tangent
        return _JVP.apply(inputs, weights, dx, dw, ctx.layer)

    @staticmethod
    def backward(ctx, cotangents):
        inputs, weights = ctx.saved_tensors
        need_x, need_w = ctx.needs_input_grad[:2]
        dx, dw = _VJP.apply(inputs, weights, cotangents, ctx.layer, need_x, need_w)
        return dx if need_x else None, dw if need_w else None, None

    @staticmethod
    def vmap(info, in_dims, inputs, weights, layer):
        values, _ = _batch_values(info.batch_size, in_dims[:2], (inputs, weights))
        return ExternalQuantumFunction.apply(*values, layer), 0
