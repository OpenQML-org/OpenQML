"""Matrix-free first-order JAX rules for external quantum execution.

Directionals and pullbacks are linear in their vector arguments at fixed primal
weights/inputs. Native batching keeps transform axes separate from model batch
dimensions. No Jacobian, bound values or quantum states are cached.

Extension contracts: https://docs.jax.dev/en/latest/jax-primitives.html
"""

from functools import partial
from inspect import signature

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend import core
from jax.interpreters import ad, batching, mlir

from ..errors import ValidationError
from ._spec import broadcast_batch_shape

_HIGHER_ORDER = (
    "External quantum derivatives are first-order; select a native JAX backend "
    "for higher derivatives."
)
_CALLBACK_METHOD = "vmap_method" in signature(jax.pure_callback).parameters


def _callback(*args, **kwargs):
    if not _CALLBACK_METHOD:
        kwargs.pop("vmap_method", None)
        kwargs["vectorized"] = False
    return jax.pure_callback(*args, **kwargs)


def _promote(args, dimensions):
    size = next(v.shape[d] for v, d in zip(args, dimensions, strict=True) if d is not None)
    shapes = [
        tuple(n for i, n in enumerate(v.shape) if i != d)
        for v, d in zip(args, dimensions, strict=True)
    ]
    rank, values = max(len(s) - 1 for s in shapes), []
    for value, dim, shape in zip(args, dimensions, shapes, strict=True):
        moved = jnp.expand_dims(value, 0) if dim is None else jnp.moveaxis(value, dim, 0)
        trailing = (1,) * (rank - len(shape) + 1) + shape
        moved = moved.reshape((1 if dim is None else size,) + trailing)
        values.append(jnp.broadcast_to(moved, (size,) + trailing))
    return values, shapes, size


def _lower(implementation, multiple=False):
    def lowering(ctx, *args, **params):
        return mlir.lower_fun(partial(implementation, **params), multiple_results=multiple)(
            ctx, *args
        )

    return lowering


def _no_higher(*args, **kwargs):
    raise ValidationError(_HIGHER_ORDER)


def _output_aval(w, x, *, layer):
    return jax.core.ShapedArray(broadcast_batch_shape(w, x) + (len(layer.observables),), x.dtype)


def _primal_impl(weights, inputs, *, layer):
    from ._spec import numeric_expectations

    shape = broadcast_batch_shape(weights, inputs) + (len(layer.observables),)

    def callback(w, x):
        return np.asarray(numeric_expectations(layer, w, x), dtype=x.dtype).reshape(shape)

    return _callback(
        callback,
        jax.ShapeDtypeStruct(shape, inputs.dtype),
        weights,
        inputs,
        vmap_method="sequential",
    )


_primal = core.Primitive("openqml_external_primal")
_primal.def_impl(_primal_impl)
_primal.def_abstract_eval(_output_aval)
mlir.register_lowering(_primal, _lower(_primal_impl))
ad.primitive_jvps[_primal] = _no_higher


def _primal_batch(args, dimensions, *, layer):
    values, _, _ = _promote(args, dimensions)
    return _primal.bind(*values, layer=layer), 0


batching.primitive_batchers[_primal] = _primal_batch


def _directional_impl(weights, inputs, dw, dx, *, layer):
    from ._spec import numeric_jvp

    shape = broadcast_batch_shape(weights, inputs) + (len(layer.observables),)
    return _callback(
        lambda w, x, tw, tx: numeric_jvp(layer, w, x, tw, tx),
        jax.ShapeDtypeStruct(shape, inputs.dtype),
        weights,
        inputs,
        dw,
        dx,
        vmap_method="sequential",
    )


_directional = core.Primitive("openqml_external_directional")
_directional.def_impl(_directional_impl)


@_directional.def_abstract_eval
def _directional_abstract(w, x, dw, dx, *, layer):
    if (w.shape, x.shape, w.dtype, x.dtype) != (dw.shape, dx.shape, dw.dtype, dx.dtype):
        raise ValidationError("External JVP tangent shapes and dtypes must match primals.")
    return _output_aval(w, x, layer=layer)


mlir.register_lowering(_directional, _lower(_directional_impl))


def _directional_jvp(primals, tangents, *, layer):
    if any(not isinstance(t, ad.Zero) for t in tangents[:2]):
        raise ValidationError(_HIGHER_ORDER)
    w, x, dw, dx = primals
    return _directional.bind(*primals, layer=layer), _directional.bind(
        w, x, ad.instantiate_zeros(tangents[2]), ad.instantiate_zeros(tangents[3]), layer=layer
    )


ad.primitive_jvps[_directional] = _directional_jvp


def _directional_transpose(g, w, x, dw, dx, *, layer):
    if ad.is_undefined_primal(w) or ad.is_undefined_primal(x):
        raise ValidationError(_HIGHER_ORDER)
    need_w, need_x = ad.is_undefined_primal(dw), ad.is_undefined_primal(dx)
    if isinstance(g, ad.Zero):
        return (
            None,
            None,
            ad.Zero(dw.aval) if need_w else None,
            ad.Zero(dx.aval) if need_x else None,
        )
    if not (need_w or need_x):
        return None, None, None, None
    values = iter(_pullback.bind(w, x, g, layer=layer, need_w=need_w, need_x=need_x))
    return None, None, next(values) if need_w else None, next(values) if need_x else None


ad.primitive_transposes[_directional] = _directional_transpose


def _directional_batch(args, dimensions, *, layer):
    values, _, _ = _promote(args, dimensions)
    return _directional.bind(*values, layer=layer), 0


batching.primitive_batchers[_directional] = _directional_batch


def _pullback_impl(weights, inputs, cotangent, *, layer, need_w, need_x):
    from ._spec import numeric_vjp

    specs = tuple(
        jax.ShapeDtypeStruct(v.shape, v.dtype)
        for v, needed in ((weights, need_w), (inputs, need_x))
        if needed
    )

    def callback(w, x, g):
        dw, dx = numeric_vjp(layer, w, x, g, weight_grad=need_w, input_grad=need_x)
        return tuple(value for value in (dw, dx) if value is not None)

    return _callback(callback, specs, weights, inputs, cotangent, vmap_method="sequential")


_pullback = core.Primitive("openqml_external_pullback")
_pullback.multiple_results = True
_pullback.def_impl(_pullback_impl)


@_pullback.def_abstract_eval
def _pullback_abstract(w, x, g, *, layer, need_w, need_x):
    return [
        jax.core.ShapedArray(v.shape, v.dtype) for v, needed in ((w, need_w), (x, need_x)) if needed
    ]


mlir.register_lowering(_pullback, _lower(_pullback_impl, multiple=True))


def _pullback_batch(args, dimensions, *, layer, need_w, need_x):
    values, shapes, size = _promote(args, dimensions)
    outputs = _pullback.bind(*values, layer=layer, need_w=need_w, need_x=need_x)
    targets = [s for s, needed in zip(shapes[:2], (need_w, need_x), strict=True) if needed]
    return [
        value.reshape((size,) + shape) for value, shape in zip(outputs, targets, strict=True)
    ], [0] * len(outputs)


batching.primitive_batchers[_pullback] = _pullback_batch


def _pullback_jvp(primals, tangents, **params):
    if any(not isinstance(t, ad.Zero) for t in tangents[:2]):
        raise ValidationError(_HIGHER_ORDER)
    w, x, _ = primals
    return _pullback.bind(*primals, **params), _pullback.bind(
        w, x, ad.instantiate_zeros(tangents[2]), **params
    )


ad.primitive_jvps[_pullback] = _pullback_jvp


def _pullback_transpose(cotangents, w, x, g, *, layer, need_w, need_x):
    if ad.is_undefined_primal(w) or ad.is_undefined_primal(x):
        raise ValidationError(_HIGHER_ORDER)
    if not ad.is_undefined_primal(g):
        return None, None, None
    values = iter(cotangents)
    dw = ad.instantiate_zeros(next(values)) if need_w else jnp.zeros_like(w)
    dx = ad.instantiate_zeros(next(values)) if need_x else jnp.zeros_like(x)
    return None, None, _directional.bind(w, x, dw, dx, layer=layer)


ad.primitive_transposes[_pullback] = _pullback_transpose


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def external_call(layer, weights, inputs):
    return _primal.bind(weights, inputs, layer=layer)


@external_call.defjvp
def _external_jvp(layer, primals, tangents):
    weights, inputs = primals
    return _primal.bind(weights, inputs, layer=layer), _directional.bind(
        weights, inputs, *tangents, layer=layer
    )
