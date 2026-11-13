"""JAX first-order linearization with a shared primal/parameter-shift callback."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend import core
from jax.interpreters import ad, batching, mlir

from ._jax_ad import _callback, _lower, _no_higher, _promote
from ._spec import broadcast_batch_shape


def _impure(layer):
    backend = layer.shift_plan._execution_backend
    return not isinstance(backend, str) and not backend.capabilities.deterministic_expectations


def _impl(weights, inputs, *, layer):
    from ._spec import numeric_linearization

    shape = broadcast_batch_shape(weights, inputs) + (len(layer.observables),)
    columns = len(layer.input_parameters) + len(layer.weight_parameters)
    specs = (
        jax.ShapeDtypeStruct(shape, inputs.dtype),
        jax.ShapeDtypeStruct(shape + (columns,), inputs.dtype),
    )

    def callback(w, x):
        return tuple(np.asarray(v, dtype=x.dtype) for v in numeric_linearization(layer, w, x))

    if _impure(layer):
        from jax.experimental import io_callback

        return io_callback(callback, specs, weights, inputs, ordered=False)
    return _callback(callback, specs, weights, inputs, vmap_method="sequential")


_linearization = core.Primitive("openqml_batched_shift_linearization")
_linearization.multiple_results = True
_linearization.def_impl(_impl)


@_linearization.def_effectful_abstract_eval
def _abstract(w, x, *, layer):
    shape = broadcast_batch_shape(w, x) + (len(layer.observables),)
    columns = len(layer.input_parameters) + len(layer.weight_parameters)
    # Propagate the same upstream I/O effect as io_callback through this custom
    # derivative boundary. Pure callbacks could erase or merge charged/noisy jobs.
    from jax._src.callback import _IOEffect

    return [
        jax.core.ShapedArray(shape, x.dtype),
        jax.core.ShapedArray(shape + (columns,), x.dtype),
    ], ({_IOEffect} if _impure(layer) else set())


mlir.register_lowering(_linearization, _lower(_impl, multiple=True))
ad.primitive_jvps[_linearization] = _no_higher


def _batch(args, dimensions, *, layer):
    values, _, _ = _promote(args, dimensions)
    return _linearization.bind(*values, layer=layer), (0, 0)


batching.primitive_batchers[_linearization] = _batch


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def batched_shift_call(layer, weights, inputs):
    return _linearization.bind(weights, inputs, layer=layer)[0]


@batched_shift_call.defjvp
def _jvp(layer, primals, tangents):
    weights, inputs = primals
    values, jacobian = _linearization.bind(weights, inputs, layer=layer)
    shape = broadcast_batch_shape(weights, inputs)
    dw, dx = tangents
    direction = jnp.concatenate(
        (
            jnp.broadcast_to(dx, shape + (dx.shape[-1],)),
            jnp.broadcast_to(dw, shape + (dw.shape[-1],)),
        ),
        axis=-1,
    )
    return values, jnp.einsum("...mp,...p->...m", jacobian, direction)
