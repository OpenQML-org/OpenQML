"""JIT-compatible sample-weighted microbatch optimization for JAX/Flax."""

import jax
import jax.numpy as jnp

from ...autodiff.compilation import compile_compute
from ...compute._norms import global_norm
from ...errors import ValidationError
from ...ir import integer
from ...ir.parameters import real


def accumulate_gradients(
    loss_function, parameters, batch, *, microbatch_size=32, axis_name=None, has_aux=False
):
    """Sum per-example losses/gradients using scan, including an unpadded tail.

    loss_function(parameters, microbatch) must return shape (microbatch_size,).
    A named pmap/shard_map axis uses psum for globally sample-weighted means;
    ordinary global sharded arrays can be supplied to a jitted caller directly.
    Batch-dependent layers retain microbatch semantics.
    """
    size = integer(microbatch_size, "microbatch_size", 1)
    leaves = jax.tree_util.tree_leaves(batch)
    if not leaves or any(x.ndim == 0 for x in leaves):
        raise ValidationError("Batch leaves need a shared nonempty leading sample dimension.")
    count = leaves[0].shape[0]
    if count < 1 or any(x.shape[0] != count for x in leaves):
        raise ValidationError("Batch leaves need the same nonempty leading sample dimension.")

    if type(has_aux) is not bool:
        raise ValidationError("has_aux must be a boolean.")
    if axis_name is not None and (hasattr(jax.lax, "pcast") or hasattr(jax.lax, "pvary")):
        # New JAX manual-axis AD automatically sums cotangents of invariant
        # inputs. Differentiate varying local copies so accumulation performs
        # exactly one explicit reduction, on both new and legacy JAX releases.
        axes = axis_name if isinstance(axis_name, tuple) else (axis_name,)

        def local_parameter(value):
            for axis in axes:
                if hasattr(jax.lax, "pcast"):
                    manual_type = getattr(jax.typeof(value), "mat", None)
                    if manual_type is not None and axis not in manual_type.varying:
                        value = jax.lax.pcast(value, axis, to="varying")
                elif axis not in getattr(jax.core.get_aval(value), "vma", frozenset()):
                    value = jax.lax.pvary(value, axis)
            return value

        parameters = jax.tree_util.tree_map(local_parameter, parameters)

    def objective(params, micro):
        result = loss_function(params, micro)
        losses, auxiliary = result if has_aux else (result, ())
        n = jax.tree_util.tree_leaves(micro)[0].shape[0]
        if losses.shape != (n,):
            raise ValidationError("JAX training loss must return one loss per example.")
        losses = losses.astype(jnp.result_type(losses.dtype, jnp.float32))

        def reduce(value):
            value = jnp.asarray(value)
            if value.ndim == 0:
                return value * n
            if value.shape[0] != n:
                raise ValidationError(
                    "Auxiliary metrics need a scalar mean or per-example leading dimension."
                )
            return jnp.sum(value, axis=0)

        return jnp.sum(losses), jax.tree_util.tree_map(reduce, auxiliary)

    differentiate = jax.value_and_grad(objective, has_aux=True)

    def body(state, micro):
        (value, auxiliary), gradient = differentiate(parameters, micro)
        return (
            state[0] + value,
            jax.tree_util.tree_map(jnp.add, state[1], gradient),
            jax.tree_util.tree_map(jnp.add, state[2], auxiliary),
        ), None

    chunks, tail = divmod(count, size)
    first = min(count, size)
    (value, auxiliary), gradient = differentiate(
        parameters, jax.tree_util.tree_map(lambda x: x[:first], batch)
    )
    state = value, gradient, auxiliary
    if chunks > 1:
        stacked = jax.tree_util.tree_map(
            lambda x: x[size : chunks * size].reshape((chunks - 1, size) + x.shape[1:]), batch
        )
        state, _ = jax.lax.scan(body, state, stacked)
    if tail and chunks:
        state, _ = body(state, jax.tree_util.tree_map(lambda x: x[chunks * size :], batch))
    total, gradient, auxiliary = state
    samples = jnp.asarray(count, jnp.int32)
    if axis_name is not None:
        total, samples = jax.lax.psum((total, samples), axis_name)
        gradient = jax.tree_util.tree_map(lambda x: jax.lax.psum(x, axis_name), gradient)
        auxiliary = jax.tree_util.tree_map(lambda x: jax.lax.psum(x, axis_name), auxiliary)
    result = total / samples, jax.tree_util.tree_map(lambda x: x / samples, gradient), samples
    return (
        result + (jax.tree_util.tree_map(lambda x: x / samples, auxiliary),) if has_aux else result
    )


def make_train_step(
    optimizer,
    loss_function,
    *,
    microbatch_size=32,
    max_grad_norm=1.0,
    axis_name=None,
    jit=True,
    has_aux=False,
    gradient_metrics=None,
):
    """Return step(params, optimizer_state, batch) with finite-gradient rejection.

    The optimizer follows Optax's update contract. No Optax dependency is needed
    merely to import the JAX integration. Compile outside pmap/shard_map when
    jit=False; NamedSharding inputs are also supported by the returned jit.
    """
    integer(microbatch_size, "microbatch_size", 1)
    if max_grad_norm is not None and real(max_grad_norm) <= 0:
        raise ValidationError("max_grad_norm must be positive or None.")
    if (
        type(jit) is not bool
        or not callable(loss_function)
        or not callable(getattr(optimizer, "update", None))
    ):
        raise ValidationError("Provide an optimizer, loss callable and boolean jit.")

    def step(parameters, optimizer_state, batch):
        result = accumulate_gradients(
            loss_function,
            parameters,
            batch,
            microbatch_size=microbatch_size,
            axis_name=axis_name,
            has_aux=has_aux,
        )
        loss, gradient, samples = result[:3]
        auxiliary = result[3] if has_aux else {}
        if not isinstance(auxiliary, dict) or set(auxiliary) & {
            "loss",
            "grad_norm",
            "samples",
            "skipped",
        }:
            raise ValidationError(
                "Train-step auxiliary metrics must be a dictionary without reserved keys."
            )
        norm = global_norm(jax.tree_util.tree_leaves(gradient), jnp, "jax")
        diagnostics = {} if gradient_metrics is None else gradient_metrics(gradient)
        if not isinstance(diagnostics, dict) or set(diagnostics) & (
            set(auxiliary) | {"loss", "grad_norm", "samples", "skipped"}
        ):
            raise ValidationError("Gradient metrics must use distinct nonreserved dictionary keys.")
        finite = jnp.isfinite(loss) & jnp.isfinite(norm)
        if max_grad_norm is not None:
            factor = jnp.minimum(1.0, max_grad_norm / jnp.maximum(norm, 1e-30))
            gradient = jax.tree_util.tree_map(lambda x: x * factor, gradient)

        def update(state):
            params, opt_state = state
            updates, opt_state = optimizer.update(gradient, opt_state, params)
            return jax.tree_util.tree_map(jnp.add, params, updates), opt_state

        params, opt_state = jax.lax.cond(
            finite, update, lambda state: state, (parameters, optimizer_state)
        )
        return (
            params,
            opt_state,
            {
                "loss": loss,
                "grad_norm": norm,
                "samples": samples,
                "skipped": ~finite,
                **auxiliary,
                **diagnostics,
            },
        )

    return compile_compute(step, backend="jax", check_outputs=False) if jit else step
