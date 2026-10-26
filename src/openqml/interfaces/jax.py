"""Pure functional JAX layer supporting jit, grad, vmap and higher derivatives."""

from __future__ import annotations

try:
    import jax
    import jax.numpy as jnp
except ImportError as exc:
    from ..errors import MissingDependencyError

    raise MissingDependencyError("Install openqml[jax] for QuantumLayer.") from exc

from ..errors import ValidationError
from ..parameters import real
from ..simulator import _resource_check, expectations
from ._spec import LayerSpecMixin, prepare_layer


class QuantumLayer(LayerSpecMixin):
    """Functional layer: init(key) returns weights; layer(weights, inputs) evaluates.

    optimize=True preserves source ownership and returns zero gradients for
    eliminated symbols. circuit is executable IR; optimization holds the report.
    """

    def __init__(
        self,
        circuit,
        input_parameters,
        weight_parameters,
        observables=None,
        *,
        max_qubits=20,
        max_memory_mb=256,
        backend="jax",
        batch_chunk_size=None,
        gradient_chunk_size=64,
        max_branches=256,
        optimize=False,
        optimization_options=None,
    ):
        from ..ir import integer

        integer(max_branches, "max_branches", 1)
        self.source_circuit = circuit
        (
            self.input_parameters,
            self.weight_parameters,
            self.observables,
            circuit,
            self.optimization,
        ) = prepare_layer(
            circuit,
            input_parameters,
            weight_parameters,
            observables,
            optimize=optimize,
            optimization_options=optimization_options,
        )
        if backend == "jax":
            if circuit.has_effects:
                from ..density import _planning_limit
                from ..density import _resource_check as density_check
                from ..planning import plan_density

                density_check(
                    circuit,
                    max_qubits,
                    max_memory_mb,
                    branches=plan_density(
                        circuit,
                        max_branches=_planning_limit(
                            circuit, max_qubits, max_memory_mb, max_branches=max_branches
                        ),
                    ).peak_branches,
                )
            else:
                _resource_check(circuit, max_qubits, max_memory_mb)
        self.execution_backend = backend
        self.batch_chunk_size = (
            None if batch_chunk_size is None else integer(batch_chunk_size, "batch_chunk_size", 1)
        )

        self.gradient_chunk_size = integer(gradient_chunk_size, "gradient_chunk_size", 2)
        self.circuit = circuit
        self._active_parameters = frozenset(circuit.parameters)
        self.limits = {
            "max_qubits": max_qubits,
            "max_memory_mb": max_memory_mb,
            "max_branches": max_branches,
        }

    def init(self, key, *, scale=0.1, dtype=jnp.float32):
        scale = real(scale)
        if scale < 0 or jnp.dtype(dtype) not in {jnp.dtype("float32"), jnp.dtype("float64")}:
            raise ValidationError("Use nonnegative scale and float32/float64 dtype.")
        return scale * jax.random.normal(key, (len(self.weight_parameters),), dtype=dtype)

    @classmethod
    def from_checkpoint(cls, checkpoint, *, dtype=None, **kwargs):
        """Return (layer, weights), preserving saved precision unless dtype is explicit."""
        from ..layers import LayerCheckpoint

        if not isinstance(checkpoint, LayerCheckpoint):
            raise ValidationError("Expected a LayerCheckpoint.")
        layer = cls.from_spec(checkpoint.spec, **kwargs)
        return layer, layer.load_checkpoint(checkpoint, dtype=dtype)

    def load_checkpoint(self, checkpoint, *, dtype=None):
        """Return weights for this exact configuration without mutating the layer."""
        return _checkpoint_weights(checkpoint, self.to_spec(), dtype)

    def jvp(self, weights, inputs, weight_tangent, input_tangent):
        """Return (primal, directional derivative), including external backends.

        Native JAX retains higher derivatives. External callbacks support jit/vmap
        and first-order products; use ordinary calls for streamed reverse mode.
        """
        weights, inputs = jnp.asarray(weights), jnp.asarray(inputs)
        dw, dx = jnp.asarray(weight_tangent), jnp.asarray(input_tangent)
        if (dw.shape, dx.shape, dw.dtype, dx.dtype) != (
            weights.shape,
            inputs.shape,
            weights.dtype,
            inputs.dtype,
        ):
            raise ValidationError("JVP tangent shapes and dtypes must match weights/inputs.")
        if self.execution_backend == "jax":
            return jax.jvp(self, (weights, inputs), (dw, dx))
        from ._spec import numeric_jvp

        primal = self(weights, inputs)
        spec = jax.ShapeDtypeStruct(primal.shape, primal.dtype)

        def callback(w, x, tw, tx):
            return numeric_jvp(self, w, x, tw, tx)

        return primal, jax.pure_callback(
            callback, spec, weights, inputs, dw, dx, vmap_method="sequential"
        )

    def __call__(self, weights, inputs):
        weights, inputs = jnp.asarray(weights), jnp.asarray(inputs)
        if weights.shape != (len(self.weight_parameters),):
            raise ValidationError(f"Expected {len(self.weight_parameters)} weights.")
        if inputs.ndim < 1 or inputs.shape[-1] != len(self.input_parameters):
            raise ValidationError(f"Expected input shape (..., {len(self.input_parameters)}).")
        if inputs.dtype != weights.dtype or inputs.dtype not in {
            jnp.dtype("float32"),
            jnp.dtype("float64"),
        }:
            raise ValidationError("Inputs and weights must share float32 or float64 dtype.")
        if self.batch_chunk_size is not None:
            import math

            shape = inputs.shape[:-1]
            count = math.prod(shape)
            if count > self.batch_chunk_size:
                flat = inputs.reshape(count, len(self.input_parameters))
                # Mapping whole chunks avoids tracing one copy of the circuit
                # per row and retains JIT, reverse mode and higher derivatives.
                size = self.batch_chunk_size
                full = count // size * size
                chunks = flat[:full].reshape(full // size, size, len(self.input_parameters))
                result = jax.lax.map(lambda chunk: self(weights, chunk), chunks)
                result = result.reshape(full, len(self.observables))
                if full < count:
                    result = jnp.concatenate((result, self(weights, flat[full:])), axis=0)
                return result.reshape(shape + (len(self.observables),))
        if self.execution_backend != "jax":
            return _external_layer(self, weights, inputs)
        bindings = {
            name: inputs[..., i]
            for i, name in enumerate(self.input_parameters)
            if name in self._active_parameters
        }
        bindings.update(
            (name, value)
            for name, value in zip(self.weight_parameters, weights, strict=True)
            if name in self._active_parameters
        )
        result = expectations(
            self.circuit,
            self.observables,
            bindings,
            backend="jax",
            like=weights,
            batch_shape=inputs.shape[:-1],
            **self.limits,
        )
        if self.optimization is not None and self.optimization.removed_parameters:
            result = result + jnp.square(inputs * 0).sum(-1, keepdims=True)
            result = result + jnp.square(weights * 0).sum()
        return result


def _checkpoint_weights(checkpoint, spec, dtype=None):
    from ..layers import LayerCheckpoint
    from ._spec import checkpoint_values

    if not isinstance(checkpoint, LayerCheckpoint):
        raise ValidationError("Expected a LayerCheckpoint.")
    target = checkpoint.dtype if dtype is None else dtype
    values = checkpoint_values(checkpoint, spec, target)
    # JAX can silently truncate float64 arrays when x64 mode is disabled.
    if values.dtype.name == "float64" and not jax.config.x64_enabled:
        raise ValidationError("Enable JAX x64 or explicitly request dtype=float32.")
    return jnp.asarray(values)


class NativePennyLaneLayer:
    """Functional QNode layer with named PyTree weights, preserving native JAX differentiation."""

    def __init__(self, qnode, weight_shapes):
        self.qnode = qnode.update(interface="jax-jit")
        self.weight_shapes = {name: tuple(shape) for name, shape in weight_shapes.items()}

    def init(self, key, *, scale=0.1, dtype=jnp.float32):
        keys = jax.random.split(key, len(self.weight_shapes))
        return {
            name: scale * jax.random.normal(k, shape, dtype=dtype)
            for k, (name, shape) in zip(keys, self.weight_shapes.items(), strict=True)
        }

    def __call__(self, weights, inputs):
        import math

        shape = inputs.shape[:-1]
        flat = jnp.reshape(inputs, (math.prod(shape), inputs.shape[-1]))

        def one(row):
            value = self.qnode(inputs=row, **weights)
            leaves = jax.tree_util.tree_leaves(value)
            return jnp.concatenate([jnp.ravel(leaf) for leaf in leaves])

        result = jax.vmap(one)(flat)
        return result.reshape(shape + (result.shape[-1],))


def _external_layer(layer, weights, inputs):
    import numpy as np

    from ..batch import expectations_batch
    from ._spec import numeric_rows, numeric_vjp

    def rows(w, x):
        return numeric_rows(layer, w, x)

    def forward_numpy(w, x):
        result = expectations_batch(
            layer.circuit,
            layer.observables,
            rows(w, x),
            backend=layer.execution_backend,
            **layer.limits,
        )
        return np.asarray(result, dtype=x.dtype).reshape(x.shape[:-1] + (len(layer.observables),))

    def backward_numpy(w, x, g):
        return numeric_vjp(layer, w, x, g)

    def primal(w, x):
        spec = jax.ShapeDtypeStruct(x.shape[:-1] + (len(layer.observables),), x.dtype)
        return jax.pure_callback(forward_numpy, spec, w, x, vmap_method="sequential")

    @jax.custom_vjp
    def call(w, x):
        return primal(w, x)

    def forward(w, x):
        return primal(w, x), (w, x)

    def backward(res, g):
        w, x = res
        specs = (jax.ShapeDtypeStruct(w.shape, w.dtype), jax.ShapeDtypeStruct(x.shape, x.dtype))
        return jax.pure_callback(backward_numpy, specs, w, x, g, vmap_method="sequential")

    call.defvjp(forward, backward)
    return call(weights, inputs)
