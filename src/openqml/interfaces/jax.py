"""Functional JAX layers with composable transforms and native higher derivatives."""

from __future__ import annotations

try:
    import jax
    import jax.numpy as jnp
except ImportError as exc:
    from ..errors import MissingDependencyError

    raise MissingDependencyError("Cannot load JAX. Run: python -m pip install openqml.") from exc

from ..errors import ValidationError
from ..parameters import real as _real
from ..simulator import _checkpoint_interval, _resource_check, expectations
from ._spec import (
    LayerSpecMixin,
    broadcast_batch_shape,
    prepare_execution,
    prepare_gradients,
    prepare_layer,
    prepare_readout,
    resolve_gradient_method,
)


class QuantumLayer(LayerSpecMixin):
    """Functional layer: init(key) returns weights; layer(weights, inputs) evaluates.

    Weights (..., n_weights) and inputs (..., n_inputs) broadcast over leading
    dimensions, including weights produced by a classical network.

    optimize=True preserves source ownership and returns zero gradients for
    eliminated symbols. circuit is executable IR; optimization holds the report.
    gradient_method='auto' retains native autodiff or numerical parameter shifts.
    NumPy unitary simulation also supports explicit first-order adjoint gradients.
    External backends support standard jvp/vjp, jacfwd/jacrev, vmap and fixed-point
    linearize/linear_transpose. Nonlinear higher derivatives require native JAX.
    reduce_readout=True compacts the declared unitary readout cone; readout_plan
    records provenance, while to_spec()/checkpoint() retain the full source.
    execution_method='pauli' compiles exact real expectation arithmetic with
    pauli_options growth limits; native autodiff retains higher derivatives.
    execution_method='planned' factors readouts and selects exact local engines;
    execution_plan exposes choices, provenance and bounded reference checks.
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
        checkpoint_interval=None,
        gradient_chunk_size=64,
        gradient_method="auto",
        max_branches=256,
        optimize=False,
        optimization_options=None,
        reduce_readout=False,
        execution_method="auto",
        pauli_options=None,
        planning_options=None,
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
        self.source_observables = self.observables
        circuit, self.observables, self.readout_plan = prepare_readout(
            circuit, self.observables, enabled=reduce_readout, backend=backend
        )
        self.execution_method = execution_method
        self.execution_plan = prepare_execution(
            circuit,
            self.observables,
            method=execution_method,
            options=pauli_options,
            planning_options=planning_options,
            backend=backend,
            limits={"max_qubits": max_qubits, "max_memory_mb": max_memory_mb},
            checkpoint_interval=checkpoint_interval,
        )
        self.pauli_plan = self.execution_plan if execution_method == "pauli" else None
        self.checkpoint_interval = _checkpoint_interval(circuit, backend, checkpoint_interval)
        if backend == "jax" and self.execution_plan is None:
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
        self._active_parameters = frozenset(
            circuit.parameters
            if self.execution_plan is None
            else self.execution_plan.active_parameters
        )
        self._has_removed_parameters = len(self._active_parameters) < len(
            self.input_parameters + self.weight_parameters
        )
        self.limits = {
            "max_qubits": max_qubits,
            "max_memory_mb": max_memory_mb,
            "max_branches": max_branches,
        }
        if self.checkpoint_interval is not None:
            self.limits["checkpoint_interval"] = self.checkpoint_interval
        self.gradient_method = resolve_gradient_method(
            circuit, backend, gradient_method, self.limits
        )
        self.adjoint_plan, self.shift_plan = prepare_gradients(self)

    def init(self, key, *, scale=0.1, dtype=jnp.float32):
        scale = _real(scale)
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

        Delegates to standard jax.jvp. External products stream without a full
        Jacobian; native JAX also retains nonlinear higher derivatives.
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
        return jax.jvp(self, (weights, inputs), (dw, dx))

    def __call__(self, weights, inputs):
        weights, inputs = jnp.asarray(weights), jnp.asarray(inputs)
        if weights.ndim < 1 or weights.shape[-1] != len(self.weight_parameters):
            raise ValidationError(f"Expected weight shape (..., {len(self.weight_parameters)}).")
        if inputs.ndim < 1 or inputs.shape[-1] != len(self.input_parameters):
            raise ValidationError(f"Expected input shape (..., {len(self.input_parameters)}).")
        if inputs.dtype != weights.dtype or inputs.dtype not in {
            jnp.dtype("float32"),
            jnp.dtype("float64"),
        }:
            raise ValidationError("Inputs and weights must share float32 or float64 dtype.")
        import math

        shape = broadcast_batch_shape(weights, inputs)
        count = math.prod(shape)
        if count == 0:
            return (
                jnp.zeros(shape + (len(self.observables),), dtype=inputs.dtype)
                + jnp.square(inputs * 0).sum()
                + jnp.square(weights * 0).sum()
            )
        if self.batch_chunk_size is not None:
            if count > self.batch_chunk_size:
                flat = jnp.broadcast_to(inputs, shape + (len(self.input_parameters),)).reshape(
                    count, len(self.input_parameters)
                )
                flat_weights = jnp.broadcast_to(
                    weights, shape + (len(self.weight_parameters),)
                ).reshape(count, len(self.weight_parameters))
                # Mapping whole chunks avoids tracing one copy of the circuit
                # per row and retains JIT, reverse mode and higher derivatives.
                size = self.batch_chunk_size
                full = count // size * size
                chunks = flat[:full].reshape(full // size, size, len(self.input_parameters))
                weight_chunks = flat_weights[:full].reshape(
                    full // size, size, len(self.weight_parameters)
                )
                result = jax.lax.map(lambda pair: self(*pair), (weight_chunks, chunks))
                result = result.reshape(full, len(self.observables))
                if full < count:
                    result = jnp.concatenate(
                        (result, self(flat_weights[full:], flat[full:])), axis=0
                    )
                return result.reshape(shape + (len(self.observables),))
        if self.execution_backend != "jax":
            return _external_layer(self, weights, inputs)
        bindings = {
            name: inputs[..., i]
            for i, name in enumerate(self.input_parameters)
            if name in self._active_parameters
        }
        bindings.update(
            (name, weights[..., i])
            for i, name in enumerate(self.weight_parameters)
            if name in self._active_parameters
        )
        result = (
            self.execution_plan.evaluate(bindings, backend="jax", like=weights, batch_shape=shape)
            if self.execution_plan is not None
            else expectations(
                self.circuit,
                self.observables,
                bindings,
                backend="jax",
                like=weights,
                batch_shape=shape,
                **self.limits,
            )
        )
        if self._has_removed_parameters:
            result = result + jnp.square(inputs * 0).sum(-1, keepdims=True)
            result = result + jnp.square(weights * 0).sum(-1, keepdims=True)
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
    from ._jax_ad import external_call

    return external_call(layer, weights, inputs)


def __getattr__(name):
    """Expose native JAX transforms, arrays and random keys through one import."""
    if name == "enable_x64" and not hasattr(jax, name):
        from jax.experimental import enable_x64

        return enable_x64
    return getattr(jax, name)


def __dir__():
    return sorted(set(globals()) | set(dir(jax)) | {"enable_x64"})
