"""Functional circuit hypernetworks; parameters are ordinary native JAX PyTrees."""

import math

import jax
import jax.numpy as jnp

from ...compute.kernel import get_compute_kernel
from ...errors import ValidationError
from ...ir.core import integer
from ...ir.parameters import real
from ...sim.dense import _TensorOps
from .._native_training import solve_metric
from ..iqp import IQPGraph


class IQPQuantumLayer:
    def __init__(self, num_qubits, *, edges=None, max_memory_mb=256, max_work=2_000_000):
        self.graph = IQPGraph(num_qubits, edges, max_memory_mb, max_work)

    def init(self, key, *, scale=0.1, dtype=jnp.float32):
        _dtype(dtype)
        scale = real(scale)
        if scale < 0:
            raise ValidationError("scale must be nonnegative.")
        value = scale * jax.random.normal(key, (self.graph.num_coordinates,), dtype=dtype)
        n, m = self.graph.num_qubits, len(self.graph.edges)
        return value.at[n + m :].set(math.pi / 4)

    def __call__(self, weights, inputs):
        weights, inputs = jnp.asarray(weights), jnp.asarray(inputs)
        _input(inputs, self.graph.num_qubits)
        _input(weights, self.graph.num_coordinates)
        shape = jnp.broadcast_shapes(weights.shape[:-1], inputs.shape[:-1])
        w = jnp.broadcast_to(weights, shape + (self.graph.num_coordinates,))
        x = jnp.broadcast_to(inputs, shape + (self.graph.num_qubits,))
        n = self.graph.num_qubits
        coordinates = jnp.concatenate((w[..., :n] + x, w[..., n:]), axis=-1)
        value = (
            coordinates.astype(jnp.float32)
            if coordinates.dtype in {jnp.float16, jnp.bfloat16}
            else coordinates
        )
        return self.graph.evaluate(value, backend="jax").astype(inputs.dtype)


class HyperQuantumAdapter:
    def __init__(
        self,
        features,
        *,
        num_qubits=64,
        rank=16,
        edges=None,
        residual_scale=0.01,
        batch_chunk_size=256,
        max_memory_mb=256,
    ):
        self.features = integer(features, "features", 1)
        self.rank = integer(rank, "rank", 1)
        self.graph = IQPGraph(num_qubits, edges, max_memory_mb)
        self.residual_scale = real(residual_scale)
        self.batch_chunk_size = (
            None if batch_chunk_size is None else integer(batch_chunk_size, "batch_chunk_size", 1)
        )

    def init(self, key, *, dtype=jnp.float32):
        _dtype(dtype)
        keys = jax.random.split(key, 4)
        n, m, r = self.graph.num_qubits, len(self.graph.edges), self.rank
        return {
            "encoder": {
                "down_kernel": jax.random.normal(keys[0], (self.features, r), dtype=dtype)
                / math.sqrt(self.features),
                "down_bias": jnp.zeros((r,), dtype=dtype),
                "angle_kernel": 0.1
                * jax.random.normal(keys[1], (r, n + m), dtype=dtype)
                / math.sqrt(r),
                "angle_bias": jnp.zeros((n + m,), dtype=dtype),
            },
            "axes": {
                "kernel": 0.1 * jax.random.normal(keys[2], (r, 2 * n), dtype=dtype) / math.sqrt(r),
                "bias": jnp.full((2 * n,), math.pi / 4, dtype=dtype),
            },
            "projection": jax.random.normal(keys[3], (n, self.features), dtype=dtype)
            / math.sqrt(n),
            "residual_scale": jnp.asarray(self.residual_scale, dtype=dtype),
        }

    def _encode(self, parameters, inputs):
        code = jnp.tanh(
            get_compute_kernel().matmul(inputs, parameters["down_kernel"], xp=jnp)
            + parameters["down_bias"]
        )
        return code, math.pi * jnp.tanh(
            get_compute_kernel().matmul(code, parameters["angle_kernel"], xp=jnp)
            + parameters["angle_bias"]
        )

    def __call__(self, parameters, inputs):
        inputs = jnp.asarray(inputs)
        _input(inputs, self.features)
        shape = inputs.shape
        flat = inputs.reshape((-1, self.features))
        size = self.batch_chunk_size
        if size is not None and flat.shape[0] > size:
            count = flat.shape[0] // size * size
            chunks = flat[:count].reshape((-1, size, self.features))
            output = jax.lax.map(lambda x: self._forward(parameters, x), chunks).reshape(
                (count, self.features)
            )
            if count < flat.shape[0]:
                output = jnp.concatenate((output, self._forward(parameters, flat[count:])), axis=0)
            return output.reshape(shape)
        return self._forward(parameters, inputs)

    def _forward(self, parameters, inputs):
        code, angles = self._encode(parameters["encoder"], inputs)
        axes = (
            get_compute_kernel().matmul(code, parameters["axes"]["kernel"], xp=jnp)
            + parameters["axes"]["bias"]
        )
        coordinates = jnp.concatenate((angles, axes), axis=-1)
        value = (
            coordinates.astype(jnp.float32)
            if coordinates.dtype in {jnp.float16, jnp.bfloat16}
            else coordinates
        )
        quantum = self.graph.evaluate(value, backend="jax")
        return (
            inputs
            + parameters["residual_scale"]
            * (get_compute_kernel().matmul(quantum, parameters["projection"], xp=jnp))
        ).astype(inputs.dtype)

    def metric_product(self, parameters, inputs, tangent):
        inputs = jnp.asarray(inputs)
        _input(inputs, self.features)
        encoder = parameters["encoder"]
        _vector(encoder, tangent)
        rows = math.prod(inputs.shape[:-1])
        if not rows:
            raise ValidationError("Metric products need a nonempty input batch.")

        def coordinates(p):
            return self._encode(p, inputs)[1]

        _, output = jax.jvp(coordinates, (encoder,), (tangent,))
        _, pullback = jax.vjp(coordinates, encoder)
        return {k: v / (4 * rows) for k, v in pullback(output)[0].items()}

    def natural_gradient(self, parameters, inputs, gradient, **options):
        _vector(parameters["encoder"], gradient)
        like = parameters["encoder"]["down_kernel"]
        return solve_metric(
            lambda v: self.metric_product(parameters, inputs, v),
            gradient,
            _TensorOps("jax", like),
            **options,
        )


def _dtype(dtype):
    if jnp.dtype(dtype) not in {jnp.dtype("float32"), jnp.dtype("float64")}:
        raise ValidationError("Initialization requires float32/float64.")
    if jnp.dtype(dtype) == jnp.float64 and not jax.config.x64_enabled:
        raise ValidationError("Enable JAX x64 before requesting float64 parameters.")


def _input(value, features):
    if (
        value.ndim < 1
        or value.shape[-1] != features
        or value.dtype
        not in {jnp.dtype(d) for d in (jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64)}
    ):
        raise ValidationError(f"Expected (..., {features}) real floating arrays.")


def _vector(parameters, vector):
    if not isinstance(vector, dict) or set(vector) != set(parameters):
        raise ValidationError("Encoder vectors must match encoder parameter names.")
    if any(
        jnp.shape(vector[k]) != p.shape or jnp.asarray(vector[k]).dtype != p.dtype
        for k, p in parameters.items()
    ):
        raise ValidationError("Encoder vectors must match parameter shapes and dtypes.")
