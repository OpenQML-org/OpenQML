"""Functional JAX tensor-network layers with ordinary Optax parameter trees."""

import math

import jax
import jax.numpy as jnp

from ..errors import ValidationError
from ..ir import integer
from ..parameters import real
from ..quantum_tree import QuantumTreePlan
from ..tensor_train import TensorTrainMatrix, TensorTrainPlan
from ._jax_hybrid import _dtype, _input
from ._spec import LayerSpecMixin


class TreeQuantumLayer:
    def __init__(
        self,
        num_qubits,
        *,
        num_outputs=4,
        batch_chunk_size=256,
        max_memory_mb=256,
        max_work=2_000_000,
    ):
        self.plan = QuantumTreePlan(num_qubits, num_outputs, max_memory_mb, max_work)
        self.num_outputs = num_outputs
        self.batch_chunk_size = (
            None if batch_chunk_size is None else integer(batch_chunk_size, "batch_chunk_size", 1)
        )

    def init(self, key, *, scale=0.02, dtype=jnp.float32):
        _dtype(dtype)
        scale = real(scale)
        if scale < 0:
            raise ValidationError("scale must be nonnegative.")
        value = scale * jax.random.normal(key, (self.plan.num_parameters,), dtype)
        count = 12 * self.num_outputs * (self.plan.num_qubits - 1)
        if count:
            nodes = value[:count].reshape(self.num_outputs, self.plan.num_qubits - 1, 12)
            value = value.at[:count].set(nodes.at[..., 6:9].add(math.pi / 4).reshape(-1))
        return value.at[count:].set(math.pi / 4)

    def __call__(self, weights, inputs):
        weights, inputs = jnp.asarray(weights), jnp.asarray(inputs)
        _input(inputs, self.plan.num_qubits)
        _input(weights, self.plan.num_parameters)
        if weights.ndim != 1 or self.batch_chunk_size is None:
            return self.plan.evaluate(weights, inputs, backend="jax").astype(inputs.dtype)
        shape = inputs.shape[:-1]
        flat = inputs.reshape((-1, self.plan.num_qubits))
        size = self.batch_chunk_size
        if flat.shape[0] <= size:
            return self.plan.evaluate(weights, inputs, backend="jax").astype(inputs.dtype)
        count = flat.shape[0] // size * size
        output = jax.lax.map(
            lambda x: self.plan.evaluate(weights, x, backend="jax"),
            flat[:count].reshape((-1, size, self.plan.num_qubits)),
        ).reshape((count, self.num_outputs))
        if count < flat.shape[0]:
            output = jnp.concatenate(
                (output, self.plan.evaluate(weights, flat[count:], backend="jax")), axis=0
            )
        return output.reshape(shape + (self.num_outputs,)).astype(inputs.dtype)

    def to_spec(self):
        return self.plan.to_spec()

    def checkpoint(self, weights):
        return LayerSpecMixin.checkpoint(self, weights)


class TensorTrainLinear:
    def __init__(
        self,
        in_features,
        out_features,
        *,
        bond_dimension=8,
        input_modes=None,
        output_modes=None,
        bias=True,
        batch_chunk_size=256,
        max_memory_mb=256,
        max_work=2_000_000,
    ):
        if (input_modes is None) != (output_modes is None) or type(bias) is not bool:
            raise ValidationError("Specify both mode lists and a boolean bias.")
        options = dict(
            bond_dimension=bond_dimension, max_memory_mb=max_memory_mb, max_work=max_work
        )
        self.plan = (
            TensorTrainPlan.from_features(in_features, out_features, **options)
            if input_modes is None
            else TensorTrainPlan(
                input_modes,
                output_modes,
                in_features=in_features,
                out_features=out_features,
                **options,
            )
        )
        self.in_features, self.out_features, self.bias = (
            self.plan.in_features,
            self.plan.out_features,
            bias,
        )
        self.batch_chunk_size = (
            self.plan.batch_capacity
            if batch_chunk_size is None
            else min(self.plan.batch_capacity, integer(batch_chunk_size, "batch_chunk_size", 1))
        )
        self.initial_compression = None

    def init(self, key, *, dtype=jnp.float32):
        _dtype(dtype)
        count = len(self.plan.core_shapes)
        keys = jax.random.split(key, count)
        factor = math.prod(self.plan.ranks[1:-1]) ** (-1 / (2 * count))
        params = {
            "cores": tuple(
                jax.random.normal(k, s, dtype) * factor / math.sqrt(s[1])
                for k, s in zip(keys, self.plan.core_shapes, strict=True)
            )
        }
        if self.bias:
            params["bias"] = jnp.zeros((self.out_features,), dtype)
        return params

    def __call__(self, parameters, inputs):
        inputs = jnp.asarray(inputs)
        _input(inputs, self.in_features)
        shape = inputs.shape[:-1]
        flat = inputs.reshape((-1, self.in_features))
        size = self.batch_chunk_size
        if flat.shape[0] > size:
            count = flat.shape[0] // size * size
            output = jax.lax.map(
                lambda x: self.plan.apply(parameters["cores"], x, backend="jax"),
                flat[:count].reshape((-1, size, self.in_features)),
            ).reshape((count, self.out_features))
            if count < flat.shape[0]:
                output = jnp.concatenate(
                    (output, self.plan.apply(parameters["cores"], flat[count:], backend="jax")),
                    axis=0,
                )
            output = output.reshape(shape + (self.out_features,))
        else:
            output = self.plan.apply(parameters["cores"], inputs, backend="jax")
        return output + parameters["bias"] if self.bias else output

    @classmethod
    def from_dense(cls, weight, bias=None, *, dtype=jnp.float32, **options):
        _dtype(dtype)
        compressed = TensorTrainMatrix.from_dense(weight, **options)
        p = compressed.plan
        model = cls(
            p.in_features,
            p.out_features,
            bond_dimension=p.ranks[1:-1],
            input_modes=p.input_modes,
            output_modes=p.output_modes,
            bias=bias is not None,
            max_memory_mb=p.max_memory_mb,
            max_work=p.max_work,
        )
        parameters = {"cores": tuple(jnp.asarray(c, dtype) for c in compressed.cores)}
        if bias is not None:
            parameters["bias"] = jnp.asarray(bias, dtype)
            if parameters["bias"].shape != (p.out_features,):
                raise ValidationError("Dense bias has the wrong shape.")
        model.initial_compression = compressed.to_dict()
        return model, parameters


class TensorNetworkAdapter:
    def __init__(
        self,
        features,
        *,
        num_qubits=64,
        num_outputs=8,
        bond_dimension=8,
        residual_scale=0.01,
        batch_chunk_size=256,
        max_memory_mb=256,
    ):
        self.features = integer(features, "features", 1)
        options = dict(
            bond_dimension=bond_dimension,
            batch_chunk_size=batch_chunk_size,
            max_memory_mb=max_memory_mb,
        )
        self.encoder = TensorTrainLinear(features, num_qubits, **options)
        self.quantum = TreeQuantumLayer(
            num_qubits,
            num_outputs=num_outputs,
            batch_chunk_size=batch_chunk_size,
            max_memory_mb=max_memory_mb,
        )
        self.projection = TensorTrainLinear(num_outputs, features, **options)
        self.residual_scale = real(residual_scale)

    def init(self, key, *, dtype=jnp.float32):
        a, b, c = jax.random.split(key, 3)
        return {
            "encoder": self.encoder.init(a, dtype=dtype),
            "quantum": self.quantum.init(b, dtype=dtype),
            "projection": self.projection.init(c, dtype=dtype),
            "residual_scale": jnp.asarray(self.residual_scale, dtype),
        }

    def __call__(self, parameters, inputs):
        inputs = jnp.asarray(inputs)
        _input(inputs, self.features)
        controls = math.pi * jnp.tanh(self.encoder(parameters["encoder"], inputs))
        features = self.quantum(parameters["quantum"], controls)
        return (
            inputs
            + parameters["residual_scale"] * self.projection(parameters["projection"], features)
        ).astype(inputs.dtype)
