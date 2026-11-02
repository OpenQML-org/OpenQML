"""Functional deep quantum residual models with ordinary JAX parameter trees."""

import math

import jax
import jax.numpy as jnp

from ..errors import ValidationError
from ..ir import integer


class QuantumDeepNetwork:
    def __init__(
        self,
        input_size,
        output_size,
        *,
        num_qubits=4,
        depth=4,
        reupload=True,
        initial_mixing=0.1,
        backend=None,
        max_memory_mb=256,
    ):
        from .jax import QuantumLayer

        self.input_size = integer(input_size, "input_size", 1)
        self.output_size = integer(output_size, "output_size", 1)
        self.num_qubits = integer(num_qubits, "num_qubits", 1)
        self.quantum = QuantumLayer.deep(
            num_qubits,
            depth,
            reupload=reupload,
            initial_mixing=initial_mixing,
            backend=backend,
            max_memory_mb=max_memory_mb,
        )

    def init(self, key, *, dtype=jnp.float32):
        encoder, quantum, readout = jax.random.split(key, 3)
        return {
            "encoder": jax.random.normal(encoder, (self.input_size, self.num_qubits), dtype)
            / math.sqrt(self.input_size),
            "encoder_bias": jnp.zeros((self.num_qubits,), dtype),
            "quantum": self.quantum.init(quantum, dtype=dtype),
            "readout": jax.random.normal(readout, (self.num_qubits, self.output_size), dtype)
            / math.sqrt(self.num_qubits),
            "readout_bias": jnp.zeros((self.output_size,), dtype),
        }

    def __call__(self, parameters, inputs):
        inputs = jnp.asarray(inputs)
        if inputs.ndim < 1 or inputs.shape[-1] != self.input_size:
            raise ValidationError("Input width differs from the deep quantum encoder.")
        controls = math.pi * jnp.tanh(inputs @ parameters["encoder"] + parameters["encoder_bias"])
        features = self.quantum(parameters["quantum"], controls)
        return features @ parameters["readout"] + parameters["readout_bias"]
