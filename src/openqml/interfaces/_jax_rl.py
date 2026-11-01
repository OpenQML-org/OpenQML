"""Functional quantum policies for JIT/vmap/grad and Optax parameter trees."""

import math

import jax
import jax.numpy as jnp

from ..errors import ValidationError
from ..ir import integer
from ..rl import ActorCriticOutput, surrogate_fit_loss
from ._jax_hybrid import _dtype, _input


class QuantumActorCritic:
    def __init__(
        self,
        observation_size,
        action_size,
        *,
        num_qubits=4,
        depth=2,
        hidden_features=64,
        continuous=False,
        quantum_layer=None,
        backend="auto",
        max_memory_mb=256,
    ):
        self.observation_size = integer(observation_size, "observation_size", 1)
        self.action_size = integer(action_size, "action_size", 1)
        self.num_qubits = integer(num_qubits, "num_qubits", 1)
        self.hidden_features = integer(hidden_features, "hidden_features", 1)
        if type(continuous) is not bool:
            raise ValidationError("continuous must be a boolean.")
        self.continuous = continuous
        if quantum_layer is None:
            from .jax import QuantumLayer

            quantum_layer = QuantumLayer.identity_blocks(
                num_qubits,
                depth,
                backend=backend,
                max_memory_mb=max_memory_mb,
                gradient_chunk_size=256,
            )
        if not callable(quantum_layer) or not callable(getattr(quantum_layer, "init", None)):
            raise ValidationError("Provide a functional quantum layer with init(key, dtype=...).")
        self.quantum = quantum_layer

    def init(self, key, *, dtype=jnp.float32):
        _dtype(dtype)
        keys = jax.random.split(key, 6)
        obs, n, actions, hidden = (
            self.observation_size,
            self.num_qubits,
            self.action_size,
            self.hidden_features,
        )

        def dense(key, a, b, scale=1.0):
            return {
                "kernel": scale * jax.random.normal(key, (a, b), dtype) / math.sqrt(a),
                "bias": jnp.zeros((b,), dtype),
            }

        parameters = {
            "encoder": dense(keys[0], obs, n),
            "input_scale": jnp.ones((n,), dtype),
            "quantum": self.quantum.init(keys[1], dtype=dtype),
            "actor": dense(keys[2], n, actions, 0.01),
            "critic_hidden": dense(keys[3], obs, hidden),
            "critic_value": dense(keys[4], hidden, 1),
        }
        if self.continuous:
            parameters["log_std"] = jnp.full((actions,), -0.5, dtype)
        return parameters

    def __call__(self, parameters, observations):
        _input(observations, self.observation_size)

        def linear(name, inputs):
            p = parameters[name]
            return inputs @ p["kernel"] + p["bias"]

        controls = math.pi * jnp.tanh(linear("encoder", observations)) * parameters["input_scale"]
        features = self.quantum(parameters["quantum"], controls)
        if features.shape != controls.shape:
            raise ValidationError("Quantum policies require one readout per declared quantum wire.")
        policy = linear("actor", features)
        value = linear("critic_value", jnp.tanh(linear("critic_hidden", observations)))[..., 0]
        log_std = (
            jnp.broadcast_to(jnp.clip(parameters["log_std"], -20, 2), policy.shape)
            if self.continuous
            else None
        )
        return ActorCriticOutput(policy, value, log_std)


class QuantumSurrogate:
    """Functional counterpart of the explicit PyTorch response surrogate."""

    def __init__(self, quantum, surrogate, *, binary_logits=False):
        if not callable(quantum) or not callable(surrogate) or type(binary_logits) is not bool:
            raise ValidationError(
                "Provide quantum/surrogate callables and a boolean binary_logits."
            )
        self.quantum, self.surrogate, self.binary_logits = quantum, surrogate, binary_logits

    def __call__(self, parameters, controls, *, mode="quantum"):
        if mode == "quantum":
            return self.quantum(controls)
        if mode not in {"surrogate", "straight_through"}:
            raise ValidationError("mode must be quantum, surrogate or straight_through.")
        fixed = jax.tree.map(jax.lax.stop_gradient, parameters)
        prediction = self.surrogate(fixed, controls)
        prediction = jax.nn.sigmoid(prediction) if self.binary_logits else prediction
        if mode == "surrogate":
            return prediction
        measured = jax.lax.stop_gradient(self.quantum(controls))
        if measured.shape != prediction.shape:
            raise ValidationError("Measured and surrogate readouts must have identical shapes.")
        return measured + (prediction - jax.lax.stop_gradient(prediction))

    def fit_loss(self, parameters, controls, measurements):
        return surrogate_fit_loss(
            self.surrogate(parameters, jax.lax.stop_gradient(controls)),
            measurements,
            binary_logits=self.binary_logits,
        )
