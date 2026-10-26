"""Qiskit nonlinear feature map inside Flax; one Optax state updates all parameters."""

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from sdk_programs import build_qiskit_program

from openqml import as_circuit
from openqml.interfaces.flax import QuantumLayer

native, inputs, weights = build_qiskit_program()
circuit = as_circuit(native)
input_names = tuple(p.name for p in inputs)
weight_names = tuple(p.name for p in weights)


class Model(nn.Module):
    @nn.compact
    def __call__(self, x):
        features = jnp.tanh(nn.Dense(4)(nn.gelu(nn.Dense(12)(x))))
        quantum = QuantumLayer(circuit, input_names, weight_names)(features)
        return nn.Dense(1)(nn.gelu(nn.Dense(8)(jnp.concatenate((features, quantum), axis=-1))))


x = jax.random.normal(jax.random.key(23), (32, 8))
target = 0.4 * x[:, :1] - 0.2 * x[:, 1:2] + 0.1 * x[:, 2:3] ** 2
model = Model()
params = model.init(jax.random.key(17), x)
optimizer = optax.adam(0.025)
state = optimizer.init(params)


@jax.jit
def train(params, state):
    loss, grads = jax.value_and_grad(lambda p: jnp.mean((model.apply(p, x) - target) ** 2))(params)
    updates, state = optimizer.update(grads, state, params)
    return optax.apply_updates(params, updates), state, loss


losses = []
for step in range(60):
    params, state, loss = train(params, state)
    losses.append(float(loss))
    if step % 20 == 0 or step == 59:
        print(f"step={step:02d} loss={float(loss):.7f}")
assert losses[-1] < 0.2 * losses[0]
print("One Flax parameter tree and one Optax optimizer include every classical and quantum layer.")
