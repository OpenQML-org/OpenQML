"""Train a classical readout and quantum weights in one PyTree. Requires openqml[jax]."""

import jax
import jax.numpy as jnp

from openqml.interfaces.jax import QuantumLayer

layer = QuantumLayer.hardware_efficient(2, depth=1)
parameters = {
    "quantum": layer.init(jax.random.key(17)),
    "readout": jnp.array([[0.2], [-0.1]], dtype=jnp.float32),
    "bias": jnp.zeros(1, dtype=jnp.float32),
}
inputs = jnp.array([[0.1, -0.2], [0.5, 0.3], [-0.4, 0.7]], dtype=jnp.float32)
targets = jnp.array([[0.2], [0.4], [-0.3]], dtype=jnp.float32)


def loss(parameters):
    features = layer(parameters["quantum"], inputs)
    prediction = features @ parameters["readout"] + parameters["bias"]
    return jnp.mean((prediction - targets) ** 2)


train_step = jax.jit(jax.value_and_grad(loss))
initial = float(loss(parameters))
for step in range(80):
    value, gradients = train_step(parameters)
    parameters = jax.tree_util.tree_map(lambda p, g: p - 0.1 * g, parameters, gradients)
    if step % 20 == 0 or step == 79:
        print(f"step={step:02d} loss={float(value):.6f}")
assert float(loss(parameters)) < initial
