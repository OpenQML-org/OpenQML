"""Train classical encoders and quantum weight generators with one optimizer.

Run with python examples/conditional_weights.py --interface torch (or jax).
Both models reuse the same portable definition imported from OpenQASM 3.
"""

import argparse

from openqml import LayerSpec

SOURCE = """OPENQASM 3.0;
include "stdgates.inc";
input float[64] feature;
input float[64] weight;
qubit q;
ry(feature + weight) q;
"""


def torch_example(spec):
    import torch

    from openqml.interfaces.torch import QuantumLayer

    torch.manual_seed(7)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(3, 1)
            self.generator = torch.nn.Linear(3, 1)
            self.quantum = QuantumLayer.from_spec(spec, batch_chunk_size=4)
            # This model uses generated weights exclusively.
            self.quantum.weights.requires_grad_(False)

        def forward(self, data):
            return self.quantum.functional(self.generator(data), self.encoder(data))

    model, data = Model(), torch.ones(12, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(10):
        optimizer.zero_grad()
        loss = model(data).square().mean()
        loss.backward()
        optimizer.step()
    print("PyTorch conditional quantum loss:", loss.item())


def jax_example(spec):
    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    import optax

    from openqml.interfaces.flax import QuantumLayer

    class Model(nn.Module):
        @nn.compact
        def __call__(self, data):
            features = nn.Dense(1, name="encoder")(data)
            weights = nn.Dense(1, name="generator")(data)
            return QuantumLayer.from_spec(spec, batch_chunk_size=4)(features, weights=weights)

    model, data = Model(), jnp.ones((12, 3))
    params = model.init(jax.random.key(7), data)["params"]
    optimizer = optax.adam(0.01)
    state = optimizer.init(params)

    @jax.jit
    def step(params, state):
        def loss_fn(p):
            return jnp.square(model.apply({"params": p}, data)).mean()

        loss, gradients = jax.value_and_grad(loss_fn)(params)
        updates, state = optimizer.update(gradients, state, params)
        return optax.apply_updates(params, updates), state, loss

    for _ in range(10):
        params, state, loss = step(params, state)
    print("Flax conditional quantum loss:", float(loss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", choices=("torch", "jax"), default="torch")
    args = parser.parse_args()
    spec = LayerSpec.from_program(SOURCE, ("feature",), framework="openqasm3", parser="builtin")
    (torch_example if args.interface == "torch" else jax_example)(spec)
