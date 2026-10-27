"""Reuse portable blocks and trained weights across hybrid interfaces.

Run with --interface numpy (default), torch or jax.
"""

import argparse

import numpy as np

from openqml import Circuit, LayerCheckpoint, LayerSpec, Parameter, expectations


def build_checkpoint():
    encoder = LayerSpec.from_program(
        "OPENQASM 3.0; input float feature; input float theta; qubit q; ry(feature+theta) q;",
        ("feature",),
        framework="openqasm3",
        parser="builtin",
    )
    ansatz = LayerSpec.from_program(
        Circuit(1).ry(Parameter("feature") + Parameter("phi"), 0).to_json(),
        ("feature",),
        framework="json",
    )
    return LayerCheckpoint(encoder, [0.2], "float32").compose(
        LayerCheckpoint(ansatz, [0.3], "float32"),
        prefix="ansatz",
        shared_inputs={"feature": "feature"},
    )


def run(interface):
    checkpoint = build_checkpoint()
    print("Inputs:", checkpoint.spec.input_parameters)
    print("Weights:", checkpoint.spec.weight_parameters)
    if interface == "numpy":
        values = checkpoint.bindings | {"feature": 0.4}
        result = expectations(checkpoint.spec.circuit, checkpoint.spec.observables, values)
        np.testing.assert_allclose(result, [np.cos(0.8 + sum(checkpoint.weights))])
        print("Composed quantum readout:", result)
    elif interface == "torch":
        import torch

        from openqml.interfaces.torch import QuantumLayer

        torch.manual_seed(3)
        model = torch.nn.Sequential(torch.nn.Linear(2, 1), QuantumLayer.from_checkpoint(checkpoint))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss = model(torch.ones(5, 2)).square().mean()
        loss.backward()
        optimizer.step()
        print("Joint PyTorch training loss:", float(loss.detach()))
    else:
        import jax
        import jax.numpy as jnp

        from openqml.interfaces.jax import QuantumLayer

        layer, weights = QuantumLayer.from_checkpoint(checkpoint)
        params = {"quantum": weights, "encoder": jnp.full((2, 1), 0.1)}

        def loss(p):
            return jnp.square(layer(p["quantum"], jnp.ones((5, 2)) @ p["encoder"])).mean()

        gradients = jax.jit(jax.grad(loss))(params)
        updated = jax.tree_util.tree_map(lambda p, g: p - 0.01 * g, params, gradients)
        print("Joint JAX training loss:", float(loss(updated)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", choices=("numpy", "torch", "jax"), default="numpy")
    run(parser.parse_args().interface)
