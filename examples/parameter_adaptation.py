"""Retie imported quantum symbols and freeze trained weights for hybrid learning.

Run with --interface numpy (default), torch or jax.
Add --gradient-method adjoint for reversible NumPy simulator differentiation.
"""

import argparse

import numpy as np

from openqml import LayerCheckpoint, LayerSpec, Parameter, expectations, plan_conversion


def build_checkpoint():
    source = LayerSpec.from_program(
        'OPENQASM 3.0; include "stdgates.inc"; '
        "input float feature; input float alpha; input float beta; input float bias; "
        "qubit q; ry(feature) q; ry(alpha) q; ry(beta) q; ry(bias) q; gphase(beta/7);",
        ("feature",),
        framework="openqasm3",
        parser="builtin",
    )
    feature, gain = Parameter("feature"), Parameter("gain")
    spec = source.reparameterize(
        {"feature": feature.sin(), "alpha": feature * gain, "beta": gain**2},
        input_parameters=("feature",),
        weight_parameters=("bias", "gain"),
    )
    trained = LayerCheckpoint(spec, {"bias": 0.1, "gain": 0.3}, "float32")
    # Frozen values become IR constants; only gain remains an optimizer parameter.
    return trained.freeze(["bias"])


def run(interface, gradient_method="auto"):
    checkpoint = LayerCheckpoint.from_json(build_checkpoint().to_json())
    options = (
        {} if gradient_method == "auto" else {"backend": "numpy", "gradient_method": "adjoint"}
    )
    print("Trainable quantum weights:", checkpoint.spec.weight_parameters)
    if interface == "numpy":
        x, g = 0.4, checkpoint.bindings["gain"]
        result = expectations(
            checkpoint.spec.circuit, checkpoint.spec.observables, {"feature": x, "gain": g}
        )
        bias = float(np.float32(0.1))
        np.testing.assert_allclose(result, [np.cos(np.sin(x) + x * g + g**2 + bias)])
        print("Adapted quantum readout:", result)
        if gradient_method == "adjoint":
            from openqml import adjoint_vjp

            print(
                "Adjoint weight gradient:",
                adjoint_vjp(
                    checkpoint.spec.circuit,
                    checkpoint.spec.observables,
                    [{"feature": x, "gain": g}],
                    [[1.0]],
                    parameters=checkpoint.spec.weight_parameters,
                ),
            )
    elif interface == "torch":
        import torch

        from openqml.interfaces.torch import QuantumLayer

        torch.manual_seed(3)
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 1), QuantumLayer.from_checkpoint(checkpoint, **options)
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(3):
            optimizer.zero_grad()
            loss = model(torch.ones(5, 2)).square().mean()
            loss.backward()
            optimizer.step()
        checkpoint = model[1].checkpoint()
        print("Joint PyTorch training loss:", float(loss.detach()))
    else:
        import jax
        import jax.numpy as jnp

        from openqml.interfaces.jax import QuantumLayer

        layer, weights = QuantumLayer.from_checkpoint(checkpoint, **options)
        params = {"quantum": weights, "encoder": jnp.full((2, 1), 0.1)}

        def loss(p):
            return jnp.square(layer(p["quantum"], jnp.ones((5, 2)) @ p["encoder"])).mean()

        gradient = jax.jit(jax.grad(loss))
        for _ in range(3):
            params = jax.tree_util.tree_map(lambda p, g: p - 0.01 * g, params, gradient(params))
        checkpoint = layer.checkpoint(params["quantum"])
        print("Joint JAX training loss:", float(loss(params)))

    plan = plan_conversion(checkpoint.spec.circuit, target="openqasm3")
    plan.export()
    print("Portable OpenQASM export passed compatibility checks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", choices=("numpy", "torch", "jax"), default="numpy")
    parser.add_argument("--gradient-method", choices=("auto", "adjoint"), default="auto")
    args = parser.parse_args()
    run(args.interface, args.gradient_method)
