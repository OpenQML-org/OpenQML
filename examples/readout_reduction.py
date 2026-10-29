"""Reduce a declared readout, export its compact IR, and train in source order."""

import argparse

import numpy as np

from openqml import Circuit, LayerSpec, Parameter, PauliString, convert, expectations, plan_readout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", choices=("numpy", "torch", "jax"), default="numpy")
    parser.add_argument(
        "--framework", choices=("ir", "qiskit", "pennylane", "cirq", "openqasm3"), default="ir"
    )
    args = parser.parse_args()
    circuit = (
        Circuit(8)
        .ry(Parameter("unused"), 0)
        .ry(Parameter("feature") * Parameter("weight"), 7)
        .cx(7, 3)
        .rx(Parameter("unused"), 7)
    )
    spec = LayerSpec(circuit, ("feature",), ("unused", "weight"), (PauliString("IIIZIIII"),))
    bindings = {"feature": 0.4, "unused": 0.7, "weight": 0.3}
    native = convert(circuit, target=args.framework, bindings=bindings).value
    imported = convert(native, source=args.framework, target="ir").value
    plan = plan_readout(imported, spec.observables)
    np.testing.assert_allclose(
        expectations(imported, spec.observables),
        expectations(plan.circuit, plan.observables),
        atol=1e-12,
    )
    print(plan.to_json())
    # The compact circuit must travel with its remapped readouts and wire map.
    print(convert(plan.circuit, target="openqasm3").value)
    if args.interface == "torch":
        import torch

        from openqml.interfaces.torch import QuantumLayer

        layer = QuantumLayer.from_spec(spec, reduce_readout=True, dtype=torch.float64)
        optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
        x = torch.tensor([[0.2], [0.5]], dtype=torch.float64, requires_grad=True)
        optimizer.zero_grad()
        layer(x).square().mean().backward()
        optimizer.step()
        assert layer.weights.grad[0] == 0  # Source parameter order is unchanged.
        assert layer.checkpoint().spec == spec
    elif args.interface == "jax":
        import jax
        import jax.numpy as jnp

        from openqml.interfaces.jax import QuantumLayer

        layer = QuantumLayer.from_spec(spec, reduce_readout=True)
        weights = layer.init(jax.random.key(3))
        inputs = jnp.array([[0.2], [0.5]], dtype=weights.dtype)
        gradient = jax.jit(jax.grad(lambda w: jnp.square(layer(w, inputs)).mean()))(weights)
        assert gradient[0] == 0
        weights = weights - 0.01 * gradient
        assert layer.checkpoint(weights).spec == spec


if __name__ == "__main__":
    main()
