"""Compile connected GHZ readouts once, train with native AD, export source IR."""

import argparse

import numpy as np

from openqml import Circuit, LayerSpec, Parameter, PauliPlan, PauliString, convert


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", choices=("numpy", "torch", "jax"), default="numpy")
    parser.add_argument("--qubits", type=int, default=512)
    parser.add_argument(
        "--framework", choices=("ir", "qiskit", "cirq", "pennylane", "openqasm3"), default="ir"
    )
    args = parser.parse_args()
    if args.qubits < 2:
        parser.error("Use at least two quantum wires.")
    c = Circuit(args.qubits).ry(Parameter("feature") * Parameter("weight"), 0)
    for q in range(args.qubits - 1):
        c = c.cx(q, q + 1)
    obs = (PauliString("I" * (args.qubits - 1) + "Z"), PauliString("X" * args.qubits))
    # Binding a native import keeps this example portable to tape-based SDKs.
    native = convert(c, target=args.framework, bindings={"feature": 0.4, "weight": 0.7}).value
    imported = convert(native, source=args.framework, target="ir").value
    compiled = PauliPlan(imported, obs)
    np.testing.assert_allclose(compiled.evaluate(), [np.cos(0.28), np.sin(0.28)], atol=1e-12)
    print(compiled.to_json())
    source_export = convert(c, target="openqasm3").value
    assert f"qubit[{args.qubits}]" in source_export
    spec = LayerSpec(c, ("feature",), ("weight",), obs)
    if args.interface == "torch":
        import torch

        from openqml.interfaces.torch import QuantumLayer

        model = torch.nn.Sequential(
            torch.nn.Linear(4, 1, dtype=torch.float64),
            QuantumLayer.from_spec(
                spec, execution_method="pauli", max_qubits=args.qubits, dtype=torch.float64
            ),
            torch.nn.Linear(2, 1, dtype=torch.float64),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        data = torch.ones((8, 4), dtype=torch.float64)
        optimizer.zero_grad()
        model(data).square().mean().backward()
        optimizer.step()
        assert model[1].checkpoint().spec == spec
    elif args.interface == "jax":
        import jax
        import jax.numpy as jnp

        from openqml.interfaces.jax import QuantumLayer

        layer = QuantumLayer.from_spec(spec, execution_method="pauli", max_qubits=args.qubits)
        weights = layer.init(jax.random.key(3))
        inputs = jnp.array([[0.2], [0.6]], dtype=weights.dtype)
        # A nonconstant readout loss; both outputs share one real coefficient DAG.
        loss = jax.jit(jax.value_and_grad(lambda w: layer(w, inputs)[..., 0].mean()))
        _, gradient = loss(weights)
        weights = weights - 0.01 * gradient
        assert layer.checkpoint(weights).spec == spec
    print(f"Verified {args.qubits} connected wires without allocating a state vector.")


if __name__ == "__main__":
    main()
