"""Combine a wide Pauli block and opaque local matrices in one trainable plan."""

import argparse

import numpy as np

from openqml import (
    Circuit,
    ExpectationPlan,
    Hamiltonian,
    LayerSpec,
    Parameter,
    PauliString,
    convert,
)


def workload(ghz_qubits=64, blocks=32):
    width = ghz_qubits + 2 * blocks
    theta = Parameter("feature") * Parameter("weight")
    c = Circuit(width).ry(theta, 0)
    for q in range(ghz_qubits - 1):
        c = c.cx(q, q + 1)
    # A fixed entangling matrix, kept opaque in the source IR.
    x = np.array([[0, 1], [1, 0]])
    matrix = np.cos(0.07 / 2) * np.eye(4) - 1j * np.sin(0.07 / 2) * np.kron(x, x)
    angle = 0.1
    ry = np.array([[np.cos(angle / 2), -np.sin(angle / 2)], [np.sin(angle / 2), np.cos(angle / 2)]])
    matrix = matrix @ np.diag([1, 1, 1, -1]) @ np.kron(ry, np.eye(2))
    terms = []
    for i in range(blocks):
        q = ghz_qubits + 2 * i
        c = c.ry(theta, q).unitary(matrix, q, q + 1)
        terms.append(PauliString("I" * q + "Z" + "I" * (width - q - 1), 1 / blocks))
    obs = (
        PauliString("I" * (ghz_qubits - 1) + "Z" + "ZI" * blocks),
        PauliString("I" * (ghz_qubits - 1) + "Z" + "I" * (2 * blocks)),
        Hamiltonian(tuple(terms)),
    )
    return LayerSpec(c, ("feature",), ("weight",), obs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", choices=("numpy", "torch", "jax"), default="numpy")
    parser.add_argument("--ghz-qubits", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=32)
    args = parser.parse_args()
    if min(args.ghz_qubits, args.blocks) < 1:
        parser.error("Use positive wire/block counts.")
    spec = workload(args.ghz_qubits, args.blocks)
    plan = ExpectationPlan(spec.circuit, spec.observables, max_memory_mb=1)
    theta = np.array([0.04, 0.12, -0.08])
    local = np.cos(theta + 0.1) * np.cos(0.07)
    expected = np.stack((np.cos(theta) * local**args.blocks, np.cos(theta), local), axis=-1)
    np.testing.assert_allclose(plan.evaluate(dict(feature=theta, weight=1)), expected, atol=3e-13)
    assert {g.method for g in plan.groups} == {"pauli", "statevector"}
    # Source exports remain complete, including the opaque matrix blocks.
    assert convert(plan.circuit, target="json").circuit == spec.circuit
    mirror = workload(3, 2)
    verification = ExpectationPlan(mirror.circuit, mirror.observables).verify(
        dict(feature=0.4, weight=0.3)
    )
    assert verification.matched
    print(plan.to_json())
    print(verification.to_json())
    if args.interface == "torch":
        import torch

        from openqml.interfaces.torch import QuantumLayer

        model = torch.nn.Sequential(
            torch.nn.Linear(4, 1, dtype=torch.float64),
            QuantumLayer.from_spec(
                spec,
                execution_method="planned",
                max_qubits=spec.circuit.num_qubits,
                dtype=torch.float64,
                batch_chunk_size=8,
            ),
            torch.nn.Linear(3, 1, dtype=torch.float64),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        optimizer.zero_grad()
        model(torch.ones((8, 4), dtype=torch.float64)).square().mean().backward()
        optimizer.step()
        assert model[1].checkpoint().spec == spec
    elif args.interface == "jax":
        import jax
        import jax.numpy as jnp

        from openqml.interfaces.jax import QuantumLayer

        layer = QuantumLayer.from_spec(
            spec, execution_method="planned", max_qubits=spec.circuit.num_qubits
        )
        weights = layer.init(jax.random.key(4))
        x = jnp.array([[0.1], [0.2]], dtype=weights.dtype)
        gradient = jax.jit(jax.grad(lambda w: layer(w, x)[..., 0].sum()))(weights)
        assert layer.checkpoint(weights - 0.01 * gradient).spec == spec
    print(
        f"Verified {spec.circuit.num_qubits} source wires with "
        f"{len(plan.groups)} shared local circuits."
    )


if __name__ == "__main__":
    main()
