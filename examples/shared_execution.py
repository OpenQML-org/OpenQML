"""Share repeated quantum functions and pack independent native parameter lanes."""

import argparse

import numpy as np

from openqml import Circuit, ExpectationPlan, Hamiltonian, LayerSpec, Parameter, PauliString


def workload(blocks=64):
    width, shared = blocks * 2, blocks // 2
    names = ("shared",) + tuple(f"weight{i}" for i in range(blocks - shared))
    x = np.array([[0, 1], [1, 0]])
    matrix = np.cos(0.035) * np.eye(4) - 1j * np.sin(0.035) * np.kron(x, x)
    ry = np.array([[np.cos(0.05), -np.sin(0.05)], [np.sin(0.05), np.cos(0.05)]])
    matrix = matrix @ np.diag([1, 1, 1, -1]) @ np.kron(ry, np.eye(2))
    c, terms = Circuit(width), []
    for i in range(blocks):
        name = names[0] if i < shared else names[1 + i - shared]
        c = c.ry(Parameter("feature") * Parameter(name), 2 * i).unitary(matrix, 2 * i, 2 * i + 1)
        terms.append(PauliString("II" * i + "ZI" + "II" * (blocks - i - 1), 1 / blocks))
    return LayerSpec(
        c, ("feature",), names, (PauliString("ZI" * blocks), Hamiltonian(tuple(terms)))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", choices=("numpy", "torch", "jax"), default="numpy")
    args = parser.parse_args()
    spec = workload()
    plan = ExpectationPlan(spec.circuit, spec.observables, max_dense_batch=16)
    weights = np.linspace(-0.2, 0.3, len(spec.weight_parameters))
    features = np.array([-0.3, 0.2, 0.4])
    bindings = dict(zip(spec.weight_parameters, weights, strict=True)) | {"feature": features}
    repeated = np.concatenate((np.repeat(weights[0], 32), weights[1:]))
    factors = np.cos(features[:, None] * repeated + 0.1) * np.cos(0.07)
    expected = np.stack((np.prod(factors, axis=-1), np.mean(factors, axis=-1)), axis=-1)
    np.testing.assert_allclose(plan.evaluate(bindings), expected, atol=3e-13)
    assert len(plan.groups) == 64 and len(plan.dense_batches) == 1
    assert len(plan.dense_batches[0].parameter_slots) == 33
    if args.interface == "torch":
        import torch

        from openqml.interfaces.torch import QuantumLayer

        layer = QuantumLayer.from_spec(
            spec,
            execution_method="planned",
            planning_options={"max_dense_batch": 16},
            max_qubits=128,
            dtype=torch.float64,
        )
        values = torch.tensor(weights, dtype=torch.float64, requires_grad=True)
        inputs = torch.tensor(features[:, None], dtype=torch.float64, requires_grad=True)
        torch.testing.assert_close(layer.functional(values, inputs), torch.tensor(expected))
        gradients = torch.autograd.grad(
            layer.functional(values, inputs).square().sum(), (values, inputs)
        )
        assert all(torch.isfinite(g).all() for g in gradients)
        assert layer.checkpoint().spec == spec
    elif args.interface == "jax":
        import jax
        import jax.numpy as jnp

        from openqml.interfaces.jax import QuantumLayer

        layer = QuantumLayer.from_spec(
            spec,
            execution_method="planned",
            planning_options={"max_dense_batch": 16},
            max_qubits=128,
        )
        values, inputs = (
            jnp.array(weights, dtype=jnp.float32),
            jnp.array(features[:, None], dtype=jnp.float32),
        )
        np.testing.assert_allclose(jax.jit(layer)(values, inputs), expected, atol=3e-6)
        gradients = jax.jit(jax.grad(lambda w, x: jnp.square(layer(w, x)).sum(), argnums=(0, 1)))(
            values, inputs
        )
        assert all(np.isfinite(g).all() for g in gradients)
        assert layer.checkpoint(values).spec == spec
    print("Verified 128 source wires: 64 source groups, 33 unique dense lanes, one template.")


if __name__ == "__main__":
    main()
