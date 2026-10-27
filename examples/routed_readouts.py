"""Train sparse readouts on a packed circuit while retaining source checkpoints."""

import argparse
from itertools import product

import numpy as np

from openqml import Circuit, ExpectationPlan, Hamiltonian, LayerSpec, Parameter, PauliString


def workload(blocks=32):
    rng = np.random.default_rng(18)
    matrix, _ = np.linalg.qr(rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8)))
    labels = ["".join(p) for p in product("IXYZ", repeat=3) if set(p) != {"I"}][:blocks]
    inputs, weights = tuple(f"x{i}" for i in range(blocks)), tuple(f"w{i}" for i in range(blocks))
    c, terms = Circuit(3 * blocks), []
    for i, label in enumerate(labels):
        c = (
            c.ry(Parameter(inputs[i]) * Parameter(weights[i]), 3 * i)
            .rx(0.23, 3 * i + 2)
            .unitary(matrix, 3 * i, 3 * i + 1, 3 * i + 2)
        )
        terms.append(PauliString("III" * i + label + "III" * (blocks - i - 1), 1 / blocks))
    return LayerSpec(c, inputs, weights, (PauliString("".join(labels)), Hamiltonian(tuple(terms))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", choices=("numpy", "torch", "jax"), default="numpy")
    args = parser.parse_args()
    spec = workload()
    plan = ExpectationPlan(spec.circuit, spec.observables)
    report = plan.dense_batches[0].to_dict()
    assert report["requested_readout_pairs"] == 32
    assert report["unrouted_readout_pairs"] == 1024
    mirror = workload(3)
    check = ExpectationPlan(mirror.circuit, mirror.observables).verify(
        {n: 0.3 for n in mirror.input_parameters + mirror.weight_parameters}
    )
    assert check.matched, check.to_json()
    if args.interface == "torch":
        import torch

        from openqml.interfaces.torch import QuantumLayer

        model = torch.nn.Sequential(
            torch.nn.Linear(4, 32, dtype=torch.float64),
            QuantumLayer.from_spec(
                spec, execution_method="planned", max_qubits=96, dtype=torch.float64
            ),
            torch.nn.Linear(2, 1, dtype=torch.float64),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        optimizer.zero_grad()
        model(torch.ones(8, 4, dtype=torch.float64)).square().mean().backward()
        optimizer.step()
        assert model[1].checkpoint().spec == spec
    elif args.interface == "jax":
        import jax
        import jax.numpy as jnp

        from openqml.interfaces.jax import QuantumLayer

        layer = QuantumLayer.from_spec(spec, execution_method="planned", max_qubits=96)
        weights = layer.init(jax.random.key(7))
        x = jnp.ones((8, 32), dtype=weights.dtype) * 0.2
        gradient = jax.jit(jax.grad(lambda w: jnp.square(layer(w, x)).mean()))(weights)
        assert np.isfinite(gradient).all()
        assert layer.checkpoint(weights - 0.01 * gradient).spec == spec
    else:
        values = {
            n: np.linspace(-0.3, 0.2, 8) for n in spec.input_parameters + spec.weight_parameters
        }
        assert np.isfinite(plan.evaluate(values)).all()
    print("Verified 96 source wires: 32 requested lane/readout pairs instead of 1024.")


if __name__ == "__main__":
    main()
