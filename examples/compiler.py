"""Exact transforms, wire maps, and phase-sensitive verification; NumPy only."""

import numpy as np

from openqml import Circuit, Parameter, circuit_stats, decompose, simplify, statevector

theta = Parameter("theta")
subcircuit = Circuit(2).h(0).crx(theta, 0, 1).x(1).x(1).u(theta, 0.2, -0.4, 1)
circuit = Circuit(3).compose(subcircuit, qubits=(2, 0)).with_global_phase(0.17)
compiled = simplify(decompose(circuit, basis={"p", "ry", "cz"}))
print("Original:", circuit_stats(circuit))
print("Compiled:", circuit_stats(compiled))
np.testing.assert_allclose(
    statevector(circuit, {"theta": 0.43}), statevector(compiled, {"theta": 0.43}), atol=2e-12
)
identity = circuit.compose(circuit.inverse())
np.testing.assert_allclose(statevector(identity, {"theta": 0.43}), np.eye(8)[0], atol=2e-12)
print("Global-phase-sensitive equivalence and inversion verified.")
