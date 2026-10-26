"""Variational ground-state optimization with occurrence-wise gradients; NumPy only."""

import numpy as np

from openqml import Circuit, Hamiltonian, Parameter, PauliString, expectation, parameter_shift

hamiltonian = Hamiltonian((PauliString("ZZ", -1), PauliString("XI", -1), PauliString("IX", -1)))
ansatz = Circuit(2).ry(Parameter("a"), 0).ry(Parameter("b"), 1).cx(0, 1)
weights = {"a": 0.2, "b": 0.1}
for step in range(100):
    gradient = parameter_shift(ansatz, hamiltonian, weights)
    weights = {name: value - 0.2 * gradient[name] for name, value in weights.items()}
    if step % 25 == 0 or step == 99:
        print(f"step={step:02d} energy={float(expectation(ansatz, hamiltonian, weights)):.8f}")
energy = float(expectation(ansatz, hamiltonian, weights))
exact_ground_energy = -np.sqrt(5)
assert abs(energy - exact_ground_energy) < 1e-5
print("Exact ground energy:", exact_ground_energy)
