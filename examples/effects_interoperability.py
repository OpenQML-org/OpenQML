"""Channel roundtrips, exact ensembles and atomic feedback conversion."""

import numpy as np

from openqml import Circuit, ClassicalCondition, QuantumProgram, convert, density_matrix, sample

circuit = Circuit(3).h(0).cx(0, 2).amplitude_damping(0.23, 2).depolarizing(0.17, 0).reset(1)
expected = density_matrix(circuit)
for framework in ("qiskit", "pennylane", "cirq"):
    native = convert(circuit, target=framework).value
    restored = convert(native, source=framework, target="ir").value
    np.testing.assert_allclose(density_matrix(restored), expected, atol=1e-12)
    np.testing.assert_allclose(density_matrix(restored, backend=framework), expected, atol=1e-12)
    print(framework, "channel/reset roundtrip passed")

feedback = (
    Circuit(2)
    .h(0)
    .measure(0, 0)
    .if_else(
        ClassicalCondition((0,), 1),
        Circuit(2).x(0).measure(0, 0).x(1),
    )
    .measure(1, 1)
)
qasm = convert(feedback, target="openqasm3").value
restored = QuantumProgram.from_qasm(qasm).to_ir()
np.testing.assert_allclose(density_matrix(restored), density_matrix(feedback))
counts = sample(restored, shots=1000, seed=3)
assert set(counts) == {"00", "01"}
print("Atomic feedback, including the condition-bit overwrite:", counts)
