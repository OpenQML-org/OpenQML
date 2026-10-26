"""IR -> Qiskit -> Cirq -> OpenQASM 3 -> IR; requires both framework extras."""

import numpy as np

from openqml import Circuit, ConversionError, Parameter, convert, statevector

circuit = Circuit(3).h(0).ry(2 * Parameter("theta") + 0.1, 2).cx(2, 1)
qiskit = convert(circuit, source="ir", target="qiskit").value
cirq = convert(qiskit, source="qiskit", target="cirq").value
qasm = convert(cirq, source="cirq", target="openqasm3").value
restored = convert(qasm, source="openqasm3", target="ir").value
np.testing.assert_allclose(
    statevector(restored, {"theta": 0.3}), statevector(circuit, {"theta": 0.3}), atol=1e-12
)
print(qasm)
try:
    convert(circuit, source="ir", target="openqasm2")
except ConversionError as error:
    print("Expected diagnostic:", error)
