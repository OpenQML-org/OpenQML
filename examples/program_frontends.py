"""Native QASM compilation, cross-ecosystem export and preserved dynamic execution."""

from qiskit import QuantumCircuit

from openqml import QuantumProgram, sample

source = """OPENQASM 3.0;
include "stdgates.inc";
gate block(t) a,b { ry(t) a; cx a,b; }
qubit[2] q;
block(0.43) q[0],q[1];
"""
program = QuantumProgram.from_qasm(source)
print("Converted native program:", program.convert("cirq").value)
q = QuantumCircuit(2, 2)
q.h(0)
q.measure(0, 0)
with q.if_test((q.clbits[0], 1)):
    q.x(1)
q.measure(1, 1)
dynamic = QuantumProgram.capture(q)
print("Native dynamic result:", dynamic.run(shots=64, seed=11).get_counts())
ir = dynamic.to_ir()
print("IR feedback result:", sample(ir, shots=64, seed=11))
qasm = dynamic.convert("openqasm3").value
restored = QuantumProgram.from_qasm(qasm).to_ir()
assert set(sample(restored, 256, seed=11)) <= {"00", "11"}
print("Feedback exported to OpenQASM 3:", qasm)
