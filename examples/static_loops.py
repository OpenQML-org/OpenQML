"""Import a bounded QASM loop, keep its parameter trainable and export to Cirq."""

import numpy as np

from openqml import QuantumProgram, convert, statevector

source = """OPENQASM 3.0;
include "stdgates.inc";
input float w;
qubit[2] q;
for int i in [0:3] { ry(w * 0.2 * i) q[1]; cx q[1],q[0]; }
"""
program = QuantumProgram.from_qasm(source)
result = program.convert("cirq", bindings={"w": 0.3})
assert "STATIC_LOOP_UNROLLED" in {d.code for d in result.diagnostics}
restored = convert(result.value, source="cirq", target="ir").value
np.testing.assert_allclose(
    statevector(restored), statevector(program.to_ir({"w": 0.3})), atol=1e-12
)
print("Imported loop:", program.to_ir())
print("Cross-framework diagnostics:", result.diagnostics)
