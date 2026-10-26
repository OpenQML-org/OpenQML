"""Native MPS expectations and samples on a low-bond-dimension 64-qubit GHZ state."""

from openqml import Circuit, PauliString, execute, expectation

circuit = Circuit(64).h(0)
for q in range(1, 64):
    circuit = circuit.cx(q - 1, q)
value = expectation(circuit, PauliString("Z" + "I" * 62 + "Z"), backend="aer-mps")
result = execute(circuit, backend="aer-mps", shots=128, seed=17)
assert abs(value - 1) < 1e-10
assert set(result.counts) <= {"0" * 64, "1" * 64}
print("64-qubit GHZ endpoint correlation:", value)
print("Sample counts:", {f"{key[:8]}... ({len(key)} bits)": n for key, n in result.counts.items()})
assert result.statevector is None and result.probabilities is None
