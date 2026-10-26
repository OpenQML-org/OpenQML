"""Reusable Qiskit library program consumed by Torch and JAX/Flax models."""


def build_qiskit_program(qubits=4):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import real_amplitudes, zz_feature_map

    feature_map = zz_feature_map(qubits, reps=1)
    ansatz = real_amplitudes(qubits, reps=2)
    circuit = QuantumCircuit(qubits)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)
    return circuit, tuple(feature_map.parameters), tuple(ansatz.parameters)
