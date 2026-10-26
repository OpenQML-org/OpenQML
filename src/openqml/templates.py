"""Feature maps and ansatz builders."""

from .ir import Circuit, integer
from .layers import LayerSpec
from .parameters import Parameter


def hardware_efficient(num_qubits: int, depth: int = 2) -> LayerSpec:
    """RY input encoding, trainable RY/RZ blocks, and a nearest-neighbour CX chain."""
    integer(depth, "depth", 1)
    circuit = Circuit(num_qubits)
    inputs = tuple(f"x_{q}" for q in range(num_qubits))
    weights = []
    for q, name in enumerate(inputs):
        circuit = circuit.ry(Parameter(name), q)
    for layer in range(depth):
        for q in range(num_qubits):
            for axis in ("ry", "rz"):
                name = f"w_{layer}_{q}_{axis}"
                weights.append(name)
                circuit = circuit.gate(axis, q, params=(Parameter(name),))
        for q in range(num_qubits - 1):
            circuit = circuit.cx(q, q + 1)
    return LayerSpec(circuit, inputs, tuple(weights))
