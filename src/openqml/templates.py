"""Feature maps and ansatz builders."""

from .errors import ValidationError
from .ir import CircuitBuilder, integer
from .layers import LayerSpec
from .parameters import Parameter


def data_reuploading(num_qubits: int, depth: int = 2, *, entanglement: str = "linear") -> LayerSpec:
    """Repeated RY encoding with trainable RX/RY/RZ blocks and CX entanglers.

    The same input symbols are reused at every depth; each variational block
    owns independent weights. Entanglement may be linear, ring, full or none.
    """
    integer(depth, "depth", 1)
    circuit = CircuitBuilder(num_qubits)
    if not isinstance(entanglement, str) or entanglement not in {"linear", "ring", "full", "none"}:
        raise ValidationError("entanglement must be linear, ring, full or none.")
    inputs = tuple(f"x_{q}" for q in range(num_qubits))
    weights = []
    edges = []
    if entanglement in {"linear", "ring"}:
        edges = [(q, q + 1) for q in range(num_qubits - 1)]
        if entanglement == "ring" and num_qubits > 2:
            edges.append((num_qubits - 1, 0))
    elif entanglement == "full":
        edges = [(a, b) for a in range(num_qubits) for b in range(a + 1, num_qubits)]
    for layer in range(depth):
        for q, name in enumerate(inputs):
            circuit = circuit.ry(Parameter(name), q)
            for axis in ("rx", "ry", "rz"):
                weight = f"w_{layer}_{q}_{axis}"
                weights.append(weight)
                circuit = circuit.gate(axis, q, params=(Parameter(weight),))
        for a, b in edges:
            circuit = circuit.cx(a, b)
    return LayerSpec(circuit.build(), inputs, tuple(weights))


def hardware_efficient(num_qubits: int, depth: int = 2) -> LayerSpec:
    """RY input encoding, trainable RY/RZ blocks, and a nearest-neighbour CX chain."""
    integer(depth, "depth", 1)
    circuit = CircuitBuilder(num_qubits)
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
    return LayerSpec(circuit.build(), inputs, tuple(weights))
