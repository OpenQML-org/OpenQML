"""Feature maps and ansatz builders."""

from collections.abc import Iterable, Set
from dataclasses import dataclass
from itertools import combinations, pairwise

from ..errors import ValidationError, _as_tuple
from ..qml.layers import LayerSpec
from .core import CircuitBuilder, integer
from .parameters import Parameter


def _entangler_order(width, wire_order):
    if wire_order is None:
        return range(width)
    if isinstance(wire_order, Set):
        raise ValidationError("wire_order must list circuit wires in CX traversal order.")
    items = _as_tuple(wire_order, "wire_order", max_items=width)
    order = tuple(integer(q, "wire_order entry") for q in items)
    if len(order) != width or len(set(order)) != width or max(order) >= width:
        raise ValidationError("wire_order must be a permutation of the circuit wires.")
    return order


def _entangle(circuit, order, topology):
    if topology == "none":
        return
    edges = combinations(order, 2) if topology == "full" else pairwise(order)
    for control, target in edges:
        circuit.cx(control, target)
    if topology == "ring" and len(order) > 2:
        circuit.cx(order[-1], order[0])


def data_reuploading(
    num_qubits: int,
    depth: int = 2,
    *,
    entanglement: str = "linear",
    wire_order: Iterable[int] | None = None,
) -> LayerSpec:
    """Repeated RY encoding with trainable RX/RY/RZ blocks and CX entanglers.

    The same input symbols are reused at every depth; each variational block
    owns independent weights. Entanglement may be linear, ring, full or none.
    The optional wire permutation sets CX order and direction. Inputs, weights
    and readouts retain logical wire order.
    """
    integer(depth, "depth", 1)
    circuit = CircuitBuilder(num_qubits)
    if not isinstance(entanglement, str) or entanglement not in {"linear", "ring", "full", "none"}:
        raise ValidationError("entanglement must be linear, ring, full or none.")
    order = _entangler_order(circuit.num_qubits, wire_order)
    inputs = tuple(f"x_{q}" for q in range(num_qubits))
    weights = []
    for layer in range(depth):
        for q, name in enumerate(inputs):
            circuit = circuit.ry(Parameter(name), q)
            for axis in ("rx", "ry", "rz"):
                weight = f"w_{layer}_{q}_{axis}"
                weights.append(weight)
                circuit = circuit.gate(axis, q, params=(Parameter(weight),))
        _entangle(circuit, order, entanglement)
    return LayerSpec(circuit.build(), inputs, tuple(weights))


def hardware_efficient(
    num_qubits: int, depth: int = 2, *, wire_order: Iterable[int] | None = None
) -> LayerSpec:
    """RY input encoding, trainable RY/RZ blocks and a CX chain in the given wire order."""
    integer(depth, "depth", 1)
    circuit = CircuitBuilder(num_qubits)
    order = _entangler_order(circuit.num_qubits, wire_order)
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
        _entangle(circuit, order, "linear")
    return LayerSpec(circuit.build(), inputs, tuple(weights))


@dataclass(frozen=True)
class IdentityBlockInitialization:
    spec: LayerSpec
    pairs: tuple


def identity_block_ansatz(num_qubits, depth=2, *, max_operations=100_000):
    """Independent forward/inverse parameters; RZZ entanglers remain trainable.

    Pairing reverse parameters with negative forward parameters initializes
    each variational block to identity without permanently canceling its
    entangling family. This is an initialization, not a no-barren-plateau theorem.
    """
    n = integer(num_qubits, "num_qubits", 1)
    integer(depth, "depth", 1)
    integer(max_operations, "max_operations", 1)
    from ..errors import ResourceError

    if n + 2 * depth * (3 * n - 1) > max_operations:
        raise ResourceError("Identity-block construction exceeds max_operations.")
    builder = CircuitBuilder(n)
    inputs = tuple(f"x_{q}" for q in range(n))
    for q, name in enumerate(inputs):
        builder.ry(Parameter(name), q)
    gates = tuple((axis, (q,)) for q in range(n) for axis in ("ry", "rz"))
    gates += tuple(("rzz", (q, q + 1)) for q in range(n - 1))
    weights, pairs = [], []
    for block in range(depth):
        forward = []
        for i, (gate, qubits) in enumerate(gates):
            name = f"w_{block}_f_{i}"
            forward.append(len(weights))
            weights.append(name)
            builder.gate(gate, *qubits, params=(Parameter(name),))
        for i in reversed(range(len(gates))):
            name = f"w_{block}_r_{i}"
            pairs.append((forward[i], len(weights)))
            weights.append(name)
            gate, qubits = gates[i]
            builder.gate(gate, *qubits, params=(Parameter(name),))
    return IdentityBlockInitialization(
        LayerSpec(builder.build(), inputs, tuple(weights)), tuple(pairs)
    )
