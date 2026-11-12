"""Unitary multiscale circuits with adjustable retained spaces and disentanglers.

Discarding a wire means omitting it from later gates and final observables,
equivalent to a partial trace. It never means postselection. Unlike a product
of independent binary pooling nodes, boundary disentanglers correlate adjacent
coarse blocks; evaluation therefore uses a general circuit execution engine.
"""

from dataclasses import dataclass, field

from ..errors import ResourceError, ValidationError
from ..ir.core import CircuitBuilder, integer
from ..ir.observables import PauliString
from ..ir.parameters import Parameter
from .layers import LayerSpec


@dataclass(frozen=True)
class MultiscalePlan:
    num_qubits: int
    bond_qubits: int = 1
    depth: int = 2
    num_outputs: int = 3
    disentangle: bool = True
    reupload: bool = True
    max_operations: int = 100_000
    spec: LayerSpec = field(init=False, repr=False)
    retained_wires: tuple = field(init=False)
    scales: int = field(init=False)

    def __post_init__(self):
        n = integer(self.num_qubits, "num_qubits", 1)
        k = min(integer(self.bond_qubits, "bond_qubits", 1), n)
        integer(self.depth, "depth", 1)
        outputs = integer(self.num_outputs, "num_outputs", 1)
        integer(self.max_operations, "max_operations", 1)
        if type(self.disentangle) is not bool or type(self.reupload) is not bool:
            raise ValidationError("disentangle and reupload must be booleans.")
        if outputs.bit_length() > 2 * k:
            raise ValidationError("num_outputs exceeds the nonidentity retained Pauli basis.")
        if n > self.max_operations:
            raise ResourceError("Input encoding exceeds max_operations.")
        builder, weights = CircuitBuilder(n), []
        inputs = tuple(f"ms_x_{q}" for q in range(n))
        count = 0

        def emit(axis, qubits, angle):
            nonlocal count
            count += 1
            if count > self.max_operations:
                raise ResourceError("Multiscale construction exceeds max_operations.")
            builder.gate(axis, *qubits, params=(angle,))

        def trainable(axis, *qubits):
            name = f"ms_w_{len(weights)}"
            weights.append(name)
            emit(axis, qubits, Parameter(name))

        def entangle(a, b):
            for axis in ("rxx", "ryy", "rzz"):
                trainable(axis, a, b)

        for q, name in enumerate(inputs):
            emit("ry", (q,), Parameter(name))
        blocks = [tuple(range(q, min(q + k, n))) for q in range(0, n, k)]
        scale = 0
        while True:
            merged = [
                blocks[i] + (blocks[i + 1] if i + 1 < len(blocks) else ())
                for i in range(0, len(blocks), 2)
            ]
            for _ in range(self.depth):
                for block in blocks:
                    for q in block:
                        if self.reupload:
                            emit("ry", (q,), Parameter(inputs[q]))
                        for axis in ("rz", "ry", "rz"):
                            trainable(axis, q)
                if self.disentangle:
                    # Boundaries between distinct pooling regions, as in a
                    # binary MERA layer; these cannot be reduced node by node.
                    for i in range(1, len(blocks) - 1, 2):
                        entangle(blocks[i][-1], blocks[i + 1][0])
                for block in merged:
                    for offset in (0, 1):
                        for i in range(offset, len(block) - 1, 2):
                            entangle(block[i], block[i + 1])
            scale += 1
            if len(blocks) == 1:
                roots = blocks[0]
                break
            blocks = [block if len(block) <= k else block[::2][:k] for block in merged]
        observables = []
        for value in range(1, outputs + 1):
            word = ["I"] * n
            for q in roots:
                word[q] = "IXYZ"[value % 4]
                value //= 4
            observables.append(PauliString("".join(word)))
        object.__setattr__(self, "retained_wires", tuple(roots))
        object.__setattr__(self, "scales", scale)
        object.__setattr__(
            self, "spec", LayerSpec(builder.build(), inputs, tuple(weights), tuple(observables))
        )

    def to_spec(self):
        return self.spec

    def to_dict(self):
        return {
            "architecture": "unitary_multiscale_disentangling_circuit",
            "num_qubits": self.num_qubits,
            "retained_block_qubits": min(self.bond_qubits, self.num_qubits),
            "scales": self.scales,
            "depth_per_scale": self.depth,
            "disentanglers": self.disentangle,
            "data_reuploading": self.reupload,
            "retained_wires": self.retained_wires,
            "parameters": len(self.spec.weight_parameters),
            "operations": len(self.spec.circuit.operations),
            "readouts": self.num_outputs,
            "pooling": "partial_trace_of_unobserved_wires; no_postselection",
            "capacity": "execution_width_and_work_budgets; not_polynomial_for_all_circuits",
            "unpooled_limit": self.bond_qubits >= self.num_qubits,
        }
