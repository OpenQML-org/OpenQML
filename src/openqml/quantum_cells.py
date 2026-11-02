"""Shared convolution/pooling circuits and full-width trainable feature circuits."""

from dataclasses import dataclass, field

from .errors import ResourceError, ValidationError
from .ir import CircuitBuilder, integer
from .layers import LayerSpec
from .observables import PauliString
from .parameters import Parameter


@dataclass(frozen=True)
class CustomCircuitPlan:
    spec: LayerSpec

    def to_spec(self):
        return self.spec

    def to_dict(self):
        return {
            "architecture": "portable_custom_quantum_cell",
            "num_qubits": self.spec.circuit.num_qubits,
            "parameters": len(self.spec.weight_parameters),
            "operations": len(self.spec.circuit.operations),
            "readouts": len(self.spec.observables),
        }


@dataclass(frozen=True)
class FeatureCircuitPlan:
    num_qubits: int
    depth: int = 1
    num_outputs: int = 3
    reupload: bool = True
    max_operations: int = 100_000
    pooling: bool = False
    retained_qubits: int = 1
    spec: LayerSpec = field(init=False, repr=False)
    retained_wires: tuple = field(init=False)

    def __post_init__(self):
        n = integer(self.num_qubits, "num_qubits", 1)
        depth = integer(self.depth, "depth", 1)
        outputs = integer(self.num_outputs, "num_outputs", 1)
        limit = integer(self.max_operations, "max_operations", 1)
        retained = min(integer(self.retained_qubits, "retained_qubits", 1), n)
        if type(self.pooling) is not bool or type(self.reupload) is not bool:
            raise ValidationError("pooling and reupload must be booleans.")
        if (self.pooling and outputs.bit_length() > 2 * retained) or (
            not self.pooling and outputs > 3 * n
        ):
            raise ValidationError("Requested readouts exceed the retained Pauli basis.")
        if n > limit:
            raise ResourceError("Feature encoding exceeds max_operations.")
        b, weights, count = CircuitBuilder(n), [], 0
        inputs = tuple(f"feature_{q}" for q in range(n))

        def emit(gate, wires, angle):
            nonlocal count
            count += 1
            if count > limit:
                raise ResourceError("Feature circuit exceeds max_operations.")
            b.gate(gate, *wires, params=(angle,))

        def angles(size):
            names = tuple(f"cell_{len(weights) + i}" for i in range(size))
            weights.extend(names)
            return tuple(Parameter(name) for name in names)

        def encode(wires):
            for q in wires:
                emit("ry", (q,), Parameter(inputs[q]))

        def su4(a, c, w):
            # A local Cartan block: arbitrary SU(2) on each side of XX/YY/ZZ.
            for offset, q in ((0, a), (3, c)):
                for j, axis in enumerate(("rz", "ry", "rz")):
                    emit(axis, (q,), w[offset + j])
            for j, axis in enumerate(("rxx", "ryy", "rzz")):
                emit(axis, (a, c), w[6 + j])
            for offset, q in ((9, a), (12, c)):
                for j, axis in enumerate(("rz", "ry", "rz")):
                    emit(axis, (q,), w[offset + j])

        active = list(range(n))
        encode(active)
        while True:
            for layer in range(depth):
                if self.reupload and (self.pooling or layer):
                    encode(active)
                if self.pooling:
                    if len(active) == 1:
                        for axis, angle in zip(("rz", "ry", "rz"), angles(3), strict=True):
                            emit(axis, (active[0],), angle)
                    for parity in (0, 1):
                        pairs = list(zip(active[parity::2], active[parity + 1 :: 2], strict=False))
                        if pairs:
                            shared = angles(15)
                            for a, c in pairs:
                                su4(a, c, shared)
                else:
                    for q in active:
                        for axis, angle in zip(("rx", "ry", "rz"), angles(3), strict=True):
                            emit(axis, (q,), angle)
                    for a, c in zip(active, active[1:] + active[:1], strict=True):
                        if a != c:
                            emit("rzz", (a, c), angles(1)[0])
            if not self.pooling or len(active) <= retained:
                break
            shared = angles(3)
            removed = min(len(active) // 2, len(active) - retained)
            for a, c in zip(active[: 2 * removed : 2], active[1 : 2 * removed : 2], strict=True):
                emit("ry", (a,), shared[0])
                emit("cry", (a, c), shared[1])
                emit("crz", (a, c), shared[2])
            active = active[1 : 2 * removed : 2] + active[2 * removed :]
        readouts = []
        for i in range(outputs):
            word = ["I"] * n
            if self.pooling:
                value = i + 1
                for q in active:
                    word[q] = "IXYZ"[value % 4]
                    value //= 4
            else:
                word[active[i % n]] = "XYZ"[i // n]
            readouts.append(PauliString("".join(word)))
        object.__setattr__(self, "retained_wires", tuple(active))
        object.__setattr__(
            self, "spec", LayerSpec(b.build(), inputs, tuple(weights), tuple(readouts))
        )

    def to_spec(self):
        return self.spec

    def to_dict(self):
        return {
            "architecture": "shared_su4_qcnn" if self.pooling else "full_width_quantum_features",
            "num_qubits": self.num_qubits,
            "depth": self.depth,
            "parameters": len(self.spec.weight_parameters),
            "operations": len(self.spec.circuit.operations),
            "readouts": len(self.spec.observables),
            "retained_wires": self.retained_wires,
            "pooling": "partial_trace; no_postselection" if self.pooling else "none",
        }
