"""Versioned, immutable quantum circuit IR. Wire zero is the most significant bit."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from numbers import Integral

import numpy as np

from .errors import ValidationError
from .parameters import Expression, expression

# gate -> (quantum arity, parameter count); barrier has variable positive arity.
GATES = {
    "id": (1, 0),
    "x": (1, 0),
    "y": (1, 0),
    "z": (1, 0),
    "h": (1, 0),
    "s": (1, 0),
    "sdg": (1, 0),
    "t": (1, 0),
    "tdg": (1, 0),
    "sx": (1, 0),
    "rx": (1, 1),
    "ry": (1, 1),
    "rz": (1, 1),
    "p": (1, 1),
    "cx": (2, 0),
    "cy": (2, 0),
    "cz": (2, 0),
    "swap": (2, 0),
    "ccx": (3, 0),
    "rxx": (2, 1),
    "ryy": (2, 1),
    "rzz": (2, 1),
    "u": (1, 3),
    "sxdg": (1, 0),
    "cp": (2, 1),
    "crx": (2, 1),
    "cry": (2, 1),
    "crz": (2, 1),
    "ch": (2, 0),
    "cswap": (3, 0),
    "iswap": (2, 0),
    "unitary": (None, 0),
    "barrier": (None, 0),
    "measure": (1, 0),
}
GATES_V3 = frozenset(GATES)
CHANNEL_GATES = frozenset(
    {"amplitude_damping", "phase_damping", "depolarizing", "bit_flip", "phase_flip", "kraus"}
)
UNITARY_GATES = GATES_V3 - {"measure", "barrier"}
GATES.update({name: (1, 1) for name in CHANNEL_GATES - {"kraus"}})
GATES.update({"kraus": (None, 0), "reset": (1, 0), "if_else": (0, 0)})
IR_VERSION = 4
GATES_V2 = GATES_V3 - {"unitary"}
GATES_V1 = GATES_V2 - {"u", "sxdg", "cp", "crx", "cry", "crz", "ch", "cswap", "iswap"}


def integer(value, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValidationError(f"{label} must be an integer >= {minimum}.")
    return int(value)


@dataclass(frozen=True)
class ClassicalCondition:
    """Equality against classical bits in least-significant-bit-first order."""

    bits: tuple[int, ...]
    value: int = 1

    def __post_init__(self):
        bits = tuple(integer(bit, "condition bit") for bit in self.bits)
        value = integer(self.value, "condition value")
        if not bits or len(set(bits)) != len(bits) or value.bit_length() > len(bits):
            raise ValidationError("A condition needs unique bits and a value fitting those bits.")
        object.__setattr__(self, "bits", bits)
        object.__setattr__(self, "value", value)

    def matches(self, ones):
        return all((bit in ones) == bool((self.value >> i) & 1) for i, bit in enumerate(self.bits))


@dataclass(frozen=True)
class Operation:
    name: str
    qubits: tuple[int, ...]
    params: tuple[Expression, ...] = ()
    clbits: tuple[int, ...] = ()
    matrix: tuple[tuple[complex, ...], ...] | None = None
    kraus: tuple | None = None
    condition: ClassicalCondition | None = None
    blocks: tuple[Circuit, ...] = ()

    def __post_init__(self):
        if self.name not in GATES:
            raise ValidationError(f"Unknown gate {self.name!r}.")
        qubits = tuple(integer(q, "qubit") for q in self.qubits)
        clbits = tuple(integer(c, "classical bit") for c in self.clbits)
        arity, nparams = GATES[self.name]
        if (not qubits and self.name != "if_else") or (arity is not None and len(qubits) != arity):
            raise ValidationError(f"{self.name} expects {arity or 'one or more'} qubits.")
        if len(set(qubits)) != len(qubits):
            raise ValidationError("An operation cannot use the same qubit twice.")
        if len(self.params) != nparams:
            raise ValidationError(f"{self.name} expects {nparams} parameters.")
        if len(clbits) != (1 if self.name == "measure" else 0):
            raise ValidationError("Only measurement has one classical target bit.")
        object.__setattr__(self, "qubits", qubits)
        object.__setattr__(self, "clbits", clbits)
        object.__setattr__(self, "params", tuple(expression(p) for p in self.params))
        if self.name == "unitary":
            value = np.asarray(self.matrix, dtype=complex)
            dimension = 2 ** len(qubits)
            if value.shape != (dimension, dimension) or not np.all(np.isfinite(value)):
                raise ValidationError("Unitary matrix dimensions or entries are invalid.")
            if not np.allclose(value.conj().T @ value, np.eye(dimension), atol=1e-10, rtol=0):
                raise ValidationError("The supplied matrix is not unitary.")
            object.__setattr__(
                self, "matrix", tuple(tuple(complex(x) for x in row) for row in value)
            )
        elif self.matrix is not None:
            raise ValidationError("Only unitary operations contain a matrix payload.")
        if self.name == "kraus":
            value = np.asarray(self.kraus, dtype=complex)
            dimension = 2 ** len(qubits)
            if (
                value.ndim != 3
                or value.shape[0] < 1
                or value.shape[1:] != (dimension, dimension)
                or not np.isfinite(value).all()
            ):
                raise ValidationError("Kraus operators need finite, matching square matrices.")
            completeness = (value.conj().transpose(0, 2, 1) @ value).sum(axis=0)
            if not np.allclose(completeness, np.eye(dimension), atol=1e-10, rtol=0):
                raise ValidationError(
                    "Kraus operators must define a trace-preserving quantum channel."
                )
            object.__setattr__(
                self,
                "kraus",
                tuple(tuple(tuple(complex(z) for z in row) for row in matrix) for matrix in value),
            )
        elif self.kraus is not None:
            raise ValidationError("Only kraus operations have Kraus payloads.")
        object.__setattr__(self, "blocks", tuple(self.blocks))
        if self.name == "if_else":
            if (
                not isinstance(self.condition, ClassicalCondition)
                or len(self.blocks) != 2
                or any(not isinstance(b, Circuit) for b in self.blocks)
            ):
                raise ValidationError("if_else needs a ClassicalCondition and two Circuit blocks.")
        elif self.condition is not None or self.blocks:
            raise ValidationError("Only if_else operations have conditions or blocks.")
        if self.name in CHANNEL_GATES - {"kraus"} and not self.params[0].parameters:
            probability = self.params[0].evaluate({})
            if not 0 <= probability <= 1:
                raise ValidationError("Channel probability must lie in [0, 1].")


@dataclass(frozen=True)
class Circuit:
    num_qubits: int
    operations: tuple[Operation, ...] = ()
    num_clbits: int = 0
    global_phase: Expression = Expression()

    def __post_init__(self):
        object.__setattr__(self, "num_qubits", integer(self.num_qubits, "num_qubits", 1))
        object.__setattr__(self, "num_clbits", integer(self.num_clbits, "num_clbits"))
        object.__setattr__(self, "operations", tuple(self.operations))
        object.__setattr__(self, "global_phase", expression(self.global_phase))
        for op in self.operations:
            if not isinstance(op, Operation):
                raise ValidationError("Circuit operations must be Operation objects.")
            if max(op.qubits, default=-1) >= self.num_qubits:
                raise ValidationError("Operation references an undeclared qubit.")
            if op.clbits and max(op.clbits) >= self.num_clbits:
                raise ValidationError("Operation references an undeclared classical bit.")
            if op.condition and max(op.condition.bits) >= self.num_clbits:
                raise ValidationError("Condition references an undeclared classical bit.")
            if any(
                b.num_qubits != self.num_qubits or b.num_clbits > self.num_clbits for b in op.blocks
            ):
                raise ValidationError(
                    "Blocks must share global quantum indices and fit classical storage."
                )

    @property
    def parameters(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                set(self.global_phase.parameters).union(
                    *(p.parameters for _, op in self.walk_operations() for p in op.params),
                    *(
                        b.global_phase.parameters
                        for _, op in self.walk_operations()
                        for b in op.blocks
                    ),
                )
            )
        )

    @property
    def has_measurements(self) -> bool:
        return any(op.name == "measure" for _, op in self.walk_operations())

    @property
    def has_effects(self):
        return any(op.name not in UNITARY_GATES | {"barrier"} for op in self.operations)

    def walk_operations(self, prefix=()):
        for index, op in enumerate(self.operations):
            path = prefix + (index,)
            yield path, op
            for block_index, block in enumerate(op.blocks):
                yield from block.walk_operations(path + (block_index,))

    def operation_at(self, path):
        op = self.operations[path[0]]
        return op if len(path) == 1 else op.blocks[path[1]].operation_at(path[2:])

    def replace_operation(self, path, operation):
        operations = list(self.operations)
        if len(path) == 1:
            operations[path[0]] = operation
        else:
            parent = operations[path[0]]
            blocks = list(parent.blocks)
            blocks[path[1]] = blocks[path[1]].replace_operation(path[2:], operation)
            operations[path[0]] = replace(parent, blocks=tuple(blocks))
        return replace(self, operations=tuple(operations))

    def append(self, operation: Operation) -> Circuit:
        return replace(self, operations=self.operations + (operation,))

    def gate(self, name: str, *qubits: int, params=()) -> Circuit:
        return self.append(Operation(name, tuple(qubits), tuple(params)))

    def id(self, q):
        return self.gate("id", q)

    def x(self, q):
        return self.gate("x", q)

    def y(self, q):
        return self.gate("y", q)

    def z(self, q):
        return self.gate("z", q)

    def h(self, q):
        return self.gate("h", q)

    def s(self, q):
        return self.gate("s", q)

    def sdg(self, q):
        return self.gate("sdg", q)

    def t(self, q):
        return self.gate("t", q)

    def tdg(self, q):
        return self.gate("tdg", q)

    def sx(self, q):
        return self.gate("sx", q)

    def rx(self, angle, q):
        return self.gate("rx", q, params=(angle,))

    def ry(self, angle, q):
        return self.gate("ry", q, params=(angle,))

    def rz(self, angle, q):
        return self.gate("rz", q, params=(angle,))

    def p(self, angle, q):
        return self.gate("p", q, params=(angle,))

    def cx(self, control, target):
        return self.gate("cx", control, target)

    def cy(self, control, target):
        return self.gate("cy", control, target)

    def cz(self, control, target):
        return self.gate("cz", control, target)

    def swap(self, a, b):
        return self.gate("swap", a, b)

    def ccx(self, a, b, target):
        return self.gate("ccx", a, b, target)

    def rxx(self, angle, a, b):
        return self.gate("rxx", a, b, params=(angle,))

    def ryy(self, angle, a, b):
        return self.gate("ryy", a, b, params=(angle,))

    def rzz(self, angle, a, b):
        return self.gate("rzz", a, b, params=(angle,))

    def u(self, theta, phi, lam, q):
        return self.gate("u", q, params=(theta, phi, lam))

    def sxdg(self, q):
        return self.gate("sxdg", q)

    def cp(self, angle, control, target):
        return self.gate("cp", control, target, params=(angle,))

    def crx(self, angle, control, target):
        return self.gate("crx", control, target, params=(angle,))

    def cry(self, angle, control, target):
        return self.gate("cry", control, target, params=(angle,))

    def crz(self, angle, control, target):
        return self.gate("crz", control, target, params=(angle,))

    def ch(self, control, target):
        return self.gate("ch", control, target)

    def cswap(self, control, a, b):
        return self.gate("cswap", control, a, b)

    def iswap(self, a, b):
        return self.gate("iswap", a, b)

    def unitary(self, matrix, *qubits):
        return self.append(Operation("unitary", tuple(qubits), matrix=matrix))

    def reset(self, q):
        return self.gate("reset", q)

    def amplitude_damping(self, probability, q):
        return self.gate("amplitude_damping", q, params=(probability,))

    def phase_damping(self, probability, q):
        return self.gate("phase_damping", q, params=(probability,))

    def depolarizing(self, probability, q):
        return self.gate("depolarizing", q, params=(probability,))

    def bit_flip(self, probability, q):
        return self.gate("bit_flip", q, params=(probability,))

    def phase_flip(self, probability, q):
        return self.gate("phase_flip", q, params=(probability,))

    def channel(self, kraus, *qubits):
        return self.append(Operation("kraus", tuple(qubits), kraus=kraus))

    def if_else(self, condition, when_true, when_false=None):
        if not isinstance(condition, ClassicalCondition):
            raise ValidationError("Use ClassicalCondition(bits, value) for if_else.")
        when_false = Circuit(self.num_qubits) if when_false is None else when_false
        if not isinstance(when_true, Circuit) or not isinstance(when_false, Circuit):
            raise ValidationError("Branch bodies must be Circuit instances.")
        width = max(
            self.num_clbits, max(condition.bits) + 1, when_true.num_clbits, when_false.num_clbits
        )
        return replace(self, num_clbits=width).append(
            Operation("if_else", (), condition=condition, blocks=(when_true, when_false))
        )

    def compose(self, other: Circuit, *, qubits=None, clbits=None) -> Circuit:
        """Append a circuit using explicit wire maps; equal symbol names share a parameter.

        Default maps reuse q0/c0 onwards. Classical storage grows if necessary.
        Source registers, including idle bits, are mapped in full.
        """
        if not isinstance(other, Circuit):
            raise ValidationError("compose expects a Circuit.")
        quantum = tuple(range(other.num_qubits) if qubits is None else qubits)
        classical = tuple(range(other.num_clbits) if clbits is None else clbits)
        for label, mapping, width in (
            ("qubits", quantum, other.num_qubits),
            ("clbits", classical, other.num_clbits),
        ):
            if len(mapping) != width:
                raise ValidationError(f"{label} map must cover every source bit.")
            for bit in mapping:
                integer(bit, f"{label} index")
            if len(set(mapping)) != len(mapping):
                raise ValidationError(f"{label} map must be injective.")
        if max(quantum) >= self.num_qubits:
            raise ValidationError("Quantum map exceeds the destination width.")
        width = max(self.num_clbits, max(classical, default=-1) + 1)

        def remap(op):
            condition = (
                None
                if op.condition is None
                else ClassicalCondition(
                    tuple(classical[c] for c in op.condition.bits), op.condition.value
                )
            )
            blocks = tuple(
                Circuit(self.num_qubits, num_clbits=width).compose(
                    b, qubits=quantum, clbits=classical[: b.num_clbits]
                )
                for b in op.blocks
            )
            return replace(
                op,
                qubits=tuple(quantum[q] for q in op.qubits),
                clbits=tuple(classical[c] for c in op.clbits),
                condition=condition,
                blocks=blocks,
            )

        mapped = tuple(remap(op) for op in other.operations)
        return replace(
            self,
            operations=self.operations + mapped,
            num_clbits=max(self.num_clbits, max(classical, default=-1) + 1),
            global_phase=self.global_phase + other.global_phase,
        )

    def inverse(self) -> Circuit:
        """Return an exact inverse; measurement has no unitary inverse."""
        from .compiler import inverse

        return inverse(self)

    def barrier(self, *qubits) -> Circuit:
        return self.gate("barrier", *(qubits or tuple(range(self.num_qubits))))

    def measure(self, qubit: int, cbit: int | None = None) -> Circuit:
        cbit = integer(qubit if cbit is None else cbit, "classical bit")
        return replace(self, num_clbits=max(self.num_clbits, cbit + 1)).append(
            Operation("measure", (qubit,), clbits=(cbit,))
        )

    def measure_all(self) -> Circuit:
        circuit = self
        for q in range(self.num_qubits):
            circuit = circuit.measure(q, q)
        return circuit

    def with_global_phase(self, phase) -> Circuit:
        return replace(self, global_phase=expression(phase))

    def bind(self, bindings: Mapping[str, float], *, partial: bool = False) -> Circuit:
        unknown = set(bindings) - set(self.parameters)
        missing = set(self.parameters) - set(bindings)
        if unknown or (missing and not partial):
            raise ValidationError(
                f"Parameter mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}."
            )
        operations = tuple(
            replace(
                op,
                params=tuple(p.bind(bindings) for p in op.params),
                blocks=tuple(
                    block.bind(
                        {k: v for k, v in bindings.items() if k in block.parameters},
                        partial=partial,
                    )
                    for block in op.blocks
                ),
            )
            for op in self.operations
        )
        return replace(self, operations=operations, global_phase=self.global_phase.bind(bindings))

    def unitary_part(self) -> Circuit:
        """Remove terminal measurements; reject any subsequent quantum operation."""
        if any(op.name not in UNITARY_GATES | {"measure", "barrier"} for op in self.operations):
            raise ValidationError(
                "Nonunitary effects have no pure unitary output. Use density_matrix()."
            )
        seen = False
        kept = []
        for op in self.operations:
            if op.name == "measure":
                seen = True
            elif op.name != "barrier":
                if seen:
                    raise ValidationError("Mid-circuit measurement is not supported by execution.")
                kept.append(op)
            else:
                kept.append(op)
        return replace(self, operations=tuple(kept), num_clbits=0)

    def to_dict(self) -> dict:
        def encode(p):
            return p.to_dict()

        return {
            "schema": "openqml.ir",
            "version": IR_VERSION,
            "bit_order": "q0-msb",
            "num_qubits": self.num_qubits,
            "num_clbits": self.num_clbits,
            "global_phase": encode(self.global_phase),
            "operations": [
                {
                    "name": op.name,
                    "qubits": list(op.qubits),
                    "clbits": list(op.clbits),
                    "params": [encode(p) for p in op.params],
                    "matrix": None
                    if op.matrix is None
                    else [[[z.real, z.imag] for z in row] for row in op.matrix],
                    "kraus": None
                    if op.kraus is None
                    else [
                        [[[z.real, z.imag] for z in row] for row in matrix] for matrix in op.kraus
                    ],
                    "condition": None
                    if op.condition is None
                    else {"bits": list(op.condition.bits), "value": op.condition.value},
                    "blocks": [block.to_dict() for block in op.blocks],
                }
                for op in self.operations
            ],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    @classmethod
    def from_dict(cls, payload: dict) -> Circuit:
        try:
            required = {
                "schema",
                "version",
                "bit_order",
                "num_qubits",
                "num_clbits",
                "global_phase",
                "operations",
            }
            if not isinstance(payload, dict) or set(payload) != required:
                raise ValidationError("Invalid IR fields.")
            if (
                payload["schema"] != "openqml.ir"
                or type(payload["version"]) is not int
                or payload["version"] not in {1, 2, 3, IR_VERSION}
                or payload["bit_order"] != "q0-msb"
            ):
                raise ValidationError("Unsupported IR schema, version, or bit order.")

            def decode(p):
                value = Expression.from_dict(p)
                if payload["version"] < 3 and not value.is_affine:
                    raise ValidationError("Nonlinear expressions require IR version 3.")
                return value

            ops = []
            if not isinstance(payload["operations"], list):
                raise ValidationError("Operations must be a JSON array.")
            for op in payload["operations"]:
                fields = {"name", "qubits", "clbits", "params"}
                if payload["version"] >= 3:
                    fields.add("matrix")
                if payload["version"] == 4:
                    fields.update({"kraus", "condition", "blocks"})
                if not isinstance(op, dict) or set(op) != fields:
                    raise ValidationError("Invalid operation fields.")
                if payload["version"] == 1 and op["name"] not in GATES_V1:
                    raise ValidationError("This operation requires IR version 2.")
                if payload["version"] < 4 and op["name"] not in GATES_V3:
                    raise ValidationError(
                        "Quantum effects and control blocks require IR version 4."
                    )
                if payload["version"] < 3 and op["name"] == "unitary":
                    raise ValidationError("Matrix payloads require IR version 3.")
                if any(not isinstance(op[key], list) for key in ("qubits", "clbits", "params")):
                    raise ValidationError("Operation operands and parameters must be JSON arrays.")
                condition = op.get("condition")
                if condition is not None and (
                    not isinstance(condition, dict)
                    or set(condition) != {"bits", "value"}
                    or not isinstance(condition["bits"], list)
                ):
                    raise ValidationError("Invalid condition fields.")
                if "blocks" in op and not isinstance(op["blocks"], list):
                    raise ValidationError("Blocks must be a JSON array.")
                ops.append(
                    Operation(
                        op["name"],
                        tuple(op["qubits"]),
                        tuple(decode(p) for p in op["params"]),
                        tuple(op["clbits"]),
                        None
                        if op.get("matrix") is None
                        else tuple(tuple(complex(*z) for z in row) for row in op["matrix"]),
                        None
                        if op.get("kraus") is None
                        else tuple(
                            tuple(tuple(complex(*z) for z in row) for row in matrix)
                            for matrix in op["kraus"]
                        ),
                        None
                        if condition is None
                        else ClassicalCondition(tuple(condition["bits"]), condition["value"]),
                        tuple(cls.from_dict(block) for block in op.get("blocks", [])),
                    )
                )
            return cls(
                payload["num_qubits"],
                tuple(ops),
                payload["num_clbits"],
                decode(payload["global_phase"]),
            )
        except ValidationError:
            raise
        except (KeyError, TypeError, ValueError, AttributeError, RecursionError) as exc:
            raise ValidationError("Malformed OpenQML IR.") from exc

    @classmethod
    def from_json(cls, text: str) -> Circuit:
        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValidationError(f"Duplicate JSON field {key!r}.")
                result[key] = value
            return result

        try:
            return cls.from_dict(json.loads(text, object_pairs_hook=unique_object))
        except (json.JSONDecodeError, RecursionError) as exc:
            raise ValidationError("Invalid IR JSON.") from exc
