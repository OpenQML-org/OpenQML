"""A serialisable circuit description.

A :class:`Circuit` is data, not code: every gate angle is either a number, a
reference to a trainable weight (``w(3)``) or a reference to an input feature
(``x(0)``). That is what lets a model be uploaded, downloaded and re-executed
by somebody else -- the same role ``flows`` play in OpenML.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = ["Ref", "w", "x", "Gate", "Circuit", "GATE_ARITY"]

#: name -> number of wires it acts on (``0`` means "the whole register")
GATE_ARITY = {
    "h": 1, "x": 1, "y": 1, "z": 1, "s": 1, "t": 1,
    "rx": 1, "ry": 1, "rz": 1, "phase": 1,
    "cnot": 2, "cz": 2, "swap": 2, "crx": 2, "cry": 2, "crz": 2, "rzz": 2, "rxx": 2,
    "amplitude_embedding": 0,
}


@dataclass(frozen=True)
class Ref:
    """A symbolic angle: ``kind`` is ``"weights"`` or ``"x"``."""

    kind: str
    index: int
    scale: float = 1.0
    offset: float = 0.0

    def resolve(self, weights=None, features=None) -> float:
        source = weights if self.kind == "weights" else features
        if source is None:
            raise ValueError(f"circuit needs a value for {self.kind}[{self.index}]")
        source = np.asarray(source).reshape(-1)
        if self.index >= source.size:
            raise ValueError(
                f"{self.kind}[{self.index}] is out of range (size {source.size})"
            )
        return float(np.real(source[self.index])) * self.scale + self.offset

    def to_dict(self) -> Dict[str, Any]:
        return {"ref": self.kind, "index": self.index, "scale": self.scale, "offset": self.offset}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Ref":
        return cls(payload["ref"], int(payload["index"]),
                   float(payload.get("scale", 1.0)), float(payload.get("offset", 0.0)))

    def __repr__(self) -> str:
        prefix = "w" if self.kind == "weights" else "x"
        body = f"{prefix}{self.index}"
        if self.scale != 1.0:
            body = f"{self.scale:g}*{body}"
        if self.offset:
            body = f"{body}+{self.offset:g}"
        return body


def w(index: int, scale: float = 1.0, offset: float = 0.0) -> Ref:
    """Reference to trainable weight ``index``."""
    return Ref("weights", index, scale, offset)


def x(index: int, scale: float = 1.0, offset: float = 0.0) -> Ref:
    """Reference to input feature ``index``."""
    return Ref("x", index, scale, offset)


Angle = Union[float, int, Ref]


@dataclass
class Gate:
    name: str
    wires: Tuple[int, ...]
    params: Tuple[Angle, ...] = ()

    def resolved_params(self, weights=None, features=None) -> Tuple[float, ...]:
        out = []
        for param in self.params:
            out.append(param.resolve(weights, features) if isinstance(param, Ref) else float(param))
        return tuple(out)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "wires": list(self.wires),
            "params": [p.to_dict() if isinstance(p, Ref) else float(p) for p in self.params],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Gate":
        params = tuple(
            Ref.from_dict(p) if isinstance(p, dict) else float(p) for p in payload.get("params", [])
        )
        return cls(payload["name"], tuple(payload["wires"]), params)


class Circuit:
    """An ordered list of gates over ``n_qubits`` wires."""

    def __init__(self, n_qubits: int, name: str = "circuit", gates: Optional[Iterable[Gate]] = None):
        if n_qubits < 1:
            raise ValueError("a circuit needs at least one qubit")
        self.n_qubits = int(n_qubits)
        self.name = name
        self.gates: List[Gate] = list(gates or [])

    # -- construction -----------------------------------------------------
    def append(self, name: str, wires, params=()) -> "Circuit":
        wires = (wires,) if isinstance(wires, (int, np.integer)) else tuple(int(q) for q in wires)
        if name not in GATE_ARITY:
            raise ValueError(f"unknown gate {name!r}; known gates: {sorted(GATE_ARITY)}")
        arity = GATE_ARITY[name]
        if arity and len(wires) != arity:
            raise ValueError(f"{name} acts on {arity} wire(s), got {len(wires)}")
        for wire in wires:
            if not 0 <= wire < self.n_qubits:
                raise ValueError(f"wire {wire} out of range for a {self.n_qubits}-qubit circuit")
        params = (params,) if isinstance(params, (int, float, Ref)) else tuple(params)
        self.gates.append(Gate(name, wires, params))
        return self

    def h(self, q): return self.append("h", q)
    def x_(self, q): return self.append("x", q)
    def y_(self, q): return self.append("y", q)
    def z_(self, q): return self.append("z", q)
    def s(self, q): return self.append("s", q)
    def t(self, q): return self.append("t", q)
    def rx(self, theta, q): return self.append("rx", q, theta)
    def ry(self, theta, q): return self.append("ry", q, theta)
    def rz(self, theta, q): return self.append("rz", q, theta)
    def phase(self, theta, q): return self.append("phase", q, theta)
    def cnot(self, control, target): return self.append("cnot", (control, target))
    def cz(self, control, target): return self.append("cz", (control, target))
    def swap(self, a, b): return self.append("swap", (a, b))
    def crx(self, theta, control, target): return self.append("crx", (control, target), theta)
    def cry(self, theta, control, target): return self.append("cry", (control, target), theta)
    def crz(self, theta, control, target): return self.append("crz", (control, target), theta)
    def rzz(self, theta, a, b): return self.append("rzz", (a, b), theta)
    def rxx(self, theta, a, b): return self.append("rxx", (a, b), theta)

    def amplitude_embedding(self, wires=None) -> "Circuit":
        """Load the (normalised) feature vector directly into the amplitudes."""
        wires = tuple(range(self.n_qubits)) if wires is None else tuple(wires)
        self.gates.append(Gate("amplitude_embedding", wires, ()))
        return self

    def barrier(self) -> "Circuit":  # visual only, no-op during simulation
        return self

    # -- composition ------------------------------------------------------
    def compose(self, other: "Circuit", weight_offset: int = 0) -> "Circuit":
        """Append ``other``, optionally shifting its weight indices."""
        merged = Circuit(max(self.n_qubits, other.n_qubits), self.name, list(self.gates))
        for gate in other.gates:
            params = tuple(
                Ref(p.kind, p.index + (weight_offset if p.kind == "weights" else 0), p.scale, p.offset)
                if isinstance(p, Ref) else p
                for p in gate.params
            )
            merged.gates.append(Gate(gate.name, gate.wires, params))
        return merged

    def __add__(self, other: "Circuit") -> "Circuit":
        return self.compose(other)

    def __len__(self) -> int:
        return len(self.gates)

    # -- introspection ----------------------------------------------------
    def _max_ref(self, kind: str) -> int:
        indices = [p.index for g in self.gates for p in g.params
                   if isinstance(p, Ref) and p.kind == kind]
        return max(indices) + 1 if indices else 0

    @property
    def n_parameters(self) -> int:
        """Number of distinct trainable weights the circuit reads."""
        return self._max_ref("weights")

    @property
    def n_features(self) -> int:
        """Number of distinct input features the circuit reads."""
        if any(g.name == "amplitude_embedding" for g in self.gates):
            return 2 ** self.n_qubits
        return self._max_ref("x")

    @property
    def depth(self) -> int:
        """Circuit depth: longest chain of gates sharing a wire."""
        frontier = [0] * self.n_qubits
        for gate in self.gates:
            wires = gate.wires or tuple(range(self.n_qubits))
            level = max(frontier[q] for q in wires) + 1
            for q in wires:
                frontier[q] = level
        return max(frontier) if frontier else 0

    def parameter_indices_per_gate(self) -> List[Tuple[int, int]]:
        """(gate position, weight index) for every gate driven by one weight.

        The parameter-shift rule differentiates one *gate occurrence* at a time,
        so trainers need this rather than the bare parameter count.
        """
        out = []
        for position, gate in enumerate(self.gates):
            for param in gate.params:
                if isinstance(param, Ref) and param.kind == "weights":
                    out.append((position, param.index))
        return out

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "n_qubits": self.n_qubits,
            "n_parameters": self.n_parameters,
            "n_features": self.n_features,
            "gates": [g.to_dict() for g in self.gates],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Circuit":
        circuit = cls(payload["n_qubits"], payload.get("name", "circuit"))
        circuit.gates = [Gate.from_dict(g) for g in payload.get("gates", [])]
        return circuit

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "Circuit":
        return cls.from_dict(json.loads(text))

    def to_qasm(self) -> str:
        """Best-effort OpenQASM 2.0 export (only for fully numeric circuits)."""
        lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{self.n_qubits}];"]
        for gate in self.gates:
            if any(isinstance(p, Ref) for p in gate.params):
                raise ValueError("bind weights/features before exporting to QASM")
            if gate.name == "amplitude_embedding":
                raise ValueError("amplitude_embedding has no direct QASM equivalent")
            args = ",".join(f"q[{q}]" for q in gate.wires)
            if gate.params:
                params = ",".join(f"{float(p):.10g}" for p in gate.params)
                lines.append(f"{gate.name}({params}) {args};")
            else:
                lines.append(f"{gate.name} {args};")
        return "\n".join(lines)

    # -- drawing ----------------------------------------------------------
    def draw(self) -> str:
        """ASCII diagram, one column per gate."""
        rows = [[] for _ in range(self.n_qubits)]
        for gate in self.gates:
            labels = {}
            if gate.name == "amplitude_embedding":
                for q in range(self.n_qubits):
                    labels[q] = "|x>"
            elif len(gate.wires) == 1:
                text = gate.name.upper()
                if gate.params:
                    text += f"({gate.params[0]!r})" if isinstance(gate.params[0], Ref) \
                        else f"({float(gate.params[0]):.2f})"
                labels[gate.wires[0]] = text
            else:
                control, target = gate.wires[0], gate.wires[1]
                if gate.name in ("cnot", "cz", "crx", "cry", "crz"):
                    labels[control] = "*"
                    tail = gate.name[1:].upper()
                    if gate.params:
                        tail += f"({gate.params[0]!r})" if isinstance(gate.params[0], Ref) \
                            else f"({float(gate.params[0]):.2f})"
                    labels[target] = "X" if gate.name == "cnot" else tail
                else:
                    text = gate.name.upper()
                    if gate.params:
                        text += f"({gate.params[0]!r})" if isinstance(gate.params[0], Ref) \
                            else f"({float(gate.params[0]):.2f})"
                    labels[control] = labels[target] = text
            width = max(len(v) for v in labels.values())
            for q in range(self.n_qubits):
                cell = labels.get(q, "")
                rows[q].append(cell.center(width, " ") if cell else "-" * width)
        out = []
        for q, row in enumerate(rows):
            body = "--".join(row) if row else ""
            out.append(f"q{q}: --{body}--")
        return "\n".join(out)

    def __repr__(self) -> str:
        return (f"<Circuit {self.name!r} qubits={self.n_qubits} gates={len(self.gates)} "
                f"depth={self.depth} weights={self.n_parameters} features={self.n_features}>")
