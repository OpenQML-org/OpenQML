"""Strict OpenQASM 2 subset and an optional Qiskit circuit adapter.

Unknown constructs raise an error. Source text is never executed with eval/exec.
"""

import ast
import re

import numpy as np

from ._validation import real
from .circuit import Circuit
from .exceptions import UnsupportedOperationError


def _angle(source):
    if len(source) > 128:
        raise ValueError("angle expression is too long")
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as error:
        raise ValueError("invalid QASM angle expression") from error
    if sum(1 for _ in ast.walk(tree)) > 32:
        raise ValueError("angle expression is too complex")

    def visit(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return real(node.value, "angle")
        if isinstance(node, ast.Name) and node.id == "pi":
            return np.pi
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
        ):
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return real(a + b, "angle")
            if isinstance(node.op, ast.Sub):
                return real(a - b, "angle")
            if isinstance(node.op, ast.Mult):
                return real(a * b, "angle")
            if b == 0:
                raise ValueError("division by zero in angle expression")
            return real(a / b, "angle")
        raise UnsupportedOperationError("angles support numbers, pi, +, -, *, /, and parentheses")

    return visit(tree.body)


def to_qasm(circuit):
    """OpenQASM 2 export. Reject noise, feedback, custom gates, and global phase."""
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{circuit.n_qubits}];"]
    measurements = [op for op in circuit._operations if op.name == "measure"]
    if measurements:
        lines.append(f"creg c[{len(measurements)}];")
    index = 0
    for op in circuit._operations:
        if op.condition is not None:
            raise UnsupportedOperationError("QASM 2 subset cannot export classical feedback")
        if op.name == "measure" and op.basis == "Z":
            lines.append(f"measure q[{op.wires[0]}] -> c[{index}];")
            index += 1
            continue
        name = op.name
        if op.controls:
            if len(op.controls) == 1 and name in ("x", "y", "z", "p"):
                name = "c" + name
            elif len(op.controls) == 2 and name == "x":
                name = "ccx"
            else:
                raise UnsupportedOperationError("controlled gate is outside the QASM 2 subset")
        if name not in {
            "h",
            "x",
            "y",
            "z",
            "s",
            "sdg",
            "t",
            "tdg",
            "rx",
            "ry",
            "rz",
            "p",
            "cp",
            "cx",
            "cy",
            "cz",
            "ccx",
            "swap",
        }:
            raise UnsupportedOperationError(f"QASM 2 cannot export operation {op.name!r}")
        if name in {"p", "cp"}:
            name = {"p": "u1", "cp": "cu1"}[name]
        if op.angle is not None:
            name += f"({op.angle:.17g})"
        args = ",".join(f"q[{q}]" for q in op.controls + op.wires)
        lines.append(f"{name} {args};")
    return "\n".join(lines) + "\n"


def from_qasm(source):
    if not isinstance(source, str) or len(source) > 1_000_000:
        raise ValueError("QASM source must be a string under one million characters")
    source = re.sub(r"//[^\n]*", "", source)
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    if source.rstrip() and not source.rstrip().endswith(";"):
        raise ValueError("QASM statements must end in semicolons")
    statements = [s.strip() for s in source.split(";") if s.strip()]
    if not statements or statements[0] != "OPENQASM 2.0":
        raise UnsupportedOperationError("only the documented OPENQASM 2.0 subset is supported")
    circuit, register, creg, csize = None, None, None, 0
    for statement in statements[1:]:
        if statement == 'include "qelib1.inc"':
            continue
        match = re.fullmatch(r"qreg\s+([A-Za-z_]\w*)\[(\d+)\]", statement)
        if match:
            if circuit is not None:
                raise UnsupportedOperationError("only one quantum register is supported")
            register = match[1]
            circuit = Circuit(int(match[2]))
            continue
        match = re.fullmatch(r"creg\s+([A-Za-z_]\w*)\[(\d+)\]", statement)
        if match:
            if creg is not None or int(match[2]) < 1:
                raise UnsupportedOperationError("one nonempty classical register is supported")
            creg, csize = match[1], int(match[2])
            continue
        if circuit is None:
            raise ValueError("declare qreg before quantum operations")

        def parse_wires(text):
            result = []
            for token in text.split(","):
                m = re.fullmatch(r"\s*([A-Za-z_]\w*)\[(\d+)\]\s*", token)
                if not m or m[1] != register or int(m[2]) >= circuit.n_qubits:
                    raise ValueError(f"invalid wire reference {token!r}")
                result.append(int(m[2]))
            return result

        match = re.fullmatch(r"measure\s+(.+?)\s*->\s*([A-Za-z_]\w*)\[(\d+)\]", statement)
        if match:
            targets = parse_wires(match[1])
            if len(targets) != 1 or match[2] != creg or int(match[3]) >= csize:
                raise ValueError("measurement needs one wire and a declared classical bit")
            circuit.measure(targets[0], f"{creg}{match[3]}")
            continue
        match = re.fullmatch(r"([a-z][a-z0-9]*)\s*(?:\((.*)\))?\s+(.+)", statement)
        if not match:
            raise UnsupportedOperationError(f"unsupported QASM statement: {statement!r}")
        name, angle, target_text = match.groups()
        targets = parse_wires(target_text)
        arity = {
            "h": 1,
            "x": 1,
            "y": 1,
            "z": 1,
            "s": 1,
            "sdg": 1,
            "t": 1,
            "tdg": 1,
            "rx": 1,
            "ry": 1,
            "rz": 1,
            "p": 1,
            "u1": 1,
            "cp": 2,
            "cu1": 2,
            "cx": 2,
            "cy": 2,
            "cz": 2,
            "ccx": 3,
            "swap": 2,
        }
        if name not in arity:
            raise UnsupportedOperationError(f"unsupported QASM gate: {name!r}")
        parameterized = name in {"rx", "ry", "rz", "p", "u1", "cp", "cu1"}
        if len(targets) != arity[name] or (angle is not None) != parameterized:
            raise ValueError(f"wrong argument count for {name}")
        method = getattr(circuit, {"u1": "p", "cu1": "cp"}.get(name, name))
        method(*([_angle(angle)] + targets if parameterized else targets))
    if circuit is None:
        raise ValueError("QASM source needs a qreg declaration")
    return circuit


def to_qiskit(circuit):
    """Return a Qiskit circuit; no account connection or job submission occurs."""
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import UnitaryGate
        from qiskit.quantum_info import Kraus
    except ImportError as error:
        raise ImportError('Install Qiskit with: python -m pip install "openqml[qiskit]"') from error
    keys = {
        op.key: i for i, op in enumerate(op for op in circuit._operations if op.name == "measure")
    }
    qc = QuantumCircuit(circuit.n_qubits, len(keys))

    def append(op):
        if op.name == "measure":
            q = op.wires[0]
            if op.basis == "Y":
                qc.sdg(q)
            if op.basis != "Z":
                qc.h(q)
            qc.measure(q, keys[op.key])
            if op.basis != "Z":
                qc.h(q)
            if op.basis == "Y":
                qc.s(q)
        elif op.name == "channel":
            qc.append(Kraus(list(op.kraus)).to_instruction(), list(op.wires))
        elif not op.wires:
            qc.global_phase += op.angle
        else:
            gate = UnitaryGate(op.matrix, label=op.label)
            if op.controls:
                gate = gate.control(len(op.controls))
            qc.append(gate, list(op.controls + op.wires))

    for op in circuit._operations:
        if op.condition is None:
            append(op)
        else:
            with qc.if_test((qc.clbits[keys[op.condition[0]]], op.condition[1])):
                append(op)
    return qc
