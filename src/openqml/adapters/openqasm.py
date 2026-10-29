"""OpenQASM 2/3 standard-gate and conditional-control conversion."""

from __future__ import annotations

import ast
import math
import re
from dataclasses import replace

from .._qasm import clean_source
from ..errors import ConversionError, ValidationError, conversion_error
from ..ir import GATES_V3, Circuit, ClassicalCondition, Operation, integer
from ..parameters import Expression, Parameter, expression, parameter_name
from ._common import bound_for_export


def _statements(text):
    """Split semicolons/braces outside strings, retaining original source lines."""
    tokens, start, line, start_line, quoted, escaped = [], 0, 1, 1, False, False
    for i, char in enumerate(text):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in ";{}":
            piece = text[start:i]
            leading = piece[: len(piece) - len(piece.lstrip())]
            location = start_line + leading.count("\n")
            if char == "}":
                if piece.strip():
                    raise conversion_error(
                        "SYNTAX", "Missing semicolon before closing brace.", f"line {location}"
                    )
                tokens.append(("close", "", line))
            elif piece.strip():
                tokens.append(("open" if char == "{" else "statement", piece.strip(), location))
            else:
                raise conversion_error("SYNTAX", "Empty statement or block header.", f"line {line}")
            start, start_line = i + 1, line
        if char == "\n":
            line += 1
    if text[start:].strip():
        raise conversion_error(
            "SYNTAX", "Every statement must end with a semicolon.", f"line {start_line}"
        )
    return tokens


def _phase_sum(values):
    while len(values) > 1:
        values = [values[i] + values[i + 1] for i in range(0, len(values) - 1, 2)] + (
            values[-1:] if len(values) % 2 else []
        )
    return values[0] if values else Expression()


def _angle(text, symbols, location):
    if len(text) > 16_384:
        raise conversion_error("EXPRESSION_LIMIT", "Angle expression is too long.", location)
    try:
        tree = ast.parse(text.strip(), mode="eval")
        pending, nodes = [(tree, 0)], 0
        while pending:
            part, depth = pending.pop()
            nodes += 1
            if nodes > 4096 or depth > 64:
                raise conversion_error(
                    "EXPRESSION_LIMIT", "Angle exceeds the 4096-node/64-level budget.", location
                )
            pending.extend((child, depth + 1) for child in ast.iter_child_nodes(part))

        def visit(node):
            if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
                return expression(node.value)
            if isinstance(node, ast.Name):
                if node.id in {"pi", "tau"}:
                    return expression({"pi": math.pi, "tau": math.tau}[node.id])
                if node.id in symbols:
                    return expression(Parameter(node.id))
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = visit(node.operand)
                return value if isinstance(node.op, ast.UAdd) else -value
            if isinstance(node, ast.BinOp):
                left, right = visit(node.left), visit(node.right)
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.Pow):
                    return left**right
                if isinstance(node.op, ast.Div):
                    return left / right
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and len(node.args) == 1
                and not node.keywords
            ):
                name = {"ln": "log"}.get(node.func.id, node.func.id)
                if name in {"sin", "cos", "tan", "exp", "log", "sqrt"}:
                    return getattr(visit(node.args[0]), name)()
            raise ValueError("unsupported expression")

        return visit(tree.body)
    except ConversionError:
        raise
    except RecursionError as exc:
        raise conversion_error(
            "EXPRESSION_LIMIT", "Angle exceeds parser recursion limits.", location
        ) from exc
    except (SyntaxError, ValueError, TypeError, OverflowError, ValidationError) as exc:
        raise conversion_error(
            "ANGLE_EXPRESSION",
            f"Unsupported angle expression: {text!r}.",
            location,
            "Use supported real arithmetic/functions within expression budgets.",
        ) from exc


def from_openqasm(
    text: str,
    *,
    expected_version=None,
    max_source_bytes=2_000_000,
    max_operations=100_000,
    max_register_bits=100_000,
    max_depth=64,
) -> Circuit:
    for label, limit in (
        ("max_source_bytes", max_source_bytes),
        ("max_operations", max_operations),
        ("max_register_bits", max_register_bits),
    ):
        integer(limit, label, 1)
    integer(max_depth, "max_depth")
    text = clean_source(text, max_source_bytes)
    statements = _statements(text)
    if not statements:
        raise conversion_error("QASM_VERSION", "Missing OPENQASM header.")
    header = re.fullmatch(r"OPENQASM\s+(2|3)(?:\.0)?", statements[0][1])
    if not header:
        raise conversion_error("QASM_VERSION", "Only OPENQASM 2.0 and 3.0 are supported.")
    version = int(header.group(1))
    if expected_version is not None and version != expected_version:
        raise conversion_error(
            "QASM_VERSION", "Source header differs from requested source format."
        )
    qregs, cregs, symbols = {}, {}, set()
    nq, nc, operations, phase = 0, 0, [], []
    stack, work, skip_else = [], 0, False

    def reference(token, registers, location):
        match = re.fullmatch(r"([A-Za-z_]\w*)(?:\[\s*(\d+)\s*\])?", token.strip(), re.ASCII)
        if not match or match.group(1) not in registers:
            raise conversion_error(
                "REGISTER_REFERENCE", f"Unknown register reference {token!r}.", location
            )
        start, width = registers[match.group(1)]
        if match.group(2) is None:
            return tuple(range(start, start + width))
        index = int(match.group(2))
        if index >= width:
            raise conversion_error("REGISTER_BOUNDS", "Register index is out of bounds.", location)
        return (start + index,)

    def append(op, location):
        nonlocal work
        work += 1
        if work > max_operations:
            raise conversion_error(
                "OPERATION_LIMIT", "Expanded gate count exceeds limit.", location
            )
        operations.append(op)

    for index, (kind, statement, line) in enumerate(statements[1:], 1):
        location = f"line {line}"
        if skip_else:
            skip_else = False
            continue
        if kind == "close":
            if not stack:
                raise conversion_error("SYNTAX", "Unmatched closing brace.", location)
            parent_ops, parent_phase, condition, true_body = stack.pop()
            child = Circuit(nq, tuple(operations), nc, _phase_sum(phase))
            if (
                true_body is None
                and index + 1 < len(statements)
                and statements[index + 1][:2] == ("open", "else")
            ):
                stack.append((parent_ops, parent_phase, condition, child))
                operations, phase, skip_else = [], [], True
                continue
            when_true, when_false = (
                (child, Circuit(nq)) if true_body is None else (true_body, child)
            )
            operations, phase = parent_ops, parent_phase
            operations.append(
                Operation("if_else", (), condition=condition, blocks=(when_true, when_false))
            )
            continue
        if kind == "open":
            guard = re.fullmatch(r"if\s*\(\s*(.*?)\s*\)", statement, re.S)
            if version != 3 or guard is None:
                raise conversion_error(
                    "UNSUPPORTED_STATEMENT",
                    "Only braced QASM 3 if/else blocks are supported here.",
                    location,
                )
            if len(stack) >= min(max_depth, 128):
                raise conversion_error(
                    "CONTROL_DEPTH_LIMIT",
                    "Conditional nesting exceeds max_depth or the 128-level traversal ceiling.",
                    location,
                )
            match = re.fullmatch(
                r"([A-Za-z_]\w*(?:\[\s*\d+\s*\])?)\s*(?:==\s*(true|false|\d+))?",
                guard.group(1),
                re.ASCII,
            )
            if match is None:
                raise conversion_error(
                    "UNSUPPORTED_STATEMENT", "Use bit/register equality guards.", location
                )
            bits = reference(match.group(1), cregs, location)
            literal = match.group(2)
            if literal in {"true", "false"} and len(bits) != 1:
                raise conversion_error(
                    "CLASSICAL_CONDITION", "Boolean equality requires one bit.", location
                )
            if literal is None and len(bits) != 1:
                raise conversion_error(
                    "UNSUPPORTED_STATEMENT", "Bare guards require a single bit.", location
                )
            value = 1 if literal in {None, "true"} else 0 if literal == "false" else int(literal)
            try:
                condition = ClassicalCondition(bits, value)
            except ValidationError as exc:
                raise conversion_error("CLASSICAL_CONDITION", str(exc), location) from exc
            if not nq:
                raise conversion_error(
                    "NO_QUBITS", "Declare qubits before control blocks.", location
                )
            # Empty blocks still consume work and cannot evade the source budget.
            work += 1
            if work > max_operations:
                raise conversion_error(
                    "OPERATION_LIMIT", "Conditional operations exceed limit.", location
                )
            stack.append((operations, phase, condition, None))
            operations, phase = [], []
            continue
        include = re.fullmatch(r'include\s+"([^"\n]+)"', statement)
        if include:
            if stack:
                raise conversion_error("BLOCK_DECLARATION", "Includes must be top-level.", location)
            expected = "qelib1.inc" if version == 2 else "stdgates.inc"
            if include.group(1) != expected:
                raise conversion_error(
                    "INCLUDE_UNSUPPORTED",
                    "Only the standard gate include is allowed.",
                    location,
                    "External includes are never read from disk.",
                )
            continue
        declaration2 = re.fullmatch(r"(qreg|creg)\s+([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]", statement)
        declaration3 = re.fullmatch(r"(qubit|bit)(?:\[\s*(\d+)\s*\])?\s+([A-Za-z_]\w*)", statement)
        if (declaration2 and version == 2) or (declaration3 and version == 3):
            if stack:
                raise conversion_error(
                    "BLOCK_DECLARATION", "Register declarations must be top-level.", location
                )
            if declaration2:
                kind, name, width = declaration2.groups()
            else:
                kind, width, name = declaration3.groups()
                width = width or "1"
            width = int(width)
            if name in qregs or name in cregs or name in symbols or width < 1:
                raise conversion_error(
                    "REGISTER_DECLARATION", "Invalid or duplicate register.", location
                )
            if nq + nc + width > max_register_bits:
                raise conversion_error(
                    "REGISTER_LIMIT", "Register size exceeds the configured limit.", location
                )
            if kind in {"qreg", "qubit"}:
                qregs[name] = (nq, width)
                nq += width
            else:
                cregs[name] = (nc, width)
                nc += width
            continue
        symbol = re.fullmatch(r"input\s+float(?:\[64\])?\s+([A-Za-z_]\w*)", statement)
        if symbol and version == 3:
            if stack:
                raise conversion_error(
                    "BLOCK_DECLARATION", "Input declarations must be top-level.", location
                )
            name = symbol.group(1)
            from ..conversion import _QASM_RESERVED

            if name in symbols or name in qregs or name in cregs or name in _QASM_RESERVED:
                raise conversion_error(
                    "PARAMETER_DECLARATION", "Duplicate/reserved parameter.", location
                )
            parameter_name(name)
            symbols.add(name)
            continue
        measurement2 = re.fullmatch(r"measure\s+(.+?)\s*->\s*(.+)", statement)
        measurement3 = re.fullmatch(r"(.+?)\s*=\s*measure\s+(.+)", statement)
        if (measurement2 and version == 2) or (measurement3 and version == 3):
            qref, cref = measurement2.groups() if measurement2 else reversed(measurement3.groups())
            qs, cs = reference(qref, qregs, location), reference(cref, cregs, location)
            if len(qs) != len(cs):
                raise conversion_error(
                    "MEASUREMENT_WIDTH", "Measurement register widths differ.", location
                )
            for q, c in zip(qs, cs, strict=True):
                append(Operation("measure", (q,), clbits=(c,)), location)
            continue
        global_phase = re.fullmatch(r"gphase\s*\((.*)\)", statement, flags=re.S)
        if version == 3 and global_phase:
            work += 1
            if work > max_operations:
                raise conversion_error(
                    "OPERATION_LIMIT", "Phase statements exceed limit.", location
                )
            phase.append(_angle(global_phase.group(1), symbols, location))
            continue
        prefix = re.match(r"([A-Za-z_]\w*)", statement, flags=re.ASCII)
        if not prefix:
            raise conversion_error("SYNTAX", "Expected a supported gate statement.", location)
        name = prefix.group(1)
        rest = statement[prefix.end() :].strip()
        parameters = ()
        if rest.startswith("("):
            depth, end = 0, None
            for i, char in enumerate(rest):
                depth += (char == "(") - (char == ")")
                if depth == 0:
                    end = i
                    break
            if end is None:
                raise conversion_error("SYNTAX", "Unclosed gate parameters.", location)
            parameters = tuple(_angle(p, symbols, location) for p in rest[1:end].split(","))
            rest = rest[end + 1 :].strip()
        if version == 2 and name in {"U", "u", "u2", "u3"}:
            raise conversion_error(
                "QASM2_U_PHASE_CONVENTION",
                "OpenQASM 2 U conventions differ by a global phase across dialects.",
                location,
                "Import the native Qiskit circuit, or lower U to explicit phase/rotation gates.",
            )
        name = {"u1": "p", "i": "id", "CX": "cx", "U": "u", "u3": "u"}.get(name, name)
        if name not in GATES_V3 | {"reset"} or name in {"measure", "unitary"}:
            raise conversion_error(
                "UNSUPPORTED_STATEMENT",
                f"Unsupported gate or language feature {name!r}.",
                location,
                "Use standard gates/reset; definitions and control flow need the SDK frontend.",
            )
        groups = [reference(token, qregs, location) for token in rest.split(",")]
        try:
            if name == "barrier":
                append(
                    Operation(name, tuple(q for group in groups for q in group), parameters),
                    location,
                )
            else:
                width = max(map(len, groups))
                if any(len(group) not in {1, width} for group in groups):
                    raise conversion_error(
                        "BROADCAST_WIDTH", "Gate register widths differ.", location
                    )
                for row in range(width):
                    append(
                        Operation(
                            name,
                            tuple(group[0 if len(group) == 1 else row] for group in groups),
                            parameters,
                        ),
                        location,
                    )
        except ValidationError as exc:
            raise conversion_error("INVALID_OPERATION", str(exc), location) from exc
    if stack:
        raise conversion_error("SYNTAX", "Unclosed conditional block.")
    if nq == 0:
        raise conversion_error("NO_QUBITS", "Declare at least one qubit.")

    def widths(body):
        return replace(
            body,
            num_qubits=nq,
            num_clbits=nc,
            operations=tuple(
                replace(op, blocks=tuple(widths(b) for b in op.blocks)) for op in body.operations
            ),
        )

    circuit = Circuit(
        nq,
        tuple(replace(op, blocks=tuple(widths(b) for b in op.blocks)) for op in operations),
        nc,
        _phase_sum(phase),
    )
    unused = symbols - set(circuit.parameters)
    if unused:
        raise conversion_error(
            "UNUSED_INPUT_DECLARATION",
            f"IR cannot retain unused input declarations: {sorted(unused)}.",
            "parameters",
            "Remove unused inputs before conversion.",
        )
    return circuit


def _format_angle(angle):
    if angle.op is not None:
        values = [_format_angle(v) for v in angle.args]
        if angle.op in {"add", "mul", "div", "pow"}:
            operator = {"add": "+", "mul": "*", "div": "/", "pow": "**"}[angle.op]
            return f"({values[0]} {operator} {values[1]})"
        return f"{angle.op}({values[0]})"
    parts = [repr(angle.constant)] if angle.constant or not angle.terms else []
    parts.extend(f"({coefficient!r}*{name})" for name, coefficient in angle.terms)
    return " + ".join(parts)


def to_openqasm(
    circuit: Circuit,
    *,
    version=3,
    bindings=None,
    max_operations=100_000,
    max_depth=64,
    max_output_statements=100_000,
) -> str:
    if type(version) is not int or version not in {2, 3}:
        raise conversion_error("QASM_VERSION", "Only versions 2 and 3 are supported.")
    from ..compiler import QASM_BASIS, _check_circuit_limits, _qasm_basis, decompose

    _check_circuit_limits(circuit, max_operations=max_operations, max_depth=max_depth)
    circuit = bound_for_export(circuit, bindings)
    from ..conversion import diagnose

    limits = {
        "max_operations": max_operations,
        "max_depth": max_depth,
        "max_output_statements": max_output_statements,
    }
    diagnostics = diagnose(circuit, f"openqasm{version}", **limits)
    if diagnostics:
        raise ConversionError(diagnostics)
    circuit = decompose(
        circuit, basis=_qasm_basis(circuit), max_operations=max_operations, max_depth=max_depth
    )
    if any(op.name == "unitary" for _, op in circuit.walk_operations()):
        from ._common import dependency
        from .qiskit import from_qiskit, to_qiskit

        qiskit = dependency("qiskit", "qiskit")
        circuit = from_qiskit(
            qiskit.transpile(to_qiskit(circuit), basis_gates=["u", "cx"], optimization_level=0),
            max_operations=max_operations,
            max_depth=max_depth,
        )
    circuit = decompose(
        circuit, basis=QASM_BASIS, max_operations=max_operations, max_depth=max_depth
    )
    diagnostics = diagnose(circuit, f"openqasm{version}", **limits)
    if diagnostics:
        raise ConversionError(diagnostics)
    lines = [
        f"OPENQASM {version}.0;",
        'include "qelib1.inc";' if version == 2 else 'include "stdgates.inc";',
    ]
    if version == 3:
        lines.extend(f"input float[64] {name};" for name in circuit.parameters)
    lines.append(
        f"qreg q[{circuit.num_qubits}];" if version == 2 else f"qubit[{circuit.num_qubits}] q;"
    )
    if circuit.num_clbits:
        lines.append(
            f"creg c[{circuit.num_clbits}];" if version == 2 else f"bit[{circuit.num_clbits}] c;"
        )

    def emit_body(body, indent=""):
        if version == 3 and (body.global_phase.parameters or body.global_phase.constant):
            lines.append(indent + f"gphase({_format_angle(body.global_phase)});")
        for op in body.operations:
            if len(lines) >= max_output_statements:
                raise conversion_error(
                    "QASM_EXPORT_LIMIT", "Expanded conditional output exceeds the statement budget."
                )
            if op.name == "if_else":
                if len(op.condition.bits) > 64:
                    raise conversion_error(
                        "QASM_EXPORT_LIMIT", "A conditional QASM guard is limited to 64 bits."
                    )

                # Nested guards preserve one-time branch selection, including
                # writes to condition bits inside either selected body. False
                # bodies are duplicated syntactically along exclusive paths.
                def guard(position, prefix, op=op):
                    bit = op.condition.bits[position]
                    value = (op.condition.value >> position) & 1
                    literal = "true" if value else "false"
                    lines.append(prefix + f"if (c[{bit}] == {literal}) {{")
                    if position + 1 == len(op.condition.bits):
                        emit_body(op.blocks[0], prefix + "  ")
                    else:
                        guard(position + 1, prefix + "  ")
                    lines.append(prefix + "} else {")
                    emit_body(op.blocks[1], prefix + "  ")
                    lines.append(prefix + "}")

                guard(0, indent)
            elif op.name == "measure":
                q, c = op.qubits[0], op.clbits[0]
                lines.append(
                    indent
                    + (
                        f"measure q[{q}] -> c[{c}];"
                        if version == 2
                        else f"c[{c}] = measure q[{q}];"
                    )
                )
            else:
                name, params = op.name, op.params
                if name == "p" and version == 2:
                    name = "u1"
                if name == "id":
                    name, params = "rx", (expression(0),)
                args = "(" + ", ".join(_format_angle(p) for p in params) + ")" if params else ""
                lines.append(
                    indent + f"{name}{args} " + ", ".join(f"q[{q}]" for q in op.qubits) + ";"
                )

    emit_body(circuit)
    if len(lines) > max_output_statements:
        raise conversion_error(
            "QASM_EXPORT_LIMIT", "Expanded OpenQASM exceeds max_output_statements."
        )
    return "\n".join(lines) + "\n"
