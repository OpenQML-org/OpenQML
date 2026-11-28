"""OpenQASM 2/3 standard-gate and conditional-control conversion."""

from __future__ import annotations

import ast
import math
import re
from dataclasses import replace

from ...errors import ConversionError, ValidationError, conversion_error
from ...ir.core import GATES_V3, Circuit, ClassicalCondition, Operation, integer
from ...ir.parameters import Expression, Parameter, _postorder, expression, parameter_name
from .._qasm import clean_source
from ._common import bound_for_export


def _statements(text):
    """Split semicolons/braces outside strings, retaining original source lines."""
    tokens, start, line, start_line, quoted, escaped = [], 0, 1, 1, False, False
    brackets = 0
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
        elif char == "[":
            brackets += 1
        elif char == "]":
            brackets -= 1
        elif char in ";{}" and brackets == 0:
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
    max_register_bits=None,
    max_depth=64,
) -> Circuit:
    for label, limit in (
        ("max_source_bytes", max_source_bytes),
        ("max_operations", max_operations),
    ):
        integer(limit, label, 1)
    if max_register_bits is not None:
        integer(max_register_bits, "max_register_bits", 1)
    integer(max_depth, "max_depth")
    text = clean_source(text, max_source_bytes)
    statements = _statements(text)
    if not statements:
        raise conversion_error("QASM_VERSION", "Missing OPENQASM header.")
    header = re.fullmatch(r"OPENQASM\s+(2(?:\.0)?|3(?:\.[01])?)", statements[0][1])
    if not header:
        raise conversion_error("QASM_VERSION", "Expected an OpenQASM 2.0, 3.0 or 3.1 header.")
    version = int(header.group(1)[0])
    if expected_version is not None and version != expected_version:
        raise conversion_error(
            "QASM_VERSION", "Source header differs from requested source format."
        )
    from ._qasm_runtime import (
        Predicate,
        Switch,
        addressed,
        comparison,
        counter,
        runtime_address,
        split_operands,
        unsigned_word,
    )

    qregs, cregs, symbols, unsigned = {}, {}, set(), set()
    bit_decl = re.compile(r"bit(?:\[\s*(\d+)\s*\])?\s+[A-Za-z_]\w*", re.ASCII)
    user_bits = sum(
        int(m[1] or 1)
        for k, text, _ in statements
        if version == 3 and k == "statement" and (m := bit_decl.fullmatch(text))
    )
    next_bit = 0
    bounds, saved = {}, []

    nq, nc, operations, phase = 0, 0, [], []
    stack, work, skip_else = [], 0, False

    def reference(token, registers, location, *, dynamic=False):
        match = re.fullmatch(
            r"([A-Za-z_]\w*)(?:\[\s*(\d+|\{\s*\d+(?:\s*,\s*\d+)*\s*\})\s*\])?",
            token.strip(),
            re.ASCII,
        )
        if not match and dynamic and version == 3:
            return runtime_address(token, registers, cregs, bounds, max_operations - work, location)
        if not match or match.group(1) not in registers:
            raise conversion_error(
                "REGISTER_REFERENCE", f"Unknown register reference {token!r}.", location
            )
        start, width = registers[match.group(1)]
        if match.group(2) is None:
            if width > max_operations - work:
                raise conversion_error(
                    "OPERATION_LIMIT",
                    "Register expansion exceeds the remaining work budget.",
                    location,
                )
            return tuple(range(start, start + width))
        selector = match.group(2)
        indices = (
            tuple(int(i.strip()) for i in selector[1:-1].split(","))
            if selector.startswith("{")
            else (int(selector),)
        )
        if len(indices) > max_operations - work:
            raise conversion_error(
                "OPERATION_LIMIT", "Index set exceeds the work budget.", location
            )
        if any(index >= width for index in indices):
            raise conversion_error("REGISTER_BOUNDS", "Register index is out of bounds.", location)
        return tuple(start + index for index in indices)

    def append(op, location):
        nonlocal work
        work += 1
        if work > max_operations:
            raise conversion_error(
                "OPERATION_LIMIT", "Expanded gate count exceeds limit.", location
            )
        operations.append(op)

    def add_bound(condition, truth):
        if isinstance(condition, Predicate):
            digits, upper = condition.digits, condition.high if truth else condition.low
        elif isinstance(condition, ClassicalCondition) and truth:
            digits, upper = tuple((b, i) for i, b in enumerate(condition.bits)), condition.value + 1
        else:
            return
        if upper is not None:
            bounds[digits] = min(upper, bounds.get(digits, upper))

    def extend(new_ops, location):
        nonlocal work
        for op in new_ops:
            writes = set(op.clbits)
            for child in op.blocks:
                for _, sub in child.walk_operations():
                    work += 1
                    writes.update(sub.clbits)
            if writes:
                for scope in (*saved, bounds):
                    for digits in tuple(scope):
                        if any(b in writes for b, _ in digits):
                            del scope[digits]
            append(op, location)

    def emit(groups, factory, location):
        extend(
            addressed(
                groups,
                factory,
                nq or 1,
                nc,
                max_operations - work,
                min(max_depth, 128) - len(stack),
                location,
            ),
            location,
        )

    def copy_bit(source, target):
        empty = Circuit(nq or 1, num_clbits=nc)
        return Operation(
            "if_else",
            (),
            condition=ClassicalCondition((source,), 1),
            blocks=(empty.classical_set(target, 1), empty.classical_set(target, 0)),
        )

    def classical_guard(source, location):
        """Parse equality conjunctions without expanding their branch bodies."""
        position, depth, expect_term, terms = 0, 0, True, 0
        assignments = {}
        atom = re.compile(
            r"([A-Za-z_]\w*(?:\[\s*(?:\d+|\{\s*\d+(?:\s*,\s*\d+)*\s*\})\s*\])?)"
            r"\s*(?:==\s*(true|false|\d+))?",
            re.ASCII,
        )
        while position < len(source):
            if source[position].isspace():
                position += 1
                continue
            if expect_term and source[position] == "(":
                depth += 1
                if depth > 64:
                    raise conversion_error(
                        "EXPRESSION_LIMIT", "Guard grouping exceeds 64 levels.", location
                    )
                position += 1
                continue
            if not expect_term and source[position] == ")" and depth:
                depth -= 1
                position += 1
                continue
            if not expect_term and source.startswith("&&", position):
                expect_term, position = True, position + 2
                continue
            match = atom.match(source, position) if expect_term else None
            if match is None:
                raise conversion_error(
                    "UNSUPPORTED_STATEMENT",
                    "Use bit/register equality guards joined by &&.",
                    location,
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
                ClassicalCondition(bits, value)
            except ValidationError as exc:
                raise conversion_error("CLASSICAL_CONDITION", str(exc), location) from exc
            for i, bit in enumerate(bits):
                wanted = (value >> i) & 1
                if bit in assignments and assignments[bit] != wanted:
                    raise conversion_error(
                        "CLASSICAL_CONDITION", "Contradictory equality terms.", location
                    )
                assignments[bit] = wanted
            terms += 1
            if terms > min(max_operations, 4096):
                raise conversion_error(
                    "EXPRESSION_LIMIT", "Guard exceeds its term budget.", location
                )
            position, expect_term = match.end(), False
        if expect_term or depth:
            raise conversion_error("SYNTAX", "Incomplete conditional guard.", location)
        return ClassicalCondition(
            tuple(assignments), sum(value << i for i, value in enumerate(assignments.values()))
        )

    for index, (kind, statement, line) in enumerate(statements[1:], 1):
        location = f"line {line}"
        if skip_else:
            skip_else = False
            continue
        if kind == "close":
            if not stack:
                raise conversion_error("SYNTAX", "Unmatched closing brace.", location)
            control, parent_ops, parent_phase, condition, true_body = stack.pop()
            child = Circuit(nq, tuple(operations), nc, _phase_sum(phase))
            if (
                control == "if"
                and true_body is None
                and index + 1 < len(statements)
                and statements[index + 1][:2] == ("open", "else")
            ):
                stack.append((control, parent_ops, parent_phase, condition, child))
                bounds = saved[-1].copy()
                add_bound(condition, False)
                operations, phase, skip_else = [], [], True
                continue
            operations, phase = parent_ops, parent_phase
            bounds = saved.pop()
            if control == "case":
                stack[-1][3].add(condition, child, location)
                continue
            empty = Circuit(nq, num_clbits=nc)
            if control == "switch":
                part = condition.lower(
                    nq, nc, max_operations - true_body, max_depth - len(stack), location
                )
                work = max(work, true_body + sum(1 for _ in part.walk_operations()))
                operations.extend(part.operations)
                phase.append(part.global_phase)
            elif control == "while":
                if isinstance(condition, Predicate):
                    if condition.equality is not None and not condition.inverted:
                        condition = condition.equality
                    else:
                        test = condition.select(
                            empty,
                            empty.break_loop(),
                            nq,
                            nc,
                            max_operations - work,
                            max_depth - len(stack) - 1,
                            location,
                        )
                        work += sum(1 for _ in test.walk_operations())
                        child = test.compose(child)
                        condition = None
                if condition is not False:
                    operations.append(
                        Operation("while_loop", (), condition=condition, blocks=(child,))
                    )
            else:
                high, low = (child, empty) if true_body is None else (true_body, child)
                if isinstance(condition, Predicate):
                    charged = sum(1 for body in (high, low) for _ in body.walk_operations())
                    part = condition.select(
                        high,
                        low,
                        nq,
                        nc,
                        max_operations - work + charged + 1,
                        max_depth - len(stack),
                        location,
                    )
                    work += max(0, sum(1 for _ in part.walk_operations()) - charged - 1)
                elif isinstance(condition, bool):
                    part = high if condition else low
                else:
                    part = empty.if_else(condition, high, low)
                operations.extend(part.operations)
                phase.append(part.global_phase)
            if work > max_operations:
                raise conversion_error(
                    "OPERATION_LIMIT", "Control expansion exceeds max_operations.", location
                )
            continue
        in_switch = bool(stack and stack[-1][0] == "switch")
        if in_switch and kind != "open":
            raise conversion_error(
                "SYNTAX", "A switch must contain case or default arms.", location
            )
        if kind == "open":
            guard = re.fullmatch(r"(if|while|switch)\s*\(\s*(.*?)\s*\)", statement, re.S)
            case = re.fullmatch(r"case\s+(\d+(?:\s*,\s*\d+)*)|default", statement)
            if version != 3 or not (case or guard) or bool(case) != in_switch:
                raise conversion_error(
                    "UNSUPPORTED_STATEMENT",
                    "Expected a braced if, while, switch or case.",
                    location,
                )
            if not nq:
                raise conversion_error(
                    "NO_QUBITS", "Declare qubits before control blocks.", location
                )
            if len(stack) >= min(max_depth, 128):
                raise conversion_error(
                    "CONTROL_DEPTH_LIMIT", "Control nesting exceeds max_depth.", location
                )
            checkpoint = work
            if case:
                control = "case"
                condition = None if not case[1] else tuple(int(v) for v in case[1].split(","))
                work += len(condition) if condition else 1
            else:
                control, test = guard.groups()
                if control == "switch":
                    condition = Switch(unsigned_word(test, cregs, max_operations - work, location))
                elif test.strip() in {"true", "false"}:
                    condition = test.strip() == "true"
                    if condition and control == "while":
                        condition = None
                else:
                    try:
                        condition = classical_guard(test, location)
                    except ConversionError:
                        if not re.search(r"!=|<=|>=|<|>|^uint", test.strip()):
                            raise
                        condition = comparison(
                            test, cregs, max_operations - work, max_depth - len(stack), location
                        )
            work += 1
            if work > max_operations:
                raise conversion_error(
                    "OPERATION_LIMIT", "Control exceeds the operation limit.", location
                )
            stack.append(
                (control, operations, phase, condition, checkpoint if control == "switch" else None)
            )
            saved.append(bounds)
            bounds = {} if control == "while" else bounds.copy()
            add_bound(condition, True)
            operations, phase = [], []
            continue
        if version == 3 and statement in {"break", "continue"}:
            if not any(f[0] == "while" for f in stack):
                raise conversion_error(
                    "CLASSICAL_CONTROL", f"{statement} needs an enclosing loop.", location
                )
            append(Operation(statement + "_loop", ()), location)
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
        uint = re.fullmatch(
            r"uint\[\s*(\d+)\s*\]\s+([A-Za-z_]\w*)\s*=\s*(.+)", statement, re.ASCII | re.S
        )
        if version == 3 and uint:
            from ..conversion import _QASM_RESERVED

            width, name, initial = int(uint[1]), uint[2], uint[3]
            if (
                stack
                or width < 1
                or name in (_QASM_RESERVED | symbols | qregs.keys() | cregs.keys())
            ):
                raise conversion_error(
                    "REGISTER_DECLARATION", "Expected a new top-level unsigned register.", location
                )
            start = max(nc, user_bits)
            if width > max_operations - work or (
                max_register_bits is not None and nq + start + width > max_register_bits
            ):
                raise conversion_error(
                    "REGISTER_LIMIT", "Unsigned storage exceeds the import budget.", location
                )
            literal = int(initial) if initial.isdecimal() else None
            digits = (
                {}
                if literal is not None
                else {
                    i: b for b, i in unsigned_word(initial, cregs, max_operations - work, location)
                }
            )
            if literal is not None and literal.bit_length() > width:
                raise conversion_error(
                    "CLASSICAL_STORE_WIDTH", "Initializer exceeds its width.", location
                )
            nc = start + width
            cregs[name] = (start, width)
            unsigned.add(name)
            for i in range(width):
                if i in digits:
                    extend((copy_bit(digits[i], start + i),), location)
                else:
                    append(
                        Operation(
                            "classical_set",
                            (),
                            (expression(((literal or 0) >> i) & 1),),
                            (start + i,),
                        ),
                        location,
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
            total = nq + nc + width if kind in {"qreg", "qubit"} else nq + max(nc, next_bit + width)
            if max_register_bits is not None and total > max_register_bits:
                raise conversion_error(
                    "REGISTER_LIMIT", "Register size exceeds the configured limit.", location
                )
            if kind in {"qreg", "qubit"}:
                qregs[name] = (nq, width)
                nq += width
            else:
                cregs[name] = (next_bit, width)
                next_bit += width
                nc = max(nc, next_bit)
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

            def measure(groups, location=location):
                qs, cs = groups
                if len(qs) != len(cs):
                    raise conversion_error(
                        "MEASUREMENT_WIDTH", "Measurement register widths differ.", location
                    )
                return (
                    Operation("measure", (q,), clbits=(c,)) for q, c in zip(qs, cs, strict=True)
                )

            emit(
                (
                    reference(qref, qregs, location, dynamic=True),
                    reference(cref, cregs, location, dynamic=True),
                ),
                measure,
                location,
            )
            continue
        update = re.fullmatch(r"([A-Za-z_]\w*)\s*([+-])=\s*(\d+)", statement) or re.fullmatch(
            r"([A-Za-z_]\w*)\s*=\s*\1\s*([+-])\s*(\d+)", statement
        )
        if version == 3 and update and update[1] in unsigned:
            start, width = cregs[update[1]]
            extend(
                counter(
                    start,
                    width,
                    int(update[3]),
                    update[2] == "-",
                    nq or 1,
                    nc,
                    max_operations - work,
                    max_depth - len(stack),
                    location,
                ),
                location,
            )
            continue
        assignment = re.fullmatch(r"(.+?)\s*=\s*(true|false|\d+)", statement)
        if version == 3 and assignment:
            value = (
                int(assignment[2] == "true")
                if assignment[2] in {"true", "false"}
                else int(assignment[2])
            )

            def assign(groups, value=value, location=location):
                bits = groups[0]
                if value.bit_length() > len(bits):
                    raise conversion_error(
                        "CLASSICAL_STORE_WIDTH", "Assigned value exceeds its width.", location
                    )
                return (
                    Operation("classical_set", (), (expression((value >> i) & 1),), (bit,))
                    for i, bit in enumerate(bits)
                )

            emit((reference(assignment[1], cregs, location, dynamic=True),), assign, location)
            continue
        bit_ref = r"([A-Za-z_]\w*(?:\s*\[.+\])?)"
        copy = re.fullmatch(bit_ref + r"\s*=\s*" + bit_ref, statement)
        if version == 3 and copy:

            def copy_one(groups, location=location):
                dest, src = groups
                if len(dest) != 1 or len(src) != 1:
                    raise conversion_error(
                        "UNSUPPORTED_STATEMENT", "Expected a bit-to-bit assignment.", location
                    )
                return (copy_bit(src[0], dest[0]),)

            emit(
                tuple(reference(v, cregs, location, dynamic=True) for v in copy.groups()),
                copy_one,
                location,
            )
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
        groups = [reference(token, qregs, location, dynamic=True) for token in split_operands(rest)]

        def gates(groups, name=name, parameters=parameters, location=location):
            if name == "barrier":
                yield Operation(name, tuple(b for group in groups for b in group), parameters)
                return
            width = max(map(len, groups))
            if any(len(group) not in {1, width} for group in groups):
                raise conversion_error("BROADCAST_WIDTH", "Gate register widths differ.", location)
            for row in range(width):
                yield Operation(
                    name,
                    tuple(group[0 if len(group) == 1 else row] for group in groups),
                    parameters,
                )

        try:
            emit(groups, gates, location)
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
    from ...ir.compiler import _check_circuit_limits

    _check_circuit_limits(circuit, max_operations=max_operations, max_depth=max_depth)
    unused = symbols - set(circuit.parameters)
    if unused:
        raise conversion_error(
            "UNUSED_INPUT_DECLARATION",
            f"IR cannot retain unused input declarations: {sorted(unused)}.",
            "parameters",
            "Remove unused inputs before conversion.",
        )
    return circuit


def _format_angle(angle, cache=None):
    cache = {} if cache is None else cache
    if id(angle) in cache:
        return cache[id(angle)][0]
    for value in _postorder((angle,), cache):
        children = [cache[id(arg)] for arg in value.args]
        count = 1 + len(value.terms) + sum(child[1] for child in children)
        depth = 1 + max((child[2] for child in children), default=-1)
        if count > 4096 or depth > 64:
            raise conversion_error(
                "EXPRESSION_LIMIT", "Angle exceeds the 4096-node/64-level budget."
            )
        if value.op is None:
            parts = [repr(value.constant)] if value.constant or not value.terms else []
            parts.extend(f"({coefficient!r}*{name})" for name, coefficient in value.terms)
            text = " + ".join(parts)
            if len(parts) > 1 or text.startswith("-"):
                text = "(" + text + ")"
        elif value.op in {"add", "mul", "div", "pow"}:
            operator = {"add": "+", "mul": "*", "div": "/", "pow": "**"}[value.op]
            text = f"({children[0][0]} {operator} {children[1][0]})"
        else:
            text = f"{value.op}({children[0][0]})"
        if len(text) > 16_384:
            raise conversion_error("EXPRESSION_LIMIT", "Angle expression is too long.")
        cache[id(value)] = (text, count, depth)
    result = cache[id(angle)][0]
    try:
        pending, nodes = [(ast.parse(result, mode="eval"), 0)], 0
        while pending:
            part, depth = pending.pop()
            nodes += 1
            if nodes > 4096 or depth > 64:
                raise conversion_error(
                    "EXPRESSION_LIMIT", "Angle exceeds the 4096-node/64-level budget."
                )
            pending.extend((child, depth + 1) for child in ast.iter_child_nodes(part))
    except (SyntaxError, RecursionError) as exc:
        raise conversion_error(
            "EXPRESSION_LIMIT", "Angle exceeds parser serialization limits."
        ) from exc
    return result


def _emit_qasm(circuit, version, *, max_output_statements, max_output_bytes, collect=True):
    """Use one bounded emitter for planning and artifact construction."""
    lines, cache, count, size = [], {}, 0, 0

    def write(line):
        nonlocal count, size
        count += 1
        size += len(line.encode("utf-8")) + 1
        if count > max_output_statements or size > max_output_bytes:
            raise conversion_error(
                "QASM_EXPORT_LIMIT",
                "OpenQASM exceeds its output statement or byte budget.",
            )
        if collect:
            lines.append(line)

    write(f"OPENQASM {version}.0;")
    write('include "qelib1.inc";' if version == 2 else 'include "stdgates.inc";')
    if version == 3:
        for name in circuit.parameters:
            write(f"input float[64] {name};")
    write(f"qreg q[{circuit.num_qubits}];" if version == 2 else f"qubit[{circuit.num_qubits}] q;")
    if circuit.num_clbits:
        write(f"creg c[{circuit.num_clbits}];" if version == 2 else f"bit[{circuit.num_clbits}] c;")

    def emit_body(body, indent=""):
        if version == 3 and (body.global_phase.parameters or body.global_phase.constant):
            write(indent + f"gphase({_format_angle(body.global_phase, cache)});")
        for op in body.operations:
            if op.name in {"while_loop", "if_else"}:
                if version != 3:
                    raise conversion_error(
                        "CONTROL_FLOW_UNSUPPORTED", "Conditional control requires OpenQASM 3."
                    )
                if op.condition is not None and len(op.condition.bits) > 64:
                    raise conversion_error(
                        "QASM_EXPORT_LIMIT", "A conditional QASM guard is limited to 64 bits."
                    )
                bits, value = (
                    ((), 0) if op.condition is None else (op.condition.bits, op.condition.value)
                )
                if not bits:
                    guard = "true"
                elif len(bits) == 1:
                    guard = f"c[{bits[0]}] == {'true' if value else 'false'}"
                elif len(bits) == circuit.num_clbits and all(
                    bit == index for index, bit in enumerate(bits)
                ):
                    guard = f"c == {value}"
                else:
                    selector = "c[{" + ", ".join(str(bit) for bit in bits) + "}]"
                    guard = f"{selector} == {value}"
                # Each selector is evaluated once even if its body rewrites it.
                keyword = "while" if op.name == "while_loop" else "if"
                write(indent + f"{keyword} ({guard}) {{")
                emit_body(op.blocks[0], indent + "  ")
                if op.name == "if_else":
                    write(indent + "} else {")
                    emit_body(op.blocks[1], indent + "  ")
                write(indent + "}")
            elif op.name in {"break_loop", "continue_loop"}:
                write(indent + op.name.split("_")[0] + ";")
            elif op.name == "classical_set":
                if version != 3:
                    raise conversion_error("CLASSICAL_STORE_UNSUPPORTED", "Stores require QASM 3.")
                write(indent + f"c[{op.clbits[0]}] = {int(op.params[0].constant)};")
            elif op.name == "measure":
                q, c = op.qubits[0], op.clbits[0]
                write(
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
                args = (
                    "(" + ", ".join(_format_angle(p, cache) for p in params) + ")" if params else ""
                )
                write(indent + f"{name}{args} " + ", ".join(f"q[{q}]" for q in op.qubits) + ";")

    emit_body(circuit)
    return "\n".join(lines) + "\n" if collect else None


def to_openqasm(
    circuit: Circuit,
    *,
    version=3,
    bindings=None,
    max_operations=100_000,
    max_depth=64,
    max_output_statements=100_000,
    max_output_bytes=2_000_000,
) -> str:
    if type(version) is not int or version not in {2, 3}:
        raise conversion_error("QASM_VERSION", "Only versions 2 and 3 are supported.")
    from ...ir.compiler import QASM_BASIS, _check_circuit_limits, _qasm_basis, decompose

    _check_circuit_limits(circuit, max_operations=max_operations, max_depth=max_depth)
    circuit = bound_for_export(circuit, bindings)
    from ..conversion import diagnose

    limits = {
        "max_operations": max_operations,
        "max_depth": max_depth,
        "max_output_statements": max_output_statements,
        "max_output_bytes": max_output_bytes,
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
    return _emit_qasm(
        circuit,
        version,
        max_output_statements=max_output_statements,
        max_output_bytes=max_output_bytes,
    )
