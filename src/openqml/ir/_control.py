"""Lexical scope and control-flow threading for runtime programs."""

from dataclasses import replace

from ..errors import ResourceError, ValidationError, conversion_error
from .compiler import _check_circuit_limits
from .parameters import Expression

INDEX_RANGE_ERROR = "CLASSICAL_INDEX_RANGE: A runtime index is outside its register."


class RuntimeIndexError(ValidationError):
    """Execution reached an invalid runtime address."""


def validate_loop_scope(circuit):
    pending = [(circuit, False)]
    while pending:
        body, inside = pending.pop()
        for op in body.operations:
            if op.name in {"break_loop", "continue_loop"} and not inside:
                raise conversion_error(
                    "CLASSICAL_CONTROL", f"{op.name} requires an enclosing loop."
                )
            pending.extend((b, inside or op.name == "while_loop") for b in op.blocks)


def control_program(circuit, limit):
    if limit < 1:
        raise ResourceError("Runtime control metadata exceeds max_memory_mb.")
    _check_circuit_limits(circuit, max_operations=limit, max_depth=128)
    validate_loop_scope(circuit)
    code = []

    def block(body, entry, exits=None):
        for op in reversed(body.operations):
            if op.name == "break_loop":
                entry = exits[0]
            elif op.name == "continue_loop":
                entry = exits[1]
            elif op.name == "while_loop":
                head = len(code)
                code.append(None)
                loop = block(op.blocks[0], head, (entry, head))
                code[head] = op, loop, entry
                entry = head
            elif op.name == "if_else":
                high, low = (block(child, entry, exits) for child in op.blocks)
                entry = len(code)
                code.append((op, high, low))
            else:
                code.append((op, entry, -1))
                entry = len(code) - 1
        return entry

    entry = block(circuit, -1)
    return entry, tuple(code)


def flatten_static_control(circuit):
    """Resolve loops whose execution and exits are independent of measurement."""
    if not circuit.has_runtime_control:
        return circuit
    validate_loop_scope(circuit)
    unknown, operations = set(), []
    phase = Expression()

    class Unresolved(Exception):
        pass

    def guard(condition):
        if condition is None:
            return True
        if unknown.intersection(condition.bits):
            raise Unresolved
        return condition.value == 0

    def block(body):
        nonlocal phase
        phase += body.global_phase
        for op in body.operations:
            match op.name:
                case "if_else":
                    exit_kind = block(op.blocks[0 if guard(op.condition) else 1])
                    if exit_kind:
                        return exit_kind
                case "while_loop":
                    if guard(op.condition):
                        exit_kind = block(op.blocks[0])
                        if exit_kind != "break_loop" and guard(op.condition):
                            raise Unresolved
                case "break_loop" | "continue_loop":
                    return op.name
                case "classical_set" | "index_error":
                    raise Unresolved
                case _:
                    if op.name == "measure":
                        unknown.update(op.clbits)
                    operations.append(op)
        return None

    try:
        block(circuit)
    except Unresolved:
        return circuit
    return replace(circuit, operations=tuple(operations), global_phase=phase)
