"""Lexical scope and control-flow threading for runtime programs."""

from ..errors import ResourceError, conversion_error
from .compiler import _check_circuit_limits


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
