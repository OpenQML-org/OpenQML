"""Fold diagonal phases on equal affine parities within a control-flow block."""

from dataclasses import replace
from math import pi

from .compiler import _sum_angles
from .core import Operation
from .parameters import expression

_FIXED_PHASE = {"z": pi, "s": pi / 2, "sdg": -pi / 2, "t": pi / 4, "tdg": -pi / 4}
_PHASES = _FIXED_PHASE.keys() | {"p", "rz", "rzz"}


def fold_block(body, charge):
    """Track y = parity(input wires) XOR offset. Preserve each block's full phase."""
    wires, groups, output, phases = {}, {}, list(body.operations), [body.global_phase]
    folded = 0

    def wire(q):
        return wires.get(q, (frozenset((q,)), False))

    def parity(a, b):
        left, flip_left = wire(a)
        right, flip_right = wire(b)
        charge(len(left) + len(right))
        return left ^ right, flip_left ^ flip_right

    def finish():
        for index, _, _, angles in groups.values():
            if len(angles) > 1:
                first = output[index]
                name = first.name if first.name in {"rz", "rzz"} else "p"
                angle = _sum_angles(angles)
                output[index] = (
                    Operation(name, first.qubits, (angle,)) if angle != expression(0) else None
                )
        groups.clear()
        wires.clear()

    for i, op in enumerate(body.operations):
        charge(1)
        if op.name == "x":
            q = op.qubits[0]
            word, flip = wire(q)
            wires[q] = word, not flip
        elif op.name == "cx":
            wires[op.qubits[1]] = parity(*op.qubits)
        elif op.name == "swap":
            a, b = op.qubits
            wires[a], wires[b] = wire(b), wire(a)
        elif op.name in _PHASES:
            word, flip = parity(*op.qubits) if op.name == "rzz" else wire(op.qubits[0])
            charge(len(word))
            angle = op.params[0] if op.params else expression(_FIXED_PHASE[op.name])
            # p(theta) contributes theta*y, rz/rzz contribute theta*(y-1/2).
            offset = int(flip) - (0.5 if op.name in {"rz", "rzz"} else 0.0)
            if word not in groups:
                groups[word] = (i, flip, offset, [angle])
                continue
            _, first_flip, first_offset, angles = groups[word]
            sign = -1 if flip != first_flip else 1
            angles.append(angle if sign == 1 else -angle)
            correction = offset - sign * first_offset
            if correction:
                phases.append(correction * angle)
            output[i] = None
            folded += 1
        elif op.name not in {"id", "cz", "cp", "crz"}:
            # These diagonal gates leave all parities intact. Other gates,
            # barriers and classical effects delimit independent regions.
            finish()
    finish()
    if not folded:
        return body, 0
    return replace(
        body,
        operations=tuple(op for op in output if op is not None),
        global_phase=_sum_angles(phases),
    ), folded
