"""Immutable liveness schedules for native real-arithmetic readout graphs."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RegisterProgram:
    instructions: tuple
    roots: tuple
    registers: int


def lower_program(nodes, roots):
    """Allocate output storage before releasing inputs; never overwrite tensor data."""
    last = list(range(len(nodes)))
    for index, node in enumerate(nodes):
        if node[0] not in {"const", "param", "factor"}:
            for child in node[1:]:
                last[child] = index
    for root in roots:
        last[root] = len(nodes)
    locations, free, instructions, registers = [], [], [], 0
    for index, node in enumerate(nodes):
        if free:
            target = free.pop()
        else:
            target, registers = registers, registers + 1
        if node[0] in {"const", "param", "factor"}:
            operands, dead = node[1:], ()
        else:
            operands = tuple(locations[child] for child in node[1:])
            dead = tuple(
                locations[child] for child in dict.fromkeys(node[1:]) if last[child] == index
            )
        locations.append(target)
        if last[index] == index:
            dead += (target,)
        instructions.append((node[0], target, operands, dead))
        free.extend(dead)
    return RegisterProgram(tuple(instructions), tuple(locations[root] for root in roots), registers)
