"""Exact rotation fusion through commuting Pauli words and Clifford frames."""

from dataclasses import dataclass, replace

from ..errors import ResourceError, ValidationError
from ..sim.pauli import _CLIFFORD, _ROTATIONS, _clifford, _clifford_table
from . import integer
from .compiler import _check_circuit_limits, _sum_angles, circuit_stats, simplify
from .parameters import expression

_GADGETS = frozenset({"rx", "ry", "rz", "rxx", "ryy", "rzz"})


def _generator(op):
    axis = _ROTATIONS[op.name]
    mask = sum(1 << q for q in op.qubits)
    return (mask if axis != "Z" else 0), (mask if axis != "X" else 0)


@dataclass(frozen=True)
class PauliFusionResult:
    circuit: object
    before: object
    after: object
    fused_rotations: int
    conjugations: int
    work: int


def fuse_pauli_rotations(circuit, *, max_work=1_000_000, max_operations=100_000):
    """Fuse equal Pauli rotations across exact Clifford conjugations.

    Binary symplectic masks avoid matrices of size 2**n. An incoming rotation
    moves left only through commuting rotations or fixed Clifford gates. Its
    signed generator is tracked in the Clifford frame until a matching gate
    is found. Fusion preserves exact angles and uses the existing gate order.
    Barriers and unsupported gates stop the search.
    """
    integer(max_work, "max_work", 1)
    _check_circuit_limits(circuit, max_operations=max_operations)
    if circuit.has_effects:
        raise ValidationError("Pauli fusion requires a unitary circuit without effects.")
    output, work, fused, conjugations = [], 0, 0, 0
    for op in circuit.operations:
        if op.name not in _GADGETS:
            output.append(op)
            continue
        if op.params[0] == expression(0):
            continue
        x, z = _generator(op)
        sign, matched = 1, False
        for i in range(len(output) - 1, -1, -1):
            work += 1
            if work > max_work:
                raise ResourceError("Pauli fusion exceeds max_work.")
            previous = output[i]
            if previous is None:
                continue
            if previous.name in _GADGETS:
                gx, gz = _generator(previous)
                if (x, z) == (gx, gz):
                    angle = _sum_angles((previous.params[0], sign * op.params[0]))
                    output[i] = (
                        None if angle == expression(0) else replace(previous, params=(angle,))
                    )
                    fused, matched = fused + 1, True
                    break
                if ((x & gz).bit_count() + (z & gx).bit_count()) % 2:
                    break
            elif previous.name in _CLIFFORD:
                (x, z), factor = _clifford(x, z, previous.qubits, _clifford_table(previous.name))
                sign *= factor
                conjugations += 1
            else:
                break
        if not matched:
            output.append(op)
    result = simplify(replace(circuit, operations=tuple(op for op in output if op is not None)))
    return PauliFusionResult(
        result, circuit_stats(circuit), circuit_stats(result), fused, conjugations, work
    )
