"""Atomic switch lowering with grouped labels, default branches and bounded expansion."""

from ...errors import conversion_error
from ...ir.core import Circuit, ClassicalCondition, Operation


def lower_switch(
    op,
    native,
    item,
    *,
    width,
    location,
    expand,
    max_operations,
    max_depth,
    max_matrix_qubits,
    loop_budget,
    context,
):
    from numbers import Integral

    from qiskit.circuit import CASE_DEFAULT, ClassicalRegister, Clbit
    from qiskit.circuit.classical import expr

    from .qiskit import from_qiskit

    target = op.target
    if isinstance(target, expr.Var):
        target = target.var
    if not isinstance(target, (Clbit, ClassicalRegister)):
        raise conversion_error(
            "CLASSICAL_SWITCH_TARGET",
            "Switch targets must be a classical bit or register.",
            location,
        )
    bits = (target,) if isinstance(target, Clbit) else tuple(target)
    indices = tuple(native.find_bit(bit).index for bit in bits)
    spec = tuple(op.cases_specifier())
    count = sum(sum(v is not CASE_DEFAULT for v in labels) for labels, _ in spec)
    if count > max_depth or count > max_operations:
        raise conversion_error(
            "EXPANSION_LIMIT", "Switch guards exceed the depth/operation budget.", location
        )
    empty = Circuit(width, num_clbits=native.num_clbits)
    fallback, branches, used, seen = empty, [], count, set()
    for labels, body in spec:
        numeric = tuple(v for v in labels if v is not CASE_DEFAULT)
        for value in numeric:
            if (
                not isinstance(value, Integral)
                or not 0 <= value < (1 << len(bits))
                or value in seen
            ):
                raise conversion_error(
                    "CLASSICAL_SWITCH_LABEL",
                    "Switch labels must be distinct in-range integers.",
                    location,
                )
            seen.add(value)
        copies = len(numeric) + int(any(v is CASE_DEFAULT for v in labels))
        available = max_operations - used
        if copies == 0 or available < 0:
            raise conversion_error(
                "EXPANSION_LIMIT", "Switch body exceeds max_operations.", location
            )
        child = from_qiskit(
            body,
            expand=expand,
            max_operations=available // copies,
            max_depth=max_depth - count,
            max_matrix_qubits=max_matrix_qubits,
            _loop_budget=loop_budget,
            _context=context,
        )
        mapped = empty.compose(
            child,
            qubits=tuple(native.find_bit(q).index for q in item.qubits)
            if body.num_qubits
            else (0,),
            clbits=tuple(native.find_bit(c).index for c in item.clbits),
        )
        used += copies * sum(1 for _ in mapped.walk_operations())
        branches.extend((int(v), mapped) for v in numeric)
        if any(v is CASE_DEFAULT for v in labels):
            fallback = mapped
    # A selected body may overwrite the selector. Nested guards retain the
    # entry-time decision and prevent fall-through into another matching case.
    for value, body in reversed(branches):
        fallback = Circuit(
            width,
            (
                Operation(
                    "if_else",
                    (),
                    condition=ClassicalCondition(indices, value),
                    blocks=(body, fallback),
                ),
            ),
            native.num_clbits,
        )
    return fallback
