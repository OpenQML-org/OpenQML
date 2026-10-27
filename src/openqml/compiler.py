"""Exact local circuit transformations with explicit phase accounting and limits."""

from dataclasses import dataclass, replace
from math import pi

from .errors import ConversionError, ValidationError, conversion_error
from .ir import GATES_V1, UNITARY_GATES, Circuit, Operation, integer
from .parameters import Expression, expression


def rename_parameters(circuit, names):
    from .parameters import Parameter

    mapping = {name: Parameter(value) for name, value in names.items()}
    active = set(circuit.parameters)
    return circuit.substitute({name: value for name, value in mapping.items() if name in active})


PORTABLE_BASIS = (GATES_V1 | {"unitary"}) - {"measure", "barrier"}
QASM_BASIS = PORTABLE_BASIS - {"sx", "swap", "rxx", "ryy", "rzz", "unitary"}
_SELF_INVERSE = frozenset("id x y z h cx cy cz swap ccx ch cswap".split())
_ADJOINT = {"s": "sdg", "sdg": "s", "t": "tdg", "tdg": "t", "sx": "sxdg", "sxdg": "sx"}
_ROTATIONS = frozenset("rx ry rz p rxx ryy rzz crx cry crz cp".split())


def _qasm_basis(circuit):
    # Multi-qubit matrices require SDK synthesis. Leave those matrices intact
    # until export; circuits containing only one-qubit matrices use NumPy.
    return QASM_BASIS | (
        {"unitary"}
        if any(op.name == "unitary" and len(op.qubits) > 1 for _, op in circuit.walk_operations())
        else set()
    )


def _rule(op):
    """Return chronological operations and an added global angle."""
    q, p = op.qubits, op.params

    def gate(name, *qubits, params=()):
        return Operation(name, tuple(qubits), tuple(params))

    if op.name == "unitary" and len(q) == 1:
        import numpy as np

        matrix = np.asarray(op.matrix)
        cosine, sine = abs(matrix[0, 0]), abs(matrix[1, 0])
        theta = 2 * np.arctan2(sine, cosine)
        # The diagonal and anti-diagonal cases have a free Euler angle. Use
        # a nonzero pivot so its phase is well-defined, including near poles.
        if sine == 0:
            phase = np.angle(matrix[0, 0])
            phi, lam = 0.0, np.angle(matrix[1, 1]) - phase
        elif cosine == 0:
            # With zero diagonal entries both off-diagonal phases fit directly
            # in U's free angles; do not invent an unnecessary global phase.
            phase = 0.0
            phi, lam = np.angle(matrix[1, 0]), np.angle(-matrix[0, 1])
        else:
            phase = np.angle(matrix[0, 0])
            phi = np.angle(matrix[1, 0]) - phase
            lam = np.angle(-matrix[0, 1]) - phase
        return (gate("u", *q, params=(theta, phi, lam)),), expression(float(phase))
    if op.name == "id":
        return (), Expression()
    if op.name == "u":
        theta, phi, lam = p
        return (
            gate("p", *q, params=(lam,)),
            gate("ry", *q, params=(theta,)),
            gate("p", *q, params=(phi,)),
        ), Expression()
    if op.name in {"x", "y", "h"}:
        angles = {"x": (pi, 0, pi), "y": (pi, pi / 2, pi / 2), "h": (pi / 2, 0, pi)}
        return (gate("u", *q, params=angles[op.name]),), Expression()
    if op.name in {"z", "s", "sdg", "t", "tdg"}:
        angle = {"z": pi, "s": pi / 2, "sdg": -pi / 2, "t": pi / 4, "tdg": -pi / 4}[op.name]
        return (gate("p", *q, params=(angle,)),), Expression()
    if op.name in {"sx", "sxdg"}:
        middle = "s" if op.name == "sx" else "sdg"
        return (gate("h", *q), gate(middle, *q), gate("h", *q)), Expression()
    if op.name == "rz":
        return (gate("p", *q, params=p),), -p[0] / 2
    if op.name == "p":
        return (gate("rz", *q, params=p),), p[0] / 2
    if op.name in {"rx", "ry"}:
        sign = 1 if op.name == "rx" else -1
        other = "ry" if op.name == "rx" else "rx"
        return (
            gate("p", *q, params=(sign * pi / 2,)),
            gate(other, *q, params=p),
            gate("p", *q, params=(-sign * pi / 2,)),
        ), Expression()
    if op.name == "swap":
        a, b = q
        return (gate("cx", a, b), gate("cx", b, a), gate("cx", a, b)), Expression()
    if op.name == "cx":
        a, b = q
        return (gate("h", b), gate("cz", a, b), gate("h", b)), Expression()
    if op.name == "cy":
        a, b = q
        return (gate("sdg", b), gate("cx", a, b), gate("s", b)), Expression()
    if op.name == "cz":
        a, b = q
        return (gate("h", b), gate("cx", a, b), gate("h", b)), Expression()
    if op.name == "ccx":
        a, b, t = q
        return (
            gate("h", t),
            gate("cx", b, t),
            gate("tdg", t),
            gate("cx", a, t),
            gate("t", t),
            gate("cx", b, t),
            gate("tdg", t),
            gate("cx", a, t),
            gate("t", b),
            gate("t", t),
            gate("h", t),
            gate("cx", a, b),
            gate("t", a),
            gate("tdg", b),
            gate("cx", a, b),
        ), Expression()
    if op.name in {"rxx", "ryy", "rzz"}:
        a, b = q
        center = (gate("cx", a, b), gate("rz", b, params=p), gate("cx", a, b))
        if op.name == "rzz":
            return center, Expression()
        if op.name == "rxx":
            outer = (gate("h", a), gate("h", b))
            return outer + center + outer, Expression()
        before = (gate("rx", a, params=(pi / 2,)), gate("rx", b, params=(pi / 2,)))
        after = (gate("rx", a, params=(-pi / 2,)), gate("rx", b, params=(-pi / 2,)))
        return before + center + after, Expression()
    if op.name in {"cry", "crz"}:
        a, b = q
        axis = op.name[1:]
        return (
            gate(axis, b, params=(p[0] / 2,)),
            gate("cx", a, b),
            gate(axis, b, params=(-p[0] / 2,)),
            gate("cx", a, b),
        ), Expression()
    if op.name == "crx":
        a, b = q
        return (gate("h", b), gate("crz", a, b, params=p), gate("h", b)), Expression()
    if op.name == "cp":
        a, b = q
        half = p[0] / 2
        return (
            gate("p", a, params=(half,)),
            gate("p", b, params=(half,)),
            gate("cx", a, b),
            gate("p", b, params=(-half,)),
            gate("cx", a, b),
        ), Expression()
    if op.name == "ch":
        a, b = q
        return (gate("cz", a, b), gate("cry", a, b, params=(pi / 2,))), Expression()
    if op.name == "cswap":
        c, a, b = q
        return (gate("cx", b, a), gate("ccx", c, a, b), gate("cx", b, a)), Expression()
    if op.name == "iswap":
        a, b = q
        return (gate("swap", a, b), gate("s", a), gate("s", b), gate("cz", a, b)), Expression()
    raise conversion_error("DECOMPOSITION_UNAVAILABLE", f"Cannot lower gate {op.name!r}.")


def _operation_location(path):
    return f"operations[{path[0]}]" + "".join(
        f".blocks[{block}].operations[{index}]"
        for block, index in zip(path[1::2], path[2::2], strict=True)
    )


def _check_circuit_limits(circuit, *, max_operations=100_000, max_depth=64):
    """Bound work before recursive transforms or parameter traversal start."""
    integer(max_operations, "max_operations", 1)
    integer(max_depth, "max_depth")
    if not isinstance(circuit, Circuit):
        raise ValidationError("Expected an OpenQML Circuit.")
    pending, count = [(circuit, (), 0)], 0
    while pending:
        body, prefix, depth = pending.pop()
        if depth > min(max_depth, 128):
            raise conversion_error(
                "CONTROL_DEPTH_LIMIT",
                "Circuit exceeds max_depth or the 128-level ceiling.",
                _operation_location(prefix[:-1]),
            )
        for index, op in enumerate(body.operations):
            path = prefix + (index,)
            count += 1
            if count > max_operations:
                raise conversion_error(
                    "DECOMPOSITION_LIMIT",
                    "Circuit exceeds max_operations.",
                    _operation_location(path),
                )
            pending.extend((block, path + (i,), depth + 1) for i, block in enumerate(op.blocks))


def decompose(
    circuit: Circuit, *, basis=PORTABLE_BASIS, max_operations=100_000, max_depth=64
) -> Circuit:
    """Lower to a requested gate set, preserving symbolic angles and full phase.

    Barriers and measurements are retained. A cyclic or unavailable decomposition
    fails explicitly. No approximation or phase-modulo simplification is applied.
    """
    basis = frozenset(basis)
    if not basis or basis - UNITARY_GATES:
        raise ValidationError(
            "basis must be a nonempty collection of supported unitary gate names."
        )
    _check_circuit_limits(circuit, max_operations=max_operations, max_depth=max_depth)
    count = 0

    def lower(body, prefix):
        output, phase = [], body.global_phase

        def emit(op, ancestors, location):
            nonlocal phase, count
            if op.name in basis or op.name not in UNITARY_GATES:
                count += 1
                if count > max_operations:
                    raise conversion_error(
                        "DECOMPOSITION_LIMIT", "Expanded circuit exceeds max_operations.", location
                    )
                if op.blocks:
                    op = replace(
                        op,
                        blocks=tuple(
                            lower(block, path + (i,)) for i, block in enumerate(op.blocks)
                        ),
                    )
                output.append(op)
                return
            if op.name in ancestors:
                raise conversion_error(
                    "DECOMPOSITION_UNAVAILABLE",
                    f"Gate {op.name!r} cannot be expressed in this basis.",
                    location,
                )
            try:
                children, delta = _rule(op)
            except ConversionError as exc:
                raise ConversionError(
                    tuple(replace(d, location=location) for d in exc.diagnostics)
                ) from exc
            phase += delta
            for child in children:
                emit(child, ancestors | {op.name}, location)

        for index, operation in enumerate(body.operations):
            path = prefix + (index,)
            emit(operation, frozenset(), _operation_location(path))
        return replace(body, operations=tuple(output), global_phase=phase)

    return lower(circuit, ())


def inverse(circuit: Circuit) -> Circuit:
    if circuit.has_effects:
        raise ValidationError(
            "Quantum channels, reset, measurement and classical control have no unitary inverse."
        )
    lowered = decompose(circuit)
    operations = []
    for op in reversed(lowered.operations):
        if op.name == "unitary":
            import numpy as np

            operations.append(replace(op, matrix=np.asarray(op.matrix).conj().T))
        elif op.name in _SELF_INVERSE or op.name == "barrier":
            operations.append(op)
        elif op.name in _ADJOINT:
            operations.append(replace(op, name=_ADJOINT[op.name]))
        elif op.name in _ROTATIONS:
            operations.append(replace(op, params=tuple(-p for p in op.params)))
        else:
            raise ValidationError(f"No inverse rule for {op.name!r}.")
    return replace(lowered, operations=tuple(operations), global_phase=-lowered.global_phase)


def simplify(circuit: Circuit) -> Circuit:
    """Cancel adjacent inverses and combine equal-axis rotations, without commuting gates.

    A rotation of 2*pi is retained because it may carry a global minus sign.
    There are no tolerance-based removals and no cancellation across barriers.
    """
    # Defer sums to avoid left-deep nonlinear expression graphs.
    output = []
    for op in circuit.operations:
        if op.blocks:
            op = replace(op, blocks=tuple(simplify(b) for b in op.blocks))
        if op.name == "id":
            continue
        if op.name in _ROTATIONS and op.params[0] == expression(0):
            continue
        if op.name in _ROTATIONS:
            if output and isinstance(output[-1], _RotationRun):
                run = output[-1]
                if run.operation.name == op.name and run.operation.qubits == op.qubits:
                    run.angles.append(op.params[0])
                    if len(run.angles) == 2 and all(a.is_affine for a in run.angles):
                        angle = run.angles[0] + run.angles[1]
                        if angle == expression(0):
                            output.pop()
                        else:
                            run.angles[:] = [angle]
                    continue
            output.append(_RotationRun(op, [op.params[0]]))
            continue
        if (
            output
            and not isinstance(output[-1], _RotationRun)
            and op.name not in {"barrier", "measure"}
        ):
            previous = output[-1]
            if previous.qubits == op.qubits:
                cancel = (op.name == previous.name and op.name in _SELF_INVERSE) or (
                    _ADJOINT.get(previous.name) == op.name
                )
                if cancel:
                    output.pop()
                    continue
        output.append(op)
    return replace(
        circuit,
        operations=tuple(
            replace(item.operation, params=(_sum_angles(item.angles),))
            if isinstance(item, _RotationRun)
            else item
            for item in output
        ),
    )


@dataclass
class _RotationRun:
    operation: Operation
    angles: list


def _sum_angles(angles):
    """Sum in a balanced graph; keep phase constants, without modulo reduction."""
    runs = []
    for angle in angles:
        if runs and angle == runs[-1][0]:
            runs[-1][1] += 1
        else:
            runs.append([angle, 1])
    # Compress adjacent duplicates without changing term order.
    values = [angle if count == 1 else count * angle for angle, count in runs]
    while len(values) > 1:
        values = [values[i] + values[i + 1] for i in range(0, len(values) - 1, 2)] + (
            values[-1:] if len(values) % 2 else []
        )
    return values[0] if values else Expression()


@dataclass(frozen=True)
class CircuitStats:
    num_qubits: int
    num_clbits: int
    num_parameters: int
    gates: int
    multi_qubit_gates: int
    measurements: int
    depth: int
    channels: int = 0
    resets: int = 0
    conditionals: int = 0


def circuit_stats(circuit: Circuit) -> CircuitStats:
    """Static counts include both branches; depth follows the longest dependency path.

    A classical decision adds one layer. Barriers synchronize without a gate
    layer. Sparse index maps avoid allocating storage for unused wire/register bits.
    """
    from .ir import CHANNEL_GATES

    counts = dict(gates=0, multi=0, measurements=0, channels=0, resets=0, conditionals=0)

    def walk(body, quantum, classical, floor=0):
        end = floor
        for op in body.operations:
            if op.name == "if_else":
                entry = max([floor] + [classical.get(c, 0) for c in op.condition.bits]) + 1
                counts["conditionals"] += 1
                branches = [walk(b, dict(quantum), dict(classical), entry) for b in op.blocks]
                for mapping, position in ((quantum, 0), (classical, 1)):
                    keys = set().union(*(b[position] for b in branches))
                    mapping.update({k: max(b[position].get(k, 0) for b in branches) for k in keys})
                for c in op.condition.bits:
                    classical[c] = max(classical.get(c, 0), entry)
                end = max(end, *(b[2] for b in branches))
                continue
            level = max(
                [floor]
                + [quantum.get(q, 0) for q in op.qubits]
                + [classical.get(c, 0) for c in op.clbits]
            )
            if op.name != "barrier":
                level += 1
                counts["measurements"] += op.name == "measure"
                counts["gates"] += op.name != "measure"
                counts["multi"] += op.name != "measure" and len(op.qubits) > 1
                counts["channels"] += op.name in CHANNEL_GATES
                counts["resets"] += op.name == "reset"
            for q in op.qubits:
                quantum[q] = level
            for c in op.clbits:
                classical[c] = level
            end = max(end, level)
        return quantum, classical, end

    _, _, depth = walk(circuit, {}, {})
    return CircuitStats(
        circuit.num_qubits,
        circuit.num_clbits,
        len(circuit.parameters),
        counts["gates"],
        counts["multi"],
        counts["measurements"],
        depth,
        counts["channels"],
        counts["resets"],
        counts["conditionals"],
    )
