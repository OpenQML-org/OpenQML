"""Cirq interoperability with explicit global phases and measurement-key conventions."""

import math
import re
from collections.abc import Mapping, Sequence

import numpy as np

from ..errors import ConversionError, conversion_error
from ..ir import CHANNEL_GATES, Circuit, Operation
from ..parameters import Expression
from ._common import bound_for_export, dependency, sympy_expression, to_sympy


def from_cirq(
    native,
    *,
    qubit_order=None,
    measurement_map=None,
    max_matrix_qubits=10,
    max_operations=100_000,
    max_condition_bits=10,
) -> Circuit:
    from ..ir import integer
    from ._common import check_matrix_width

    integer(max_matrix_qubits, "max_matrix_qubits", 1)
    integer(max_operations, "max_operations")
    integer(max_condition_bits, "max_condition_bits")
    # Bound both truth-table width and recursive IR depth, even if a caller
    # supplies an enormous operation budget.
    if max_condition_bits > 16:
        raise conversion_error("CONDITION_RESOURCE_LIMIT", "max_condition_bits cannot exceed 16.")
    if measurement_map is not None and not isinstance(measurement_map, Mapping):
        raise conversion_error("MEASUREMENT_MAP", "measurement_map must map keys to ordered bits.")
    key_map = {}
    for key, record in (measurement_map or {}).items():
        if (
            not isinstance(key, str)
            or isinstance(record, (str, bytes))
            or not isinstance(record, Sequence)
        ):
            raise conversion_error("MEASUREMENT_MAP", "Use string keys and ordered bit sequences.")
        try:
            record = tuple(integer(bit, "measurement bit") for bit in record)
        except (TypeError, ValueError) as exc:
            raise conversion_error("MEASUREMENT_MAP", "Invalid measurement bit sequence.") from exc
        if not record or len(set(record)) != len(record):
            raise conversion_error(
                "MEASUREMENT_MAP", "Measurement bits must be nonempty and unique."
            )
        key_map[key] = record
    cirq = dependency("cirq", "cirq")
    if not isinstance(native, cirq.AbstractCircuit):
        raise conversion_error("SOURCE_TYPE", "Expected a Cirq Circuit/FrozenCircuit.")
    order = tuple(
        cirq.QubitOrder.DEFAULT.order_for(native.all_qubits())
        if qubit_order is None
        else qubit_order
    )
    if (
        not order
        or len(set(order)) != len(order)
        or not native.all_qubits() <= set(order)
        or any(q.dimension != 2 for q in order)
    ):
        raise conversion_error("WIRE_ORDER", "Provide unique qubits of dimension 2 in qubit_order.")
    mapping = {q: i for i, q in enumerate(order)}
    fixed = [
        (cirq.I, "id"),
        (cirq.X, "x"),
        (cirq.Y, "y"),
        (cirq.Z, "z"),
        (cirq.H, "h"),
        (cirq.S, "s"),
        (cirq.S**-1, "sdg"),
        (cirq.T, "t"),
        (cirq.T**-1, "tdg"),
        (cirq.X**0.5, "sx"),
        (cirq.X**-0.5, "sxdg"),
        (cirq.CNOT, "cx"),
        (cirq.CZ, "cz"),
        (cirq.SWAP, "swap"),
        (cirq.TOFFOLI, "ccx"),
        (cirq.FREDKIN, "cswap"),
        (cirq.ISWAP, "iswap"),
        (cirq.H.controlled(), "ch"),
    ]
    rotations = [
        (cirq.XPowGate, "rx"),
        (cirq.YPowGate, "ry"),
        (cirq.ZPowGate, "rz"),
        (cirq.XXPowGate, "rxx"),
        (cirq.YYPowGate, "ryy"),
        (cirq.ZZPowGate, "rzz"),
    ]
    operations, diagnostics, keys, written_bits = [], [], {}, set()
    phase, num_clbits, work = Expression(), 0, 0
    for index, operation in enumerate(native.all_operations()):
        location = f"operations[{index}]"
        work += 1
        if work > max_operations:
            raise conversion_error(
                "EXPANSION_LIMIT", "Cirq import exceeds max_operations.", location
            )
        gate = operation.gate
        if getattr(operation, "tags", ()):
            diagnostics.extend(
                conversion_error(
                    "OPERATION_TAGS", "Cirq operation tags cannot be preserved by IR.", location
                ).diagnostics
            )
            continue
        qubits = tuple(mapping[q] for q in operation.qubits)
        try:
            if isinstance(operation, cirq.ClassicallyControlledOperation):
                from ._cirq_control import lower_controls

                def body(operation=operation, available=max_operations - work + 1):
                    return from_cirq(
                        cirq.Circuit(operation.without_classical_controls()),
                        qubit_order=order,
                        max_matrix_qubits=max_matrix_qubits,
                        max_operations=available,
                        max_condition_bits=max_condition_bits,
                    )

                lowered, used = lower_controls(
                    operation,
                    body,
                    keys,
                    max_bits=max_condition_bits,
                    max_operations=max_operations - work + 1,
                    location=location,
                )
                operations.extend(lowered.operations)
                phase += lowered.global_phase
                num_clbits = max(num_clbits, lowered.num_clbits)
                work += max(1, used) - 1
            elif isinstance(gate, cirq.WaitGate):
                raise conversion_error(
                    "TIMING_UNSUPPORTED", "IR cannot retain this wait duration.", location
                )
            elif isinstance(gate, cirq.GlobalPhaseGate):
                coefficient = gate.coefficient
                if cirq.is_parameterized(gate):
                    sympy = dependency("sympy", "cirq")
                    if getattr(coefficient, "func", None) != sympy.exp:
                        raise conversion_error(
                            "GLOBAL_PHASE", "Unsupported symbolic phase.", location
                        )
                    phase += sympy_expression(coefficient.args[0] / sympy.I, location)
                else:
                    phase += float(np.angle(complex(coefficient)))
            elif isinstance(gate, cirq.ResetChannel):
                operations.append(Operation("reset", qubits))
            elif len(qubits) == 1 and type(gate) in {
                cirq.AmplitudeDampingChannel,
                cirq.PhaseDampingChannel,
                cirq.DepolarizingChannel,
                cirq.BitFlipChannel,
                cirq.PhaseFlipChannel,
            }:
                kinds = {
                    cirq.AmplitudeDampingChannel: ("amplitude_damping", "gamma"),
                    cirq.PhaseDampingChannel: ("phase_damping", "gamma"),
                    cirq.DepolarizingChannel: ("depolarizing", "p"),
                    cirq.BitFlipChannel: ("bit_flip", "p"),
                    cirq.PhaseFlipChannel: ("phase_flip", "p"),
                }
                kind, attribute = kinds[type(gate)]
                operations.append(Operation(kind, qubits, (getattr(gate, attribute),)))
            elif isinstance(gate, cirq.MeasurementGate):
                if any(gate.invert_mask) or getattr(gate, "confusion_map", {}):
                    raise conversion_error(
                        "MEASUREMENT_TRANSFORM",
                        "Inverted or confused measurements are unsupported.",
                        location,
                    )
                key = cirq.measurement_key_name(operation)
                if key in keys:
                    raise conversion_error(
                        "REPEATED_MEASUREMENT_KEY",
                        "Repeated Cirq keys cannot preserve IR bit writes.",
                        location,
                    )
                if key in key_map:
                    targets = key_map[key]
                elif re.fullmatch(r"c(?:0|[1-9][0-9]*)", key) and len(qubits) == 1:
                    targets = (int(key[1:]),)
                else:
                    raise conversion_error(
                        "MEASUREMENT_KEY",
                        "Use keys c0, c1, ... or an explicit measurement_map in qubit order.",
                        location,
                    )
                if len(targets) != len(qubits) or written_bits.intersection(targets):
                    raise conversion_error(
                        "MEASUREMENT_MAP",
                        "Keys need disjoint bits matching measurement arity.",
                        location,
                    )
                work += len(targets) - 1
                if work > max_operations:
                    raise conversion_error(
                        "EXPANSION_LIMIT", "Measurement expansion exceeds budget.", location
                    )
                keys[key] = targets
                written_bits.update(targets)
                operations.extend(
                    Operation("measure", (q,), clbits=(bit,))
                    for q, bit in zip(qubits, targets, strict=True)
                )
                num_clbits = max(num_clbits, max(targets) + 1)
            else:
                name = next((name for candidate, name in fixed if gate == candidate), None)
                if name:
                    operations.append(Operation(name, qubits))
                    continue
                match = next(
                    ((kind, name) for kind, name in rotations if isinstance(gate, kind)), None
                )
                if match:
                    angle = sympy_expression(gate.exponent * math.pi, location)
                    operations.append(Operation(match[1], qubits, (angle,)))
                    phase += angle * (float(gate.global_shift) + 0.5)
                elif gate == cirq.ControlledGate(cirq.Y):
                    operations.append(Operation("cy", qubits))
                elif isinstance(gate, cirq.CZPowGate):
                    angle = sympy_expression(gate.exponent * math.pi, location)
                    operations.append(Operation("cp", qubits, (angle,)))
                    phase += angle * float(gate.global_shift)
                elif (
                    isinstance(gate, cirq.ControlledGate)
                    and gate.num_controls() == 1
                    and gate.control_values == cirq.ProductOfSums(((1,),))
                ):
                    sub = gate.sub_gate
                    axis = next(
                        (axis for kind, axis in rotations[:3] if isinstance(sub, kind)), None
                    )
                    if axis is None:
                        check_matrix_width(len(qubits), max_matrix_qubits, location)
                        if not cirq.is_parameterized(operation) and cirq.has_unitary(operation):
                            operations.append(
                                Operation("unitary", qubits, matrix=cirq.unitary(operation))
                            )
                            continue
                        raise conversion_error(
                            "UNSUPPORTED_OPERATION",
                            "Unsupported controlled Cirq sub-gate.",
                            location,
                        )
                    angle = sympy_expression(sub.exponent * math.pi, location)
                    operations.append(Operation("c" + axis, qubits, (angle,)))
                    # A sub-gate phase becomes a phase on the control, not a global phase.
                    correction = angle * (float(sub.global_shift) + 0.5)
                    if correction != Expression():
                        operations.append(Operation("p", (qubits[0],), (correction,)))
                elif not cirq.is_parameterized(operation):
                    check_matrix_width(len(qubits), max_matrix_qubits, location)
                    if cirq.has_unitary(operation):
                        operations.append(
                            Operation("unitary", qubits, matrix=cirq.unitary(operation))
                        )
                    elif cirq.has_kraus(operation):
                        if cirq.measurement_key_names(operation):
                            raise conversion_error(
                                "CHANNEL_RECORD_UNSUPPORTED",
                                "Recorded Kraus indices need an instrument representation.",
                                location,
                            )
                        operations.append(Operation("kraus", qubits, kraus=cirq.kraus(operation)))
                    else:
                        raise conversion_error(
                            "UNSUPPORTED_OPERATION",
                            f"Cannot represent Cirq operation {operation!s}.",
                            location,
                        )
                else:
                    raise conversion_error(
                        "UNSUPPORTED_OPERATION",
                        f"Cannot represent Cirq operation {operation!s}.",
                        location,
                    )
        except ConversionError as exc:
            if any(d.code.endswith("_LIMIT") for d in exc.diagnostics):
                raise
            diagnostics.extend(exc.diagnostics)
    if diagnostics:
        raise ConversionError(diagnostics)
    if set(key_map) - set(keys):
        raise conversion_error("MEASUREMENT_MAP", "measurement_map contains unused keys.")
    return Circuit(len(order), tuple(operations), num_clbits, phase)


def to_cirq(circuit: Circuit, bindings=None, *, max_operations=100_000, max_depth=64):
    from ..compiler import _check_circuit_limits, decompose

    _check_circuit_limits(circuit, max_operations=max_operations, max_depth=max_depth)
    cirq = dependency("cirq", "cirq")
    circuit = bound_for_export(circuit, bindings)
    from ..conversion import diagnose

    diagnostics = diagnose(circuit, "cirq", max_operations=max_operations, max_depth=max_depth)
    if diagnostics:
        raise ConversionError(diagnostics)
    circuit = decompose(circuit, max_operations=max_operations, max_depth=max_depth)
    qubits = cirq.LineQubit.range(circuit.num_qubits)
    fixed = {
        "id": cirq.I,
        "x": cirq.X,
        "y": cirq.Y,
        "z": cirq.Z,
        "h": cirq.H,
        "s": cirq.S,
        "sdg": cirq.S**-1,
        "t": cirq.T,
        "tdg": cirq.T**-1,
        "sx": cirq.X**0.5,
        "cx": cirq.CNOT,
        "cy": cirq.ControlledGate(cirq.Y),
        "cz": cirq.CZ,
        "swap": cirq.SWAP,
        "ccx": cirq.TOFFOLI,
    }
    native = cirq.Circuit()
    used = {q for _, op in circuit.walk_operations() for q in op.qubits}
    # Declare otherwise idle wires before measurement, so roundtrips stay terminal.
    for q in range(circuit.num_qubits):
        if q not in used:
            native.append(cirq.I(qubits[q]), strategy=cirq.InsertStrategy.NEW)
    written = set()

    def append(operation, controls):
        native.append(
            operation.with_classical_controls(*controls) if controls else operation,
            strategy=cirq.InsertStrategy.NEW,
        )

    def emit(block, controls=()):
        for op in block.operations:
            targets = [qubits[q] for q in op.qubits]
            if op.name == "if_else":
                sympy = dependency("sympy", "cirq")
                # Unwritten IR storage is zero; Cirq raises on missing keys.
                # Fold these constants rather than creating phantom measurements.
                expected = [
                    (bit, (op.condition.value >> i) & 1) for i, bit in enumerate(op.condition.bits)
                ]
                if any(bit not in written and value for bit, value in expected):
                    emit(op.blocks[1], controls)
                else:
                    terms = [
                        sympy.Eq(sympy.Symbol(f"c{bit}"), value)
                        for bit, value in expected
                        if bit in written
                    ]
                    if not terms:
                        emit(op.blocks[0], controls)
                    else:
                        predicate = sympy.And(*terms)
                        emit(op.blocks[0], controls + (predicate,))
                        emit(op.blocks[1], controls + (sympy.Not(predicate),))
                continue
            if op.name == "reset":
                gate = cirq.reset(targets[0])
            elif op.name in CHANNEL_GATES:
                from ..density import channel_kraus

                gate = cirq.KrausChannel(channel_kraus(op), validate=True).on(*targets)
            elif op.name == "measure":
                gate = cirq.measure(*targets, key=f"c{op.clbits[0]}")
                written.add(op.clbits[0])
            elif op.name == "unitary":
                gate = cirq.MatrixGate(np.asarray(op.matrix)).on(*targets)
            elif op.name in fixed:
                gate = fixed[op.name].on(*targets)
            else:
                expr = op.params[0]
                angle = to_sympy(expr) if expr.parameters else expr.constant
                if op.name in {"rx", "ry", "rz"}:
                    gate = getattr(cirq, op.name)(angle).on(*targets)
                elif op.name == "p":
                    gate = cirq.ZPowGate(exponent=angle / math.pi).on(*targets)
                else:
                    kind = {"rxx": cirq.XXPowGate, "ryy": cirq.YYPowGate, "rzz": cirq.ZZPowGate}[
                        op.name
                    ]
                    gate = kind(exponent=angle / math.pi, global_shift=-0.5).on(*targets)
            append(gate, controls)
        phase = block.global_phase
        if phase.parameters:
            sympy = dependency("sympy", "cirq")
            coefficient = sympy.exp(sympy.I * to_sympy(phase))
        else:
            coefficient = np.exp(1j * phase.constant)
        if phase.parameters or phase.constant:
            append(cirq.global_phase_operation(coefficient), controls)

    emit(circuit)
    return native
