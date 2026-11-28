"""PennyLane tape conversion; return measurements remain separate from IR storage writes."""

from ...errors import ConversionError, conversion_error
from ...ir.core import CHANNEL_GATES, Circuit, Operation
from ._common import bound_for_export, dependency, numeric

_NAMES = {
    "id": "Identity",
    "x": "PauliX",
    "y": "PauliY",
    "z": "PauliZ",
    "h": "Hadamard",
    "s": "S",
    "t": "T",
    "sx": "SX",
    "rx": "RX",
    "ry": "RY",
    "rz": "RZ",
    "p": "PhaseShift",
    "cx": "CNOT",
    "cy": "CY",
    "cz": "CZ",
    "swap": "SWAP",
    "ccx": "Toffoli",
    "rxx": "IsingXX",
    "ryy": "IsingYY",
    "rzz": "IsingZZ",
    "u": "U3",
    "cp": "ControlledPhaseShift",
    "crx": "CRX",
    "cry": "CRY",
    "crz": "CRZ",
    "ch": "CH",
    "cswap": "CSWAP",
    "iswap": "ISWAP",
    "amplitude_damping": "AmplitudeDamping",
    "phase_damping": "PhaseDamping",
    "depolarizing": "DepolarizingChannel",
    "bit_flip": "BitFlip",
    "phase_flip": "PhaseFlip",
}


def _wire_order(tape, requested):
    labels = tuple(tape.wires)
    if requested is None:
        requested = sorted(labels) if all(isinstance(q, int) for q in labels) else labels
    order = tuple(requested)
    if not order or len(set(order)) != len(order) or not set(labels) <= set(order):
        raise conversion_error("WIRE_ORDER", "wire_order must contain all used wires exactly once.")
    return order


def _expectation_readout(tape, *, description_limit, wire_limit, wire_order=None, **options):
    """Import analytic Pauli expectations in return order, including readout-only wires."""
    from numbers import Complex, Real

    from ...errors import ResourceError
    from ...ir.observables import Hamiltonian, PauliString

    qml = dependency("pennylane", "pennylane")
    if not isinstance(tape, qml.tape.QuantumScript):
        raise conversion_error("SOURCE_TYPE", "Expected a PennyLane QuantumScript/QuantumTape.")
    if tape.shots.total_shots is not None:
        raise conversion_error(
            "SHOTS_METADATA", "Certified tape expectations require analytic shots."
        )
    order = _wire_order(tape, wire_order)
    if len(order) > wire_limit or len(order) * len(tape.measurements) > description_limit:
        raise ResourceError("PennyLane readout descriptions exceed the configured budget.")
    mapping = {wire: i for i, wire in enumerate(order)}
    outputs, work = [], 0
    for index, measurement in enumerate(tape.measurements):
        location = f"measurements[{index}]"
        if type(measurement) is not qml.measurements.ExpectationMP or measurement.obs is None:
            raise conversion_error(
                "READOUT_UNSUPPORTED", "Expected an analytic Pauli expectation.", location
            )
        sentence = measurement.obs.pauli_rep
        if sentence is None:
            raise conversion_error(
                "READOUT_UNSUPPORTED", "The observable requires a Pauli expansion.", location
            )
        work += len(order) * max(1, len(sentence))
        if work > description_limit:
            raise ResourceError("PennyLane Pauli descriptions exceed the configured budget.")
        terms = []
        for word, coefficient in sentence.items():
            if (
                isinstance(coefficient, Complex)
                and not isinstance(coefficient, Real)
                and coefficient.imag == 0
            ):
                coefficient = coefficient.real
            paulis = ["I"] * len(order)
            for wire, pauli in word.items():
                paulis[mapping[wire]] = pauli
            terms.append(PauliString("".join(paulis), numeric(coefficient, location)))
        outputs.append(Hamiltonian(tuple(terms) or (PauliString("I" * len(order), 0.0),)))
    operations = qml.tape.QuantumScript(tape.operations)
    return from_pennylane(operations, wire_order=order, **options), tuple(outputs)


def from_pennylane(
    tape,
    *,
    wire_order=None,
    expand=True,
    max_depth=64,
    max_operations=100_000,
    max_matrix_qubits=10,
) -> Circuit:
    from ...ir.core import integer
    from ._common import check_matrix_width

    integer(max_operations, "max_operations")
    integer(max_depth, "max_depth")
    integer(max_matrix_qubits, "max_matrix_qubits", 1)
    qml = dependency("pennylane", "pennylane")
    if not isinstance(tape, qml.tape.QuantumScript):
        raise conversion_error("SOURCE_TYPE", "Expected a PennyLane QuantumScript/QuantumTape.")
    diagnostics, operations = [], []
    if tape.measurements:
        diagnostics.extend(
            conversion_error(
                "READOUT_UNSUPPORTED",
                "PennyLane return measurements are not IR classical bit writes.",
                "measurements",
                "Convert the operation-only tape; declare observables separately.",
            ).diagnostics
        )
    if tape.shots.total_shots is not None:
        diagnostics.extend(
            conversion_error(
                "SHOTS_METADATA",
                "IR does not store PennyLane shot configuration.",
                "shots",
                "Pass shots as an execution option.",
            ).diagnostics
        )
    wire_order = _wire_order(tape, wire_order)
    mapping = {q: i for i, q in enumerate(wire_order)}
    reverse = {v: k for k, v in _NAMES.items()}
    phase = 0.0
    for index, op in enumerate(tape.operations):
        location = f"operations[{index}]"
        name = reverse.get(op.name)
        if name and type(op) is not getattr(qml, _NAMES[name]):
            name = None
        if isinstance(op, qml.ops.op_math.Adjoint):
            name = {qml.S: "sdg", qml.T: "tdg", qml.SX: "sxdg"}.get(type(op.base))
        try:
            if type(op) is qml.GlobalPhase:
                # PennyLane GlobalPhase(phi) = exp(-i phi) I.
                phase -= numeric(op.data[0], location)
            elif name:
                operations.append(
                    Operation(
                        name,
                        tuple(mapping[q] for q in op.wires),
                        tuple(numeric(p, location) for p in op.data),
                    )
                )
            elif type(op) is qml.U2:
                from math import pi

                operations.append(
                    Operation(
                        "u",
                        tuple(mapping[q] for q in op.wires),
                        (pi / 2,) + tuple(numeric(p, location) for p in op.data),
                    )
                )
            elif type(op) is qml.Rot:
                wire = tuple(mapping[q] for q in op.wires)
                values = tuple(numeric(p, location) for p in op.data)
                operations.extend(
                    Operation(axis, wire, (angle,))
                    for axis, angle in zip(("rz", "ry", "rz"), values, strict=True)
                )
            elif isinstance(op, qml.operation.Channel):
                check_matrix_width(len(op.wires), max_matrix_qubits, location)
                if any(
                    getattr(p, "requires_grad", False) or "Tracer" in type(p).__name__
                    for p in op.data
                ):
                    raise conversion_error(
                        "AUTODIFF_GRAPH",
                        "Use NativePennyLaneLayer for channel training graphs.",
                        location,
                    )
                operations.append(
                    Operation(
                        "kraus", tuple(mapping[q] for q in op.wires), kraus=op.kraus_matrices()
                    )
                )
            elif type(op) is qml.QubitUnitary:
                check_matrix_width(len(op.wires), max_matrix_qubits, location)
                if any(getattr(value, "requires_grad", False) for value in op.data):
                    raise conversion_error(
                        "AUTODIFF_GRAPH",
                        "Use NativePennyLaneLayer for differentiable matrices.",
                        location,
                    )
                operations.append(
                    Operation("unitary", tuple(mapping[q] for q in op.wires), matrix=qml.matrix(op))
                )
            elif expand and getattr(op, "has_decomposition", False):
                if max_depth == 0:
                    raise conversion_error(
                        "EXPANSION_LIMIT", "Template expansion exceeds max_depth.", location
                    )
                if any(getattr(value, "requires_grad", False) for value in op.data):
                    raise conversion_error(
                        "AUTODIFF_GRAPH",
                        "Use NativePennyLaneLayer to preserve a QNode training graph.",
                        location,
                    )
                child = from_pennylane(
                    qml.tape.QuantumScript(op.decomposition(), []),
                    wire_order=wire_order,
                    expand=expand,
                    max_depth=max_depth - 1,
                    max_operations=max_operations - len(operations),
                    max_matrix_qubits=max_matrix_qubits,
                )
                operations.extend(child.operations)
                phase += child.global_phase.constant
            elif expand and getattr(op, "has_matrix", False):
                check_matrix_width(len(op.wires), max_matrix_qubits, location)
                if any(getattr(value, "requires_grad", False) for value in op.data):
                    raise conversion_error(
                        "AUTODIFF_GRAPH",
                        "Use NativePennyLaneLayer for differentiable matrix operations.",
                        location,
                    )
                operations.append(
                    Operation("unitary", tuple(mapping[q] for q in op.wires), matrix=qml.matrix(op))
                )
            else:
                raise conversion_error(
                    "UNSUPPORTED_OPERATION", f"Cannot represent PennyLane {op.name!r}.", location
                )
            if len(operations) > max_operations:
                raise conversion_error(
                    "EXPANSION_LIMIT", "Expanded tape exceeds max_operations.", location
                )
        except ConversionError as exc:
            if any(d.code.endswith("_LIMIT") for d in exc.diagnostics):
                raise
            diagnostics.extend(exc.diagnostics)
    if diagnostics:
        raise ConversionError(diagnostics)
    return Circuit(len(wire_order), tuple(operations), global_phase=phase)


def to_pennylane(circuit: Circuit, bindings=None, *, max_operations=100_000, max_depth=64):
    from ...ir._control import flatten_static_control
    from ...ir.compiler import _check_circuit_limits

    _check_circuit_limits(circuit, max_operations=max_operations, max_depth=max_depth)
    circuit = bound_for_export(circuit, bindings)
    circuit = flatten_static_control(circuit)
    qml = dependency("pennylane", "pennylane")
    from ..conversion import diagnose

    diagnostics = diagnose(circuit, "pennylane", max_operations=max_operations, max_depth=max_depth)
    if diagnostics:
        raise ConversionError(diagnostics)
    operations = []
    with qml.QueuingManager.stop_recording():
        used = set()
        for op in circuit.operations:
            used.update(op.qubits)
            if op.name == "reset" or op.name in CHANNEL_GATES:
                from ...sim.density import channel_kraus

                operations.append(qml.QubitChannel(channel_kraus(op), wires=op.qubits))
            elif op.name == "unitary":
                import numpy as np

                operations.append(qml.QubitUnitary(np.asarray(op.matrix), wires=op.qubits))
            elif op.name in {"sdg", "tdg", "sxdg"}:
                base = {"sdg": qml.S, "tdg": qml.T, "sxdg": qml.SX}[op.name]
                operations.append(qml.adjoint(base)(wires=op.qubits))
            else:
                operations.append(
                    getattr(qml, _NAMES[op.name])(*(p.constant for p in op.params), wires=op.qubits)
                )
        for q in range(circuit.num_qubits):
            if q not in used:
                operations.append(qml.Identity(wires=q))
        if circuit.global_phase.constant:
            operations.append(qml.GlobalPhase(-circuit.global_phase.constant))
    return qml.tape.QuantumScript(operations, [])
