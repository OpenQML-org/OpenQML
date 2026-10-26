"""Qiskit QuantumCircuit <-> IR, preserving wire indices, angles and global phase."""

from ..errors import ConversionError, ValidationError, conversion_error
from ..ir import CHANNEL_GATES, Circuit, ClassicalCondition, Operation
from ..parameters import Expression
from ._common import bound_for_export, dependency, numeric, reverse_local_basis, sympy_expression


def _angle(value, location):
    parameters = getattr(value, "parameters", set())
    if not parameters:
        return Expression(constant=numeric(value, location))
    return sympy_expression(value.sympify(), location)


def from_qiskit(
    native,
    *,
    expand=True,
    max_operations=100_000,
    max_depth=64,
    max_matrix_qubits=10,
    _loop_budget=None,
) -> Circuit:
    from ..ir import integer
    from ._common import check_matrix_width

    integer(max_operations, "max_operations")
    integer(max_depth, "max_depth")
    integer(max_matrix_qubits, "max_matrix_qubits", 1)
    qiskit = dependency("qiskit", "qiskit")
    library = dependency("qiskit.circuit.library", "qiskit")
    from qiskit.circuit import (
        Barrier,
        BreakLoopOp,
        ContinueLoopOp,
        Delay,
        ForLoopOp,
        IfElseOp,
        Instruction,
        Measure,
        Reset,
    )

    nested = _loop_budget is not None
    if _loop_budget is None:
        _loop_budget = {"remaining": max_operations}

    standard_classes = {
        getattr(library, name)
        for name in (
            "IGate",
            "XGate",
            "YGate",
            "ZGate",
            "HGate",
            "SGate",
            "SdgGate",
            "TGate",
            "TdgGate",
            "SXGate",
            "RXGate",
            "RYGate",
            "RZGate",
            "PhaseGate",
            "U1Gate",
            "CXGate",
            "CYGate",
            "CZGate",
            "SwapGate",
            "CCXGate",
            "RXXGate",
            "RYYGate",
            "RZZGate",
            "UGate",
            "U3Gate",
            "U2Gate",
            "SXdgGate",
            "CPhaseGate",
            "CU1Gate",
            "CRXGate",
            "CRYGate",
            "CRZGate",
            "CHGate",
            "CSwapGate",
            "iSwapGate",
        )
    } | {Barrier, Measure, Reset}
    if not isinstance(native, qiskit.QuantumCircuit):
        raise conversion_error("SOURCE_TYPE", "Expected a Qiskit QuantumCircuit.")
    if not native.num_qubits and not nested:
        raise conversion_error("NO_QUBITS", "A top-level IR circuit needs at least one qubit.")
    # Empty SDK bodies can carry only a phase/control structure. Their internal
    # placeholder has no quantum operands and is mapped onto a parent wire.
    width = max(1, native.num_qubits)
    operations, diagnostics = [], []
    used = 0

    def append(operation, location):
        nonlocal used
        cost = 1 + sum(1 for block in operation.blocks for _ in block.walk_operations())
        if used + cost > max_operations:
            raise conversion_error(
                "EXPANSION_LIMIT", "Expanded program exceeds max_operations.", location
            )
        operations.append(operation)
        used += cost

    phase = _angle(native.global_phase, "global_phase")
    from ..ir import GATES

    for index, item in enumerate(native.data):
        op, location = item.operation, f"operations[{index}]"
        if isinstance(op, (BreakLoopOp, ContinueLoopOp)):
            raise conversion_error(
                "CLASSICAL_CONTROL", "break/continue cannot be statically unrolled.", location
            )
        name = {"i": "id", "u1": "p", "u2": "u", "u3": "u", "cu1": "cp"}.get(op.name, op.name)
        if isinstance(op, IfElseOp):
            if used >= max_operations:
                raise conversion_error(
                    "EXPANSION_LIMIT", "Control node exceeds max_operations.", location
                )
            if max_depth == 0:
                raise conversion_error(
                    "EXPANSION_LIMIT", "Control blocks exceed max_depth.", location
                )
            condition = _condition(op.condition, native, location)
            blocks = []
            remaining = max_operations - used - 1
            for body in op.blocks:
                child = from_qiskit(
                    body,
                    expand=expand,
                    max_operations=remaining,
                    max_depth=max_depth - 1,
                    max_matrix_qubits=max_matrix_qubits,
                    _loop_budget=_loop_budget,
                )
                remaining -= sum(1 for _ in child.walk_operations())
                blocks.append(
                    Circuit(width, num_clbits=native.num_clbits).compose(
                        child,
                        qubits=tuple(native.find_bit(q).index for q in item.qubits)
                        if body.num_qubits
                        else (0,),
                        clbits=tuple(native.find_bit(c).index for c in item.clbits),
                    )
                )
            if len(blocks) == 1:
                blocks.append(Circuit(width))
            append(Operation("if_else", (), condition=condition, blocks=tuple(blocks)), location)
            continue
        if isinstance(op, ForLoopOp) and expand:
            from numbers import Integral

            from qiskit.circuit import Parameter

            indexset, variable, body = op.params
            if not isinstance(indexset, (range, tuple, list)):
                raise conversion_error(
                    "STATIC_LOOP_INDEX", "Use a finite integer index set.", location
                )
            try:
                count = len(indexset)
            except OverflowError as exc:
                raise conversion_error(
                    "EXPANSION_LIMIT", "Loop index set exceeds the budget.", location
                ) from exc
            if count == 0:
                continue
            if count > _loop_budget["remaining"] or max_depth == 0:
                raise conversion_error(
                    "EXPANSION_LIMIT", "Static loop exceeds the expansion budget/depth.", location
                )
            if variable is not None and not isinstance(variable, Parameter):
                raise conversion_error(
                    "CLASSICAL_LOOP_VARIABLE",
                    "Runtime loop variables need native execution.",
                    location,
                )
            if any(
                isinstance(value, bool) or not isinstance(value, Integral) for value in indexset
            ):
                raise conversion_error(
                    "STATIC_LOOP_INDEX", "Loop indices must be integers.", location
                )
            # Import one symbolic body, then bind/map immutable IR values. The
            # shared work counter also bounds nested phase-only/empty loops.
            before = _loop_budget["remaining"]
            child = from_qiskit(
                body,
                expand=expand,
                max_operations=(max_operations - used) // count,
                max_depth=max_depth - 1,
                max_matrix_qubits=max_matrix_qubits,
                _loop_budget=_loop_budget,
            )
            nested_work = before - _loop_budget["remaining"]
            extra = count + (count - 1) * nested_work
            if extra > _loop_budget["remaining"]:
                raise conversion_error(
                    "EXPANSION_LIMIT", "Nested loop work exceeds max_operations.", location
                )
            _loop_budget["remaining"] -= extra
            mapped = Circuit(width, num_clbits=native.num_clbits).compose(
                child,
                qubits=tuple(native.find_bit(q).index for q in item.qubits)
                if body.num_qubits
                else (0,),
                clbits=tuple(native.find_bit(c).index for c in item.clbits),
            )
            phases = []
            for value in indexset:
                try:
                    instance = (
                        mapped.bind({variable.name: int(value)}, partial=True)
                        if variable is not None and variable.name in mapped.parameters
                        else mapped
                    )
                except ValidationError as exc:
                    raise conversion_error(
                        "STATIC_LOOP_BINDING",
                        f"Invalid expression at loop index {value}: {exc}",
                        location,
                    ) from exc
                for operation in instance.operations:
                    append(operation, location)
                phases.append(instance.global_phase)
            # Balanced addition avoids a recursion-depth-sized phase graph.
            while len(phases) > 1:
                phases = [
                    phases[i] + phases[i + 1] if i + 1 < len(phases) else phases[i]
                    for i in range(0, len(phases), 2)
                ]
            phase += phases[0]
            continue
        native_channel = (
            op.name == "kraus" and op.base_class is Instruction and op.definition is None
        ) or (
            op.name == "quantum_channel"
            and type(op).__module__.startswith("qiskit_aer.")
            and hasattr(op, "_quantum_error")
        )
        if native_channel:
            from qiskit.quantum_info import Kraus

            check_matrix_width(len(item.qubits), max_matrix_qubits, location)
            matrices = (
                op.params
                if op.name == "kraus"
                else Kraus(op._quantum_error.to_quantumchannel()).data
            )
            append(
                Operation(
                    "kraus",
                    tuple(native.find_bit(q).index for q in item.qubits),
                    kraus=tuple(reverse_local_basis(k) for k in matrices),
                ),
                location,
            )
            continue
        if (
            name not in GATES
            or op.base_class not in standard_classes
            or getattr(op, "condition", None) is not None
            or (item.clbits and name != "measure")
        ):
            try:
                if isinstance(op, Delay):
                    raise conversion_error(
                        "TIMING_UNSUPPORTED", "IR cannot retain this delay duration/unit.", location
                    )
                if getattr(op, "condition", None) is not None or hasattr(op, "blocks"):
                    raise conversion_error(
                        "CLASSICAL_CONTROL",
                        "Only equality if/else and finite static for loops lower to IR.",
                        location,
                    )
                if isinstance(op, library.UnitaryGate):
                    check_matrix_width(len(item.qubits), max_matrix_qubits, location)
                    append(
                        Operation(
                            "unitary",
                            tuple(native.find_bit(q).index for q in item.qubits),
                            matrix=reverse_local_basis(op.to_matrix()),
                        ),
                        location,
                    )
                    continue
                definition = getattr(op, "definition", None) if expand else None
                if definition is not None:
                    if max_depth == 0:
                        raise conversion_error(
                            "EXPANSION_LIMIT", "Custom gate expansion exceeds max_depth.", location
                        )
                    child = from_qiskit(
                        definition,
                        expand=expand,
                        max_operations=max_operations - used,
                        max_depth=max_depth - 1,
                        max_matrix_qubits=max_matrix_qubits,
                        _loop_budget=_loop_budget,
                    )
                    qmap = tuple(native.find_bit(q).index for q in item.qubits)
                    cmap = tuple(native.find_bit(c).index for c in item.clbits)
                    mapped = Circuit(width, num_clbits=native.num_clbits).compose(
                        child, qubits=qmap if definition.num_qubits else (0,), clbits=cmap
                    )
                    for v in mapped.operations:
                        append(v, location)
                    phase += child.global_phase
                elif expand and not getattr(op, "parameters", set()) and not item.clbits:
                    from qiskit.quantum_info import Operator

                    check_matrix_width(len(item.qubits), max_matrix_qubits, location)
                    value = reverse_local_basis(Operator(op).data)
                    append(
                        Operation(
                            "unitary",
                            tuple(native.find_bit(q).index for q in item.qubits),
                            matrix=value,
                        ),
                        location,
                    )
                else:
                    raise conversion_error(
                        "UNSUPPORTED_OPERATION", f"Cannot expand {op.name!r}.", location
                    )
                if len(operations) > max_operations:
                    raise conversion_error(
                        "EXPANSION_LIMIT", "Expanded program exceeds max_operations.", location
                    )
                continue
            except ConversionError as exc:
                if any(d.code.endswith("_LIMIT") for d in exc.diagnostics):
                    raise
                diagnostics.extend(exc.diagnostics)
                continue
            except Exception as exc:
                diagnostics.extend(
                    conversion_error(
                        "UNSUPPORTED_OPERATION", f"Cannot import {op.name!r}: {exc}", location
                    ).diagnostics
                )
                continue
        try:
            params = tuple(_angle(p, location) for p in op.params)
            if op.name == "u2":
                from math import pi

                params = (Expression(constant=pi / 2),) + params
            append(
                Operation(
                    name,
                    tuple(native.find_bit(q).index for q in item.qubits),
                    params,
                    tuple(native.find_bit(c).index for c in item.clbits),
                ),
                location,
            )
        except ConversionError as exc:
            if any(d.code.endswith("_LIMIT") for d in exc.diagnostics):
                raise
            diagnostics.extend(exc.diagnostics)
        except ValidationError as exc:
            diagnostics.extend(
                conversion_error("INVALID_OPERATION", str(exc), location).diagnostics
            )
    if diagnostics:
        raise ConversionError(diagnostics)
    return Circuit(width, tuple(operations), native.num_clbits, phase)


def to_qiskit(circuit: Circuit, bindings=None, *, _symbols=None):
    qiskit = dependency("qiskit", "qiskit")
    from qiskit.circuit import Parameter

    circuit = bound_for_export(circuit, bindings)
    native = qiskit.QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
    symbols = (
        {name: Parameter(name) for name in circuit.parameters} if _symbols is None else _symbols
    )

    def angle(value):
        if value.op is None:
            return value.constant + sum(c * symbols[name] for name, c in value.terms)
        values = [angle(v) for v in value.args]
        if value.op == "add":
            return values[0] + values[1]
        if value.op == "mul":
            return values[0] * values[1]
        if value.op == "div":
            return values[0] / values[1]
        if value.op == "pow":
            return values[0] ** values[1]
        if value.op == "sqrt":
            return values[0] ** 0.5
        return getattr(values[0], value.op)()

    native.global_phase = angle(circuit.global_phase)
    for op in circuit.operations:
        if op.name == "if_else":
            from qiskit.circuit.classical import expr

            bits = op.condition.bits
            if len(bits) == 1:
                condition = (native.clbits[bits[0]], op.condition.value)
            elif bits == tuple(range(native.num_clbits)):
                condition = (native.cregs[0], op.condition.value)
            else:
                tests = [
                    expr.equal(native.clbits[bit], bool((op.condition.value >> i) & 1))
                    for i, bit in enumerate(bits)
                ]
                condition = tests[0]
                for test in tests[1:]:
                    condition = expr.logic_and(condition, test)
            bodies = []
            from dataclasses import replace

            for block in op.blocks:
                bodies.append(
                    to_qiskit(replace(block, num_clbits=circuit.num_clbits), _symbols=symbols)
                )
            native.if_else(condition, bodies[0], bodies[1], native.qubits, native.clbits)
        elif op.name in CHANNEL_GATES:
            if any(p.parameters for p in op.params):
                raise conversion_error(
                    "UNBOUND_CHANNEL", "Bind channel probabilities before Qiskit export."
                )
            from qiskit.quantum_info import Kraus

            from ..density import channel_kraus

            instruction = Kraus(
                [reverse_local_basis(k) for k in channel_kraus(op)]
            ).to_instruction()
            native.append(instruction, list(op.qubits))
        elif op.name == "unitary":
            native.unitary(reverse_local_basis(op.matrix), list(op.qubits))
        elif op.name == "measure":
            native.measure(op.qubits[0], op.clbits[0])
        elif op.name == "barrier":
            native.barrier(*op.qubits)
        else:
            getattr(native, op.name)(*(angle(p) for p in op.params), *op.qubits)
    return native


def _condition(value, native, location):
    from qiskit.circuit import Clbit

    if isinstance(value, tuple) and len(value) == 2:
        target, expected = value
        bits = (target,) if isinstance(target, Clbit) else tuple(target)
        return ClassicalCondition(tuple(native.find_bit(bit).index for bit in bits), int(expected))
    from qiskit.circuit.classical import expr

    assignments = {}

    def visit(node):
        if isinstance(node, expr.Binary) and node.op.name == "LOGIC_AND":
            visit(node.left)
            visit(node.right)
            return
        expected = 1
        if isinstance(node, expr.Binary) and node.op.name == "EQUAL":
            left, right = node.left, node.right
            if isinstance(left, expr.Value):
                left, right = right, left
            if not isinstance(right, expr.Value) or right.value not in {0, 1}:
                raise conversion_error(
                    "CLASSICAL_CONDITION", "Only bit equality conjunctions are supported.", location
                )
            node, expected = left, int(right.value)
        if not isinstance(node, expr.Var) or not isinstance(node.var, Clbit):
            raise conversion_error(
                "CLASSICAL_CONDITION", "Unsupported classical expression.", location
            )
        index = native.find_bit(node.var).index
        if index in assignments and assignments[index] != expected:
            raise conversion_error(
                "CLASSICAL_CONDITION", "Contradictory bit constraints.", location
            )
        assignments[index] = expected

    visit(value)
    bits = tuple(sorted(assignments))
    return ClassicalCondition(bits, sum(assignments[bit] << i for i, bit in enumerate(bits)))


def static_loop_notes(native):
    """Describe successful static control lowering without retaining SDK objects."""
    from qiskit.circuit import ForLoopOp

    from ..errors import Diagnostic

    notes = []
    stack = [(native, "")]
    while stack:
        body, prefix = stack.pop()
        for index, item in enumerate(body.data):
            op = item.operation
            path = prefix + f"operations[{index}]"
            if isinstance(op, ForLoopOp):
                notes.append(
                    Diagnostic(
                        "STATIC_LOOP_UNROLLED",
                        f"Finite loop: {len(op.params[0])} iterations lowered to IR operations.",
                        path,
                        severity="info",
                    )
                )
                if len(op.params[0]) == 0:
                    continue
            for i, block in enumerate(getattr(op, "blocks", ())):
                stack.append((block, path + f".blocks[{i}]."))
    return tuple(notes)
