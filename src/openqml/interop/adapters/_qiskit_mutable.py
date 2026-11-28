"""Linear continuation storage for bounded mutable loops and early exits."""

from numbers import Integral

from ...ir.core import Circuit, ClassicalCondition


def lower_mutable(lowering, native, qmap, cmap, *, depth=0, loop=None, loop_level=0, bounds=()):
    from qiskit import QuantumCircuit
    from qiskit.circuit import (
        CASE_DEFAULT,
        BreakLoopOp,
        Clbit,
        ContinueLoopOp,
        ForLoopOp,
        IfElseOp,
        Parameter,
        Store,
        SwitchCaseOp,
        WhileLoopOp,
    )
    from qiskit.circuit.classical import expr, types

    from ...errors import ConversionError
    from ._qiskit_flow import _Node
    from .qiskit import _angle, _condition, from_qiskit

    if depth > lowering.max_depth:
        lowering.fail("Mutable control exceeds max_depth.", "control_flow")
    width, bits = lowering.width, lowering.bits
    empty = Circuit(width, num_clbits=bits)
    operations, phase, cost, height = [], _angle(native.global_phase, "global_phase"), 0, 0

    def child(body, quantum, classical, *, child_loop=loop, child_level=loop_level, facts=None):
        return lower_mutable(
            lowering,
            body,
            quantum,
            classical,
            depth=depth + 1,
            loop=child_loop,
            loop_level=child_level,
            bounds=bounds if facts is None else facts,
        )

    def guarded(condition, true, false, location, facts=None):
        node = lowering.guard(
            condition,
            native,
            cmap,
            lowering.prefix(true, lowering.empty),
            lowering.prefix(false, lowering.empty),
            location,
            bounds if facts is None else facts,
        )
        return lowering.materialize(node)

    def arm(condition, truth, facts):
        if isinstance(condition, tuple):
            bit, value = condition
            bound = int(value) + 1 if truth else None
            left = expr.lift(bit)
        elif (
            isinstance(condition, expr.Binary)
            and isinstance(condition.left, expr.Var)
            and isinstance(condition.right, expr.Value)
        ):
            left, value = condition.left, int(condition.right.value)
            bound = (
                {"LESS": value, "LESS_EQUAL": value + 1, "EQUAL": value + 1}
                if truth
                else {"GREATER_EQUAL": value, "GREATER": value + 1, "NOT_EQUAL": value + 1}
            ).get(condition.op.name)
        else:
            return facts
        return (
            facts
            if bound is None
            else facts + ((lowering.storage.targets(left, native, cmap), bound),)
        )

    for index, item in enumerate(native.data):
        lowering.charge()
        op, location = item.operation, f"operations[{index}]"
        quantum = tuple(qmap[native.find_bit(q).index] for q in item.qubits)
        classical = tuple(cmap[native.find_bit(c).index] for c in item.clbits)
        if isinstance(op, Store):
            part = lowering.store(op, native, cmap, location, bounds)
        elif isinstance(op, (BreakLoopOp, ContinueLoopOp)):
            if loop is None:
                lowering.fail(
                    "Loop exits need an enclosing finite loop.", location, "CLASSICAL_CONTROL"
                )
            if loop == "while":
                part = empty.break_loop() if isinstance(op, BreakLoopOp) else empty.continue_loop()
            else:
                alive, running = loop
                part = empty.classical_set(running, 0)
                if isinstance(op, BreakLoopOp):
                    part = part.classical_set(alive, 0)
        elif isinstance(op, ForLoopOp):
            indexset, variable, body = op.params
            if not lowering.expand or not isinstance(indexset, (range, tuple, list)):
                lowering.fail(
                    "Mutable loops require finite integer expansion.", location, "STATIC_LOOP_INDEX"
                )
            if len(indexset) > lowering.limit:
                lowering.fail("Mutable loop exceeds max_operations.", location)
            if variable is not None and not isinstance(variable, Parameter):
                lowering.fail(
                    "Runtime loop variables require native execution.",
                    location,
                    "CLASSICAL_LOOP_VARIABLE",
                )
            lowering.charge(len(indexset))
            alive = lowering.storage.control_start + 2 * loop_level
            running = alive + 1
            part = empty.classical_set(alive, 1)
            loop_cost = 1
            for value in indexset:
                if isinstance(value, bool) or not isinstance(value, Integral):
                    lowering.fail("Loop indices must be integers.", location, "STATIC_LOOP_INDEX")
                bound = (
                    body.assign_parameters({variable: int(value)}, inplace=False)
                    if variable in body.parameters
                    else body
                )
                payload = child(
                    bound,
                    quantum,
                    classical,
                    child_loop=(alive, running),
                    child_level=loop_level + 1,
                    facts=(),
                )
                payload = empty.classical_set(running, 1).compose(payload)
                iteration = empty.if_else(ClassicalCondition((alive,), 1), payload)
                node = lowering.prefix(iteration, lowering.empty)
                loop_cost += node.cost
                lowering.checked(_Node(cost=loop_cost, depth=node.depth))
                part = part.compose(iteration)
        elif isinstance(op, IfElseOp):
            true = child(op.blocks[0], quantum, classical, facts=arm(op.condition, True, bounds))
            false = (
                child(op.blocks[1], quantum, classical, facts=arm(op.condition, False, bounds))
                if len(op.blocks) > 1
                else empty
            )
            part = guarded(op.condition, true, false, location)
        elif isinstance(op, SwitchCaseOp):
            target = op.target
            if isinstance(target, expr.Expr):
                if not isinstance(target.type, (types.Bool, types.Uint)):
                    lowering.fail(
                        "Switch selector needs Bool or Uint.", location, "CLASSICAL_SWITCH_TARGET"
                    )
                target_width = getattr(target.type, "width", 1)
            else:
                target_width = 1 if isinstance(target, Clbit) else len(target)
            fallback, cases, seen = empty, [], set()
            for labels, body in op.cases_specifier():
                payload = child(body, quantum, classical)
                for value in labels:
                    if value is CASE_DEFAULT:
                        fallback = payload
                    else:
                        if (
                            not isinstance(value, Integral)
                            or value in seen
                            or value < 0
                            or int(value).bit_length() > target_width
                        ):
                            lowering.fail(
                                "Invalid mutable switch label.", location, "CLASSICAL_SWITCH_LABEL"
                            )
                        seen.add(value)
                        cases.append((int(value), payload))
            for value, payload in reversed(cases):
                condition = (
                    expr.equal(
                        target, bool(value) if isinstance(target.type, types.Bool) else value
                    )
                    if isinstance(target, expr.Expr)
                    else (target, value)
                )
                fallback = guarded(condition, payload, fallback, location)
            part = fallback
        elif isinstance(op, WhileLoopOp):
            body = child(
                op.blocks[0],
                quantum,
                classical,
                child_loop="while",
                facts=arm(op.condition, True, ()),
            )
            try:
                cond = _condition(op.condition, native, location)
            except ConversionError:
                part = empty.while_loop(
                    None, guarded(op.condition, body, empty.break_loop(), location, facts=())
                )
            else:
                part = empty.while_loop(
                    ClassicalCondition(tuple(cmap[c] for c in cond.bits), cond.value), body
                )
        else:
            single = QuantumCircuit()
            single.add_bits(native.qubits)
            single.add_bits(native.clbits)
            single.append(op, item.qubits, item.clbits)
            local = from_qiskit(
                single,
                expand=lowering.expand,
                max_operations=lowering.limit,
                max_depth=lowering.max_depth - depth,
                max_matrix_qubits=lowering.matrix_limit,
                _loop_budget={"remaining": lowering.limit},
                _context=lowering.context,
            )
            part = empty.compose(local, qubits=qmap if native.num_qubits else (0,), clbits=cmap)
        if bounds:
            writes = {c for _, sub in part.walk_operations() for c in sub.clbits}
            bounds = tuple(f for f in bounds if writes.isdisjoint(f[0]))
        if isinstance(loop, tuple):
            part = empty.if_else(ClassicalCondition((loop[1],), 1), part)
        node = lowering.prefix(part, lowering.empty)
        cost, height = cost + node.cost, max(height, node.depth)
        lowering.checked(_Node(cost=cost, depth=height))
        operations.extend(part.operations)
        phase += part.global_phase
    return Circuit(width, tuple(operations), bits, global_phase=phase)
