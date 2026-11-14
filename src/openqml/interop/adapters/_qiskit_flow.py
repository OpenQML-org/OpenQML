"""Bounded control lowering with immutable continuations and mutable storage."""

from dataclasses import dataclass
from numbers import Integral

from ...errors import ConversionError, conversion_error
from ...ir.core import Circuit, ClassicalCondition, Operation
from ...ir.parameters import Expression


def has_jumps(native):
    from qiskit.circuit import BreakLoopOp, ContinueLoopOp, Store

    pending, seen = [native], {}
    while pending:
        body = pending.pop()
        if id(body) in seen:
            continue
        seen[id(body)] = body
        if body.num_vars:
            return True
        for item in body.data:
            if isinstance(item.operation, (BreakLoopOp, ContinueLoopOp, Store)):
                return True
            pending.extend(getattr(item.operation, "blocks", ()))
    return False


@dataclass(frozen=True, eq=False)
class _Node:
    prefix: object = None
    tail: object = None
    condition: object = None
    blocks: tuple = ()
    cost: int = 0
    depth: int = 0


class _Lowering:
    def __init__(
        self,
        native,
        expand,
        max_operations,
        max_depth,
        max_matrix_qubits,
        context,
        classical_inputs=None,
    ):
        from ._qiskit_storage import _Storage

        self.native, self.expand = native, expand
        self.limit, self.max_depth, self.matrix_limit = max_operations, max_depth, max_matrix_qubits
        self.storage = _Storage(native, max_operations, classical_inputs)
        self.width, self.bits = max(1, native.num_qubits), self.storage.bits
        self.context = context
        self.work = 0
        self.empty = _Node()

    def fail(self, message, location, code="EXPANSION_LIMIT"):
        raise conversion_error(code, message, location)

    def charge(self, amount=1):
        self.work += amount
        if self.work > 8 * self.limit:
            self.fail("Loop-control analysis exceeds max_operations work budget.", "control_flow")

    def checked(self, node):
        if node.cost > self.limit or node.depth > self.max_depth:
            self.fail("Expanded loop control exceeds max_operations/max_depth.", "control_flow")
        return node

    def prefix(self, circuit, tail):
        cost, depth = 0, 0
        pending = [(circuit, 0)]
        while pending:
            body, level = pending.pop()
            depth = max(depth, level)
            for operation in body.operations:
                cost += 1
                pending.extend((block, level + 1) for block in operation.blocks)
        if not cost and circuit.global_phase == Expression():
            return tail
        return self.checked(
            _Node(circuit, tail, cost=cost + tail.cost, depth=max(depth, tail.depth))
        )

    def branch(self, condition, high, low):
        if high is low:
            return high
        self.charge()
        return self.checked(
            _Node(
                condition=condition,
                blocks=(high, low),
                cost=1 + high.cost + low.cost,
                depth=1 + max(high.depth, low.depth),
            )
        )

    def guard(self, value, native, cmap, high, low, location):
        from ._classical_predicate import _BDD, _compile
        from .qiskit import _condition

        try:
            condition = _condition(value, native, location)
        except ConversionError:
            bdd = _BDD(self.limit, self.max_depth, location)
            root = _compile(
                value, native, bdd, classical_map=cmap, variable_bits=self.storage.variables
            )
            self.charge(bdd.work)
            table = {0: low, 1: high}
            pending, reachable = [root], set()
            while pending:
                index = pending.pop()
                if index < 2 or index in reachable:
                    continue
                reachable.add(index)
                pending.extend(bdd.nodes[index][1:])
            for index in sorted(reachable):
                node = bdd.nodes[index]
                bit, false, true = node
                table[index] = self.branch(ClassicalCondition((bit,), 1), table[true], table[false])
            return table[root]
        return self.branch(
            ClassicalCondition(tuple(cmap[bit] for bit in condition.bits), condition.value),
            high,
            low,
        )

    def store(self, operation, native, cmap, location):
        from ._classical_predicate import _BDD, _compile

        bdd = _BDD(self.limit, self.max_depth, location)
        roots = _compile(
            operation.rvalue,
            native,
            bdd,
            classical_map=cmap,
            variable_bits=self.storage.variables,
            return_bits=True,
        )
        targets = self.storage.targets(operation.lvalue, native, cmap)
        if len(roots) != len(targets):
            self.fail("Store widths differ.", location, "CLASSICAL_STORE_WIDTH")
        self.charge(bdd.work)

        def write(bit, value):
            body = Circuit(self.width, num_clbits=self.bits).classical_set(bit, value)
            return self.prefix(body, self.empty)

        def decision(root, bit):
            table, pending, seen = {0: write(bit, 0), 1: write(bit, 1)}, [root], set()
            while pending:
                index = pending.pop()
                if index < 2 or index in seen:
                    continue
                seen.add(index)
                pending.extend(bdd.nodes[index][1:])
            for index in sorted(seen):
                source, low, high = bdd.nodes[index]
                table[index] = self.branch(
                    ClassicalCondition((source,), 1), table[high], table[low]
                )
            return self.materialize(table[root])

        result = Circuit(self.width, num_clbits=self.bits)
        if all(root < 2 for root in roots):
            for bit, root in zip(targets, roots, strict=True):
                result = result.classical_set(bit, root)
            return result
        # Resolve every RHS bit before mutating any destination bit.
        for i, root in enumerate(roots):
            result = result.compose(decision(root, self.storage.scratch + i))
        for i, bit in enumerate(targets):
            result = result.if_else(
                ClassicalCondition((self.storage.scratch + i,), 1),
                self.materialize(write(bit, 1)),
                self.materialize(write(bit, 0)),
            )
        return result

    def block(self, native, qmap, cmap, normal, break_to=None, continue_to=None, depth=0):
        from qiskit.circuit import (
            CASE_DEFAULT,
            BreakLoopOp,
            ClassicalRegister,
            Clbit,
            ContinueLoopOp,
            ForLoopOp,
            IfElseOp,
            Parameter,
            SwitchCaseOp,
            WhileLoopOp,
        )
        from qiskit.circuit.classical import expr

        from .qiskit import from_qiskit

        if depth > self.max_depth:
            self.fail("Nested control blocks exceed max_depth.", "control_flow")
        current = normal
        for index in reversed(range(len(native.data))):
            self.charge()
            item = native.data[index]
            op, location = item.operation, f"operations[{index}]"
            quantum = tuple(qmap[native.find_bit(q).index] for q in item.qubits)
            classical = tuple(cmap[native.find_bit(c).index] for c in item.clbits)
            if isinstance(op, (BreakLoopOp, ContinueLoopOp)):
                target = break_to if isinstance(op, BreakLoopOp) else continue_to
                if target is None:
                    self.fail(
                        "Loop exits require an enclosing finite loop.",
                        location,
                        "CLASSICAL_CONTROL",
                    )
                current = target
            elif isinstance(op, ForLoopOp):
                indexset, variable, body = op.params
                if not self.expand:
                    self.fail("Finite loop exits require expansion.", location, "CLASSICAL_CONTROL")
                if not isinstance(indexset, (range, tuple, list)):
                    self.fail("Use a finite integer index set.", location, "STATIC_LOOP_INDEX")
                try:
                    count = len(indexset)
                except OverflowError as exc:
                    raise conversion_error(
                        "EXPANSION_LIMIT", "Loop index set exceeds budget.", location
                    ) from exc
                if count > self.limit:
                    self.fail("Static loop exceeds max_operations.", location)
                if variable is not None and not isinstance(variable, Parameter):
                    self.fail(
                        "Runtime loop variables require native execution.",
                        location,
                        "CLASSICAL_LOOP_VARIABLE",
                    )
                self.charge(count)
                after = current
                for value in reversed(indexset):
                    if isinstance(value, bool) or not isinstance(value, Integral):
                        self.fail("Loop indices must be integers.", location, "STATIC_LOOP_INDEX")
                    bound = (
                        body.assign_parameters({variable: int(value)}, inplace=False)
                        if variable in body.parameters
                        else body
                    )
                    current = self.block(
                        bound, quantum, classical, current, after, current, depth + 1
                    )
            elif isinstance(op, IfElseOp):
                true = self.block(
                    op.blocks[0], quantum, classical, current, break_to, continue_to, depth + 1
                )
                false = (
                    self.block(
                        op.blocks[1], quantum, classical, current, break_to, continue_to, depth + 1
                    )
                    if len(op.blocks) > 1
                    else current
                )
                current = self.guard(op.condition, native, cmap, true, false, location)
            elif isinstance(op, SwitchCaseOp):
                target = op.target
                if not isinstance(target, (Clbit, ClassicalRegister, expr.Var)):
                    self.fail(
                        "Switch targets must be bits or registers.",
                        location,
                        "CLASSICAL_SWITCH_TARGET",
                    )
                fallback, branches, seen = current, [], set()
                width = (
                    len(self.storage.targets(target, native, cmap))
                    if isinstance(target, expr.Var)
                    else 1
                    if isinstance(target, Clbit)
                    else len(target)
                )
                for labels, body in op.cases_specifier():
                    payload = self.block(
                        body, quantum, classical, current, break_to, continue_to, depth + 1
                    )
                    for value in labels:
                        if value is CASE_DEFAULT:
                            fallback = payload
                        else:
                            if (
                                not isinstance(value, Integral)
                                or value in seen
                                or not 0 <= value < 1 << width
                            ):
                                self.fail(
                                    "Switch labels must be distinct in-range integers.",
                                    location,
                                    "CLASSICAL_SWITCH_LABEL",
                                )
                            seen.add(value)
                            branches.append((int(value), payload))
                for value, payload in reversed(branches):
                    fallback = self.guard(
                        expr.equal(target, value)
                        if isinstance(target, expr.Var)
                        else (target, value),
                        native,
                        cmap,
                        payload,
                        fallback,
                        location,
                    )
                current = fallback
            elif isinstance(op, WhileLoopOp):
                self.fail(
                    "Runtime while loops require a native program route.",
                    location,
                    "CLASSICAL_CONTROL",
                )
            else:
                from qiskit import QuantumCircuit

                single = QuantumCircuit()
                single.add_bits(native.qubits)
                single.add_bits(native.clbits)
                single.global_phase = 0
                single.append(op, item.qubits, item.clbits)
                local = from_qiskit(
                    single,
                    expand=self.expand,
                    max_operations=self.limit,
                    max_depth=self.max_depth - depth,
                    max_matrix_qubits=self.matrix_limit,
                    _loop_budget={"remaining": self.limit},
                    _context=self.context,
                )
                mapped = Circuit(self.width, num_clbits=self.bits).compose(
                    local, qubits=qmap if native.num_qubits else (0,), clbits=cmap
                )
                current = self.prefix(mapped, current)
        from .qiskit import _angle

        phase = _angle(native.global_phase, "global_phase")
        return self.prefix(
            Circuit(self.width, num_clbits=self.bits).with_global_phase(phase), current
        )

    def materialize(self, root):
        cache, pending = {}, [(root, False)]
        while pending:
            node, ready = pending.pop()
            if node in cache:
                continue
            prefixes, end = [], node
            while end.prefix is not None:
                prefixes.append(end.prefix)
                end = end.tail
            if end.blocks and not ready:
                pending.append((node, True))
                pending.extend((child, False) for child in end.blocks if child not in cache)
                continue
            operations = [op for prefix in prefixes for op in prefix.operations]
            phases = [prefix.global_phase for prefix in prefixes]
            if end.blocks:
                operations.append(
                    Operation(
                        "if_else",
                        (),
                        condition=end.condition,
                        blocks=tuple(cache[child] for child in end.blocks),
                    )
                )
            while len(phases) > 1:
                phases = [
                    phases[i] + phases[i + 1] if i + 1 < len(phases) else phases[i]
                    for i in range(0, len(phases), 2)
                ]
            cache[node] = Circuit(
                self.width, tuple(operations), self.bits, global_phase=phases[0] if phases else 0
            )
        return cache[root]


def lower_jumps(
    native, *, expand, max_operations, max_depth, max_matrix_qubits, context, classical_inputs=None
):
    lowering = _Lowering(
        native, expand, max_operations, max_depth, max_matrix_qubits, context, classical_inputs
    )
    if lowering.storage.mutable:
        from ._qiskit_mutable import lower_mutable

        program = lower_mutable(
            lowering, native, tuple(range(native.num_qubits)), tuple(range(native.num_clbits))
        )
    else:
        root = lowering.block(
            native, tuple(range(native.num_qubits)), tuple(range(native.num_clbits)), lowering.empty
        )
        program = lowering.materialize(root)
    result = Circuit(lowering.width, num_clbits=lowering.bits)
    for bit, value in lowering.storage.initial:
        result = result.classical_set(bit, value)
    return result.compose(program)
