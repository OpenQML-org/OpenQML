"""Bounded classical storage allocation and conservative definite-assignment checks."""

from collections.abc import Mapping
from numbers import Integral

from ...errors import conversion_error


def _variables(value, limit=100_000):
    from qiskit.circuit import ClassicalRegister, Clbit
    from qiskit.circuit.classical import expr

    if isinstance(value, tuple):
        value = value[0]
    if isinstance(value, (Clbit, ClassicalRegister)):
        return set()
    pending, seen, result = [value], {}, set()
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen[id(node)] = node
        if len(seen) > limit:
            raise conversion_error(
                "CLASSICAL_EXPRESSION_LIMIT", "Storage expression analysis exceeds its budget."
            )
        if isinstance(node, expr.Var):
            if not isinstance(node.var, (Clbit, ClassicalRegister)):
                result.add(node.var)
        elif isinstance(node, expr.Binary):
            pending.extend((node.left, node.right))
        elif isinstance(node, (expr.Cast, expr.Unary)):
            pending.append(node.operand)
        elif isinstance(node, getattr(expr, "Index", ())):
            pending.extend((node.target, node.index))
    return result


class _Storage:
    def __init__(self, native, limit, inputs=None):
        from qiskit.circuit import ForLoopOp, Store
        from qiskit.circuit.classical import expr, types

        self.bits, self.variables, self.names = native.num_clbits, {}, {}
        self.inputs, self.initial, self.scratch_width = {}, [], 0
        self.mutable, self.remaining, self.limit = False, 8 * limit, limit
        pending, seen = [native], {}
        while pending:
            body = pending.pop()
            if id(body) in seen:
                continue
            seen[id(body)] = body
            for variable in body.iter_vars():
                self.mutable = True
                if variable.var in self.variables:
                    continue
                kind = variable.type
                width = (
                    1
                    if isinstance(kind, types.Bool)
                    else (kind.width if isinstance(kind, types.Uint) else 0)
                )
                if width < 1:
                    self.fail("Only Bool/Uint runtime storage is portable.")
                if self.bits + width - native.num_clbits > limit:
                    self.fail(
                        "Runtime classical storage exceeds max_operations.", "EXPANSION_LIMIT"
                    )
                self.variables[variable.var] = tuple(range(self.bits, self.bits + width))
                self.names[variable.var] = variable.name
                self.bits += width
            for item in body.data:
                self.mutable |= isinstance(item.operation, Store)
                if isinstance(item.operation, Store) and not isinstance(
                    item.operation.rvalue, expr.Value
                ):
                    kind = item.operation.rvalue.type
                    width = (
                        1
                        if isinstance(kind, types.Bool)
                        else (kind.width if isinstance(kind, types.Uint) else 0)
                    )
                    if width < 1:
                        self.fail("Only Bool/Uint assignments are portable.")
                    self.scratch_width = max(self.scratch_width, width)
                pending.extend(getattr(item.operation, "blocks", ()))
        self.scratch = self.bits
        self.bits += self.scratch_width
        self.control_start, self.loop_depth = self.bits, 0
        if self.mutable:
            pending, seen = [(native, 0)], {}
            while pending:
                body, level = pending.pop()
                if (id(body), level) in seen:
                    continue
                seen[id(body), level] = body
                if level > 128 or len(seen) > limit:
                    self.fail("Loop storage analysis exceeds its budget.", "EXPANSION_LIMIT")
                self.loop_depth = max(self.loop_depth, level)
                for item in body.data:
                    child_level = level + int(isinstance(item.operation, ForLoopOp))
                    pending.extend((b, child_level) for b in getattr(item.operation, "blocks", ()))
            self.bits += 2 * self.loop_depth
        if self.bits - native.num_clbits > limit:
            self.fail("Assignment snapshots exceed max_operations.", "EXPANSION_LIMIT")
        if inputs is not None and not isinstance(inputs, Mapping):
            self.fail("classical_inputs must map input variable names to integer values.")
        supplied, used = dict(inputs or {}), set()
        external = tuple(native.iter_input_vars()) + tuple(native.iter_captured_vars())
        for variable in external:
            key = variable if variable in supplied else variable.name
            if key not in supplied:
                self.fail(
                    f"Bind classical input {variable.name!r} using classical_inputs.",
                    "CLASSICAL_INPUT_UNBOUND",
                )
            value = supplied[key]
            bits = self.variables[variable.var]
            if not isinstance(value, Integral) or not 0 <= value < 1 << len(bits):
                self.fail("Classical input values must fit their declared unsigned width.")
            used.add(key)
            self.inputs[variable.var] = int(value)
            self.initial.extend((bit, (int(value) >> i) & 1) for i, bit in enumerate(bits))
        if set(supplied) - used:
            self.fail("classical_inputs contains an undeclared input variable.")
        self._check(native, set(self.inputs), limit)

    def fail(self, message, code="CLASSICAL_STORAGE"):
        raise conversion_error(code, message, "classical_storage")

    def _check(self, native, initialized, limit, depth=0):
        from qiskit.circuit import (
            BreakLoopOp,
            ContinueLoopOp,
            ForLoopOp,
            IfElseOp,
            Store,
            SwitchCaseOp,
        )

        if depth > 128:
            self.fail("Storage analysis exceeds the nesting ceiling.", "EXPANSION_LIMIT")
        initialized = initialized - {v.var for v in native.iter_declared_vars()}

        def read(value):
            missing = _variables(value, self.limit) - initialized
            if missing:
                names = ", ".join(sorted(self.names[v] for v in missing))
                self.fail(f"Read before definite assignment: {names}.", "CLASSICAL_UNINITIALIZED")

        for item in native.data:
            self.remaining -= 1
            if self.remaining < 0:
                self.fail("Storage analysis exceeds max_operations.", "EXPANSION_LIMIT")
            op = item.operation
            if isinstance(op, Store):
                read(op.rvalue)
                from qiskit.circuit.classical import expr

                if isinstance(op.lvalue, expr.Var):
                    initialized.update(_variables(op.lvalue, self.limit))
                else:
                    read(op.lvalue)
            elif isinstance(op, IfElseOp):
                read(op.condition)
                branches = [self._check(b, initialized, limit, depth + 1) for b in op.blocks]
                if len(branches) == 1:
                    branches.append(initialized)
                initialized = set.intersection(*branches)
            elif isinstance(op, SwitchCaseOp):
                from qiskit.circuit import CASE_DEFAULT

                read(op.target)
                branches = [self._check(b, initialized, limit, depth + 1) for b in op.blocks]
                if not any(CASE_DEFAULT in labels for labels, _ in op.cases_specifier()):
                    branches.append(initialized)
                initialized = set.intersection(*branches)
            elif isinstance(op, ForLoopOp):
                if len(op.params[0]):
                    self._check(op.blocks[0], initialized, limit, depth + 1)
                # A loop may exit early. Keep only assignments known on entry.
            elif isinstance(op, (BreakLoopOp, ContinueLoopOp)):
                return initialized
        return initialized

    def targets(self, value, native, cmap):
        from qiskit.circuit import ClassicalRegister, Clbit
        from qiskit.circuit.classical import expr

        if isinstance(value, expr.Var):
            if isinstance(value.var, Clbit):
                return (cmap[native.find_bit(value.var).index],)
            if isinstance(value.var, ClassicalRegister):
                return tuple(cmap[native.find_bit(bit).index] for bit in value.var)
            return self.variables[value.var]
        if isinstance(value, getattr(expr, "Index", ())):
            if not isinstance(value.index, expr.Value) or not isinstance(
                value.index.value, Integral
            ):
                self.fail(
                    "Assignment targets require a statically addressed bit.",
                    "CLASSICAL_STORE_ADDRESS",
                )
            bits, index = self.targets(value.target, native, cmap), int(value.index.value)
            if not 0 <= index < len(bits):
                self.fail("Assignment target index exceeds its declared width.")
            return (bits[index],)
        self.fail("Unsupported classical assignment target.")
