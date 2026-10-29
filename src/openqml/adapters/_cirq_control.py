"""Bounded Cirq classical predicates lowered to atomic IR decisions.

Only documented key conditions and a small, explicit SymPy grammar are read.
No caller-defined Condition.resolve method or generated Python is evaluated.
"""

from dataclasses import replace
from itertools import product
from math import prod

from ..errors import conversion_error
from ..ir import Circuit, ClassicalCondition, Operation
from ._common import dependency


def lower_controls(operation, body, keys, *, max_bits, max_operations, max_depth, location):
    cirq = dependency("cirq", "cirq")
    sp = dependency("sympy", "cirq")
    expressions = []
    symbols = set()
    nodes = 0
    allowed = {
        sp.And,
        sp.Or,
        sp.Not,
        sp.Xor,
        sp.Equivalent,
        sp.Implies,
        sp.Eq,
        sp.Ne,
        sp.Lt,
        sp.Le,
        sp.Gt,
        sp.Ge,
        sp.Add,
        sp.Mul,
    }
    for condition in sorted(operation.classical_controls, key=str):
        if type(condition) is cirq.KeyCondition:
            if condition.index not in {-1, 0}:
                raise conversion_error(
                    "MEASUREMENT_HISTORY",
                    "Only the sole recorded key occurrence is supported.",
                    location,
                )
            expr = sp.Ne(sp.Symbol(str(condition.key)), 0)
        elif type(condition) is cirq.SympyCondition:
            expr = condition.expr
        else:
            raise conversion_error(
                "CLASSICAL_CONDITION", "Custom Cirq conditions cannot be converted.", location
            )
        pending = [expr]
        while pending:
            node = pending.pop()
            nodes += 1
            if nodes > 4096:
                raise conversion_error(
                    "CONDITION_RESOURCE_LIMIT",
                    "Classical predicate exceeds the node budget.",
                    location,
                )
            if type(node) is sp.Symbol:
                symbols.add(node)
            elif isinstance(node, sp.Integer) or node is sp.true or node is sp.false:
                pass
            elif node.func in allowed:
                pending.extend(node.args)
            else:
                raise conversion_error(
                    "CLASSICAL_EXPRESSION",
                    "Unsupported classical predicate expression.",
                    location,
                    "Use key truth, integer comparisons, And/Or/Not/Xor or integer sums/products.",
                )
        expressions.append(expr)
    missing = sorted(str(symbol) for symbol in symbols if str(symbol) not in keys)
    if missing:
        raise conversion_error(
            "CLASSICAL_KEY_UNDEFINED", f"Keys have not been measured: {missing}.", location
        )
    bits = tuple(sorted({bit for symbol in symbols for bit in keys[str(symbol)]}))
    if len(bits) > max_bits:
        raise conversion_error(
            "CONDITION_RESOURCE_LIMIT",
            f"Predicate uses {len(bits)} bits; max_condition_bits is {max_bits}.",
            location,
        )

    def evaluate(node, values):
        if type(node) is sp.Symbol:
            return values[node]
        if isinstance(node, sp.Integer):
            return int(node)
        if node is sp.true or node is sp.false:
            return node is sp.true
        args = [evaluate(arg, values) for arg in node.args]
        kind = node.func
        if kind is sp.And:
            return all(args)
        if kind is sp.Or:
            return any(args)
        if kind is sp.Not:
            return not args[0]
        if kind is sp.Xor:
            return sum(bool(arg) for arg in args) % 2 == 1
        if kind is sp.Equivalent:
            return len({bool(arg) for arg in args}) < 2
        if kind is sp.Implies:
            return not args[0] or bool(args[1])
        if kind is sp.Add:
            return sum(args)
        if kind is sp.Mul:
            return prod(args)
        a, b = args
        if kind is sp.Eq:
            return a == b
        if kind is sp.Ne:
            return a != b
        if kind is sp.Lt:
            return a < b
        if kind is sp.Le:
            return a <= b
        if kind is sp.Gt:
            return a > b
        return a >= b

    table = []
    for assignment in product((0, 1), repeat=len(bits)):
        storage = dict(zip(bits, assignment, strict=True))
        values = {
            symbol: sum(storage[bit] << i for i, bit in enumerate(reversed(keys[str(symbol)])))
            for symbol in symbols
        }
        table.append(all(bool(evaluate(expr, values)) for expr in expressions))
    # Count expanded IR work before constructing any decision Circuits. Shared
    # leaf objects do not hide their repeated cost from walkers/serializers.
    body = body()
    body_cost = max(1, sum(1 for _ in body.walk_operations()))

    def cost(values):
        if all(values):
            return body_cost, 0
        if not any(values):
            return 0, 0
        half = len(values) // 2
        left, right = values[:half], values[half:]
        if left == right:
            return cost(left)
        a, ad = cost(left)
        b, bd = cost(right)
        return 1 + a + b, 1 + max(ad, bd)

    work, depth = cost(table)
    if work > max_operations:
        raise conversion_error(
            "EXPANSION_LIMIT", "Classical decision expansion exceeds max_operations.", location
        )
    if depth > max_depth:
        raise conversion_error(
            "CONTROL_DEPTH_LIMIT", "Classical decision expansion exceeds max_depth.", location
        )
    empty = Circuit(body.num_qubits)
    width = max(bits, default=-1) + 1

    def build(values, depth=0):
        if all(values):
            return body
        if not any(values):
            return empty
        half = len(values) // 2
        left, right = values[:half], values[half:]
        if left == right:
            return build(left, depth + 1)
        return Circuit(
            body.num_qubits,
            (
                Operation(
                    "if_else",
                    (),
                    condition=ClassicalCondition((bits[depth],), 1),
                    blocks=(build(right, depth + 1), build(left, depth + 1)),
                ),
            ),
            width,
        )

    return build(table), work


def remap_storage_block(block, latest, width):
    result = []
    for op in block.operations:
        if op.name != "if_else":
            if op.name != "barrier":
                result.append(op)
            continue
        pairs = [(bit, (op.condition.value >> i) & 1) for i, bit in enumerate(op.condition.bits)]
        if any(bit not in latest and value for bit, value in pairs):
            selected = remap_storage_block(op.blocks[1], latest, width)
        else:
            active = [(latest[bit], value) for bit, value in pairs if bit in latest]
            if active:
                value = sum(v << i for i, (_, v) in enumerate(active))
                result.append(
                    replace(
                        op,
                        condition=ClassicalCondition(tuple(b for b, _ in active), value),
                        blocks=tuple(remap_storage_block(b, latest, width) for b in op.blocks),
                    )
                )
                continue
            selected = remap_storage_block(op.blocks[0], latest, width)
        # Keep a statically selected block's phase at its scope.
        result.extend(selected.operations)
        block = replace(block, global_phase=block.global_phase + selected.global_phase)
    return replace(block, operations=tuple(result), num_clbits=width)
