"""Bounded bit-vector/ROBDD compilation of typed Qiskit classical expressions."""

from ...errors import conversion_error
from ...ir.core import Circuit, ClassicalCondition


class _BDD:
    def __init__(self, work, depth, location):
        self.limit, self.depth, self.location = work, depth, location
        self.work = 0
        self.nodes = [None, None]
        self.heights = [0, 0]
        self.unique, self.cache = {}, {}

    def charge(self, amount=1):
        self.work += amount
        if self.work > self.limit:
            raise conversion_error(
                "CLASSICAL_EXPRESSION_LIMIT",
                "Classical predicate exceeds its work/node budget.",
                self.location,
            )

    def node(self, bit, low, high):
        if low == high:
            return low
        key = bit, low, high
        if key not in self.unique:
            self.charge()
            height = 1 + max(self.heights[low], self.heights[high])
            if height > self.depth:
                raise conversion_error(
                    "CLASSICAL_EXPRESSION_DEPTH",
                    "Classical predicate exceeds max_depth.",
                    self.location,
                )
            self.unique[key] = len(self.nodes)
            self.nodes.append(key)
            self.heights.append(height)
        return self.unique[key]

    def apply(self, op, left, right):
        def key(a, b):
            return op, min(a, b), max(a, b)

        root = key(left, right)
        pending = [(root, False)]
        while pending:
            current, ready = pending.pop()
            if current in self.cache:
                continue
            self.charge()
            _, a, b = current
            value = None
            if a < 2 and b < 2:
                value = (a & b) if op == "and" else (a | b) if op == "or" else (a ^ b)
            elif a == b:
                value = 0 if op == "xor" else a
            elif op == "and" and a < 2:
                value = b if a else 0
            elif op == "or" and a < 2:
                value = 1 if a else b
            elif op == "xor" and a == 0:
                value = b
            if value is not None:
                self.cache[current] = value
                continue
            variable = min(self.nodes[n][0] for n in (a, b) if n >= 2)

            def children(n, variable=variable):
                return self.nodes[n][1:] if n >= 2 and self.nodes[n][0] == variable else (n, n)

            al, ah = children(a)
            bl, bh = children(b)
            low, high = key(al, bl), key(ah, bh)
            if ready:
                self.cache[current] = self.node(variable, self.cache[low], self.cache[high])
            else:
                pending.append((current, True))
                pending.extend((k, False) for k in (low, high) if k not in self.cache)
        return self.cache[root]

    def truth(self, bits):
        self.charge(len(bits))
        value = 0
        for bit in bits:
            value = self.apply("or", value, bit)
        return value


class _BitVectors:
    """Fixed-width unsigned circuits, built from canonical Boolean operations."""

    def __init__(self, bdd):
        self.bdd = bdd

    def add(self, left, right, *, subtract=False, with_carry=False):
        self.bdd.charge(len(left))
        carry, result = int(subtract), []
        for a, b in zip(left, right, strict=True):
            if subtract:
                b = self.bdd.apply("xor", b, 1)
            parity = self.bdd.apply("xor", a, b)
            result.append(self.bdd.apply("xor", parity, carry))
            carry = self.bdd.apply(
                "or", self.bdd.apply("and", a, b), self.bdd.apply("and", parity, carry)
            )
        value = tuple(result)
        return (value, carry) if with_carry else value

    def select(self, condition, high, low):
        self.bdd.charge(len(high))
        return tuple(
            self.bdd.apply("xor", a, self.bdd.apply("and", condition, self.bdd.apply("xor", a, b)))
            for a, b in zip(low, high, strict=True)
        )

    def divide(self, left, right):
        if self.bdd.truth(right) != 1:
            raise conversion_error(
                "CLASSICAL_DIVISOR_DOMAIN",
                "Integer divisors must be symbolically proven nonzero for every classical input.",
                self.bdd.location,
            )
        size = len(left)
        if all(bit < 2 for bit in right):
            divisor = sum(bit << i for i, bit in enumerate(right))
            if divisor & (divisor - 1) == 0:
                shift = divisor.bit_length() - 1
                self.bdd.charge(size)
                return left[shift:] + (0,) * shift
        self.bdd.charge(size * size)
        remainder, divisor = (0,) * (size + 1), right + (0,)
        quotient = [0] * size
        for i in reversed(range(size)):
            remainder = (left[i],) + remainder[:-1]
            difference, carry = self.add(remainder, divisor, subtract=True, with_carry=True)
            remainder = self.select(carry, difference, remainder)
            quotient[i] = carry
        return tuple(quotient)

    def multiply(self, left, right):
        size = len(left)
        self.bdd.charge(size * size)
        result = (0,) * size
        for index, bit in enumerate(right):
            if bit == 0:
                continue
            row = (0,) * index + tuple(self.bdd.apply("and", bit, a) for a in left[: size - index])
            result = self.add(result, row)
        return result

    def shift(self, bits, amount, *, left=False, index=False):
        size = len(bits)
        width = max(len(amount), size.bit_length())
        extended = amount + (0,) * (width - len(amount))
        bound = tuple((size >> i) & 1 for i in range(width))
        _, greater_equal = self.add(extended, bound, subtract=True, with_carry=True)
        if greater_equal != 0:
            raise conversion_error(
                "CLASSICAL_INDEX_RANGE" if index else "CLASSICAL_SHIFT_RANGE",
                "Every possible index/shift must be symbolically proven inside its operand width.",
                self.bdd.location,
            )
        self.bdd.charge(size * len(amount))
        for i, control in enumerate(amount):
            if control == 0:
                continue
            step = 1 << i
            shifted = (0,) * step + bits[: size - step] if left else bits[step:] + (0,) * step
            if control == 1:
                bits = shifted
            else:
                bits = self.select(control, shifted, bits)
        return bits


def _compile(value, native, bdd, *, classical_map=None, variable_bits=None, return_bits=False):
    from qiskit.circuit import ClassicalRegister, Clbit
    from qiskit.circuit.classical import expr, types

    vectors = _BitVectors(bdd)
    index_type = getattr(expr, "Index", ())

    def fail(message):
        raise conversion_error("CLASSICAL_EXPRESSION", message, bdd.location)

    def width(kind):
        if isinstance(kind, types.Bool):
            return 1
        if not isinstance(kind, types.Uint):
            fail("Only Bool and unsigned-integer predicates can be lowered to portable IR.")
        if kind.width > bdd.limit - bdd.work:
            bdd.charge(kind.width)
        return kind.width

    def expand(bits, count):
        return (bits + (0,) * count)[:count]

    memo, descendants, pending = {}, {}, [(value, False, 0)]
    while pending:
        current, ready, depth = pending.pop()
        identity = id(current)
        if identity in memo:
            continue
        if depth > bdd.depth:
            raise conversion_error(
                "CLASSICAL_EXPRESSION_DEPTH",
                "Classical expression nesting exceeds max_depth.",
                bdd.location,
            )
        size = width(current.type)
        bdd.charge(size)
        if isinstance(current, expr.Var):
            if isinstance(current.var, Clbit):
                bits = (current.var,)
            elif isinstance(current.var, ClassicalRegister):
                bits = tuple(current.var)
            else:
                if variable_bits is None or current.var not in variable_bits:
                    fail("Standalone classical variables require declared storage semantics.")
                memo[identity] = tuple(bdd.node(bit, 0, 1) for bit in variable_bits[current.var])
                continue
            indices = tuple(native.find_bit(bit).index for bit in bits)
            if classical_map is not None:
                indices = tuple(classical_map[bit] for bit in indices)
            memo[identity] = tuple(bdd.node(bit, 0, 1) for bit in indices)
            continue
        if isinstance(current, expr.Value):
            if type(current.value) not in {bool, int} or not 0 <= current.value < (1 << size):
                fail("Classical literals must fit their declared unsigned width.")
            memo[identity] = tuple((int(current.value) >> i) & 1 for i in range(size))
            continue
        # Rust-backed SDK descriptors can return a fresh wrapper on each
        # property access. Retain the exact children used by this traversal.
        if identity not in descendants:
            descendants[identity] = (
                (current.left, current.right)
                if isinstance(current, expr.Binary)
                else (current.target, current.index)
                if isinstance(current, index_type)
                else (current.operand,)
                if isinstance(current, (expr.Unary, expr.Cast))
                else ()
            )
        children = descendants[identity]
        if not children:
            fail("Unsupported classical expression node.")
        if not ready:
            pending.append((current, True, depth))
            pending.extend((child, False, depth + 1) for child in children if id(child) not in memo)
            continue
        left = memo[id(children[0])]
        if isinstance(current, index_type):
            result = (vectors.shift(left, memo[id(children[1])], index=True)[0],)
        elif isinstance(current, expr.Cast):
            result = (
                (bdd.truth(left),) if isinstance(current.type, types.Bool) else expand(left, size)
            )
        elif isinstance(current, expr.Unary):
            if current.op.name == "LOGIC_NOT":
                result = (bdd.apply("xor", bdd.truth(left), 1),)
            elif current.op.name == "BIT_NOT":
                result = tuple(bdd.apply("xor", bit, 1) for bit in expand(left, size))
            else:
                fail(f"Unsupported classical unary operation {current.op.name!r}.")
        else:
            right = memo[id(children[1])]
            op = current.op.name
            if op in {"ADD", "SUB", "MUL", "DIV"}:
                a, b = expand(left, size), expand(right, size)
                result = (
                    vectors.multiply(a, b)
                    if op == "MUL"
                    else vectors.divide(a, b)
                    if op == "DIV"
                    else vectors.add(a, b, subtract=op == "SUB")
                )
            elif op in {"SHIFT_LEFT", "SHIFT_RIGHT"}:
                result = vectors.shift(expand(left, size), right, left=op == "SHIFT_LEFT")
            elif op in {"BIT_AND", "BIT_OR", "BIT_XOR"}:
                operation = op.removeprefix("BIT_").lower()
                result = tuple(
                    bdd.apply(operation, a, b)
                    for a, b in zip(expand(left, size), expand(right, size), strict=True)
                )
            elif op in {"LOGIC_AND", "LOGIC_OR"}:
                result = (
                    bdd.apply(op.removeprefix("LOGIC_").lower(), bdd.truth(left), bdd.truth(right)),
                )
            elif op in {"EQUAL", "NOT_EQUAL", "LESS", "LESS_EQUAL", "GREATER", "GREATER_EQUAL"}:
                count = max(len(left), len(right))
                bdd.charge(count)
                a, b = expand(left, count), expand(right, count)
                if op in {"GREATER", "GREATER_EQUAL"}:
                    a, b = b, a
                equal, less = 1, 0
                for x, y in zip(reversed(a), reversed(b), strict=True):
                    lower = bdd.apply("and", bdd.apply("xor", x, 1), y)
                    less = bdd.apply("or", less, bdd.apply("and", equal, lower))
                    equal = bdd.apply("and", equal, bdd.apply("xor", bdd.apply("xor", x, y), 1))
                result = (
                    (
                        equal
                        if op == "EQUAL"
                        else bdd.apply("xor", equal, 1)
                        if op == "NOT_EQUAL"
                        else bdd.apply("or", less, equal)
                        if op.endswith("_EQUAL")
                        else less
                    ),
                )
            else:
                fail(f"Classical arithmetic operation {op!r} requires a native program route.")
        memo[identity] = result
    return memo[id(value)] if return_bits else bdd.truth(memo[id(value)])


def lower_predicate(value, native, blocks, *, max_operations, max_depth, location):
    """Produce a shared, immutable equality-decision DAG with bounded expanded cost."""
    bdd = _BDD(max_operations, max_depth, location)
    root = _compile(value, native, bdd)
    # Match the adapter's outer recursive expansion-accounting boundary.
    payload_cost = tuple(sum(1 for _ in c.walk_operations()) for c in blocks)
    table = {0: blocks[1], 1: blocks[0]}
    costs = {0: payload_cost[1], 1: payload_cost[0]}
    pending, order, seen = [root], [], set()
    while pending:
        current = pending.pop()
        if current < 2 or current in seen:
            continue
        seen.add(current)
        order.append(current)
        pending.extend(bdd.nodes[current][1:])
    # Children always precede newly interned parents, even with shared nodes.
    for current in sorted(order):
        bit, low, high = bdd.nodes[current]
        cost = 1 + costs[low] + costs[high]
        if cost > max_operations:
            raise conversion_error(
                "EXPANSION_LIMIT", "Expanded predicate and payload exceed max_operations.", location
            )
        result = Circuit(native.num_qubits or 1, num_clbits=native.num_clbits).if_else(
            ClassicalCondition((bit,), 1), table[high], table[low]
        )
        table[current], costs[current] = result, cost
    result = table[root]
    # The predicate and nested payload together must fit the caller's depth.
    pending = [(result, 0)]
    while pending:
        body, depth = pending.pop()
        if depth > max_depth:
            raise conversion_error(
                "EXPANSION_LIMIT", "Predicate/payload exceed max_depth.", location
            )
        for operation in body.operations:
            pending.extend((child, depth + 1) for child in operation.blocks)
    return result
