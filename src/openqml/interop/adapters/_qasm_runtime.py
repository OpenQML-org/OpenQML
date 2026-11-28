"""Bounded classical addressing and structured OpenQASM control."""

import re
from dataclasses import dataclass, field

from ...errors import conversion_error
from ...ir.core import Circuit, ClassicalCondition
from ._classical_predicate import _BDD, _BitVectors, decision_circuit


def unsigned_word(text, registers, limit, location):
    """Return physical bits and their unsigned weights, omitting known zeroes."""
    source = text.strip()
    mask = re.fullmatch(r"(.+?)\s*&\s*(\d+)", source, re.S)
    if mask:
        source, mask = mask[1].strip(), int(mask[2])
    cast = re.fullmatch(r"uint(?:\[\s*(\d+)\s*\])?\s*\(\s*(.*?)\s*\)", source, re.S)
    if cast:
        source = cast[2]
    ref = re.fullmatch(r"([A-Za-z_]\w*)(?:\[\s*(\d+)\s*\])?", source, re.ASCII)
    if not ref or ref[1] not in registers:
        raise conversion_error(
            "REGISTER_REFERENCE", f"Unsupported unsigned operand {text!r}.", location
        )
    start, width = registers[ref[1]]
    if ref[2] is not None:
        index = int(ref[2])
        if index >= width:
            raise conversion_error(
                "REGISTER_BOUNDS", "Classical bit is outside the register.", location
            )
        start, width = start + index, 1
    if cast:
        if cast[1] is None and width > 32:
            raise conversion_error(
                "UNSUPPORTED_STATEMENT", "Specify the cast width above 32 bits.", location
            )
        width = min(width, int(cast[1]) if cast[1] else 32)
        if width == 0:
            raise conversion_error(
                "REGISTER_REFERENCE", "An unsigned cast needs positive width.", location
            )
    if mask is not None:
        width = min(width, mask.bit_length())
    if width > limit:
        raise conversion_error(
            "OPERATION_LIMIT", "Unsigned operand exceeds the work budget.", location
        )
    return tuple((start + i, i) for i in range(width) if mask is None or (mask >> i) & 1)


@dataclass(frozen=True)
class Address:
    start: int
    digits: tuple
    extent: int
    high_bits: tuple = ()
    checked: bool = False


def runtime_address(text, registers, classical, bounds, limit, location):
    ref = re.fullmatch(r"([A-Za-z_]\w*)\s*\[(.+)\]", text.strip(), re.S | re.ASCII)
    if not ref or ref[1] not in registers:
        raise conversion_error(
            "REGISTER_REFERENCE", f"Unknown register reference {text!r}.", location
        )
    base, width = registers[ref[1]]
    digits = unsigned_word(ref[2], classical, limit, location)
    upper = bounds.get(digits)
    if upper is not None:
        digits = tuple((b, i) for b, i in digits if i < max(0, upper - 1).bit_length())
    extent = width if upper is None else min(width, upper)
    if upper is not None and upper <= width:
        return Address(base, digits, extent)
    size = (width - 1).bit_length()
    low = tuple((b, i) for b, i in digits if i < size)
    high = tuple(b for b, i in digits if i >= size)
    return Address(base, low, extent, high, bool(high) or sum(1 << i for _, i in low) >= width)


def split_operands(text):
    depth = start = 0
    for i, ch in enumerate(text):
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        elif ch == "," and not depth:
            yield text[start:i]
            start = i + 1
    yield text[start:]


def addressed(groups, emit, nq, nc, limit, depth, location):
    indices = [g for g in groups if isinstance(g, Address)]
    if not indices:
        yield from emit(groups)
        return
    high = {b for g in indices for b in g.high_bits}
    bits = tuple(sorted({b for g in indices for b, _ in g.digits} - high))
    if len(bits) + bool(high) > depth or len(bits) >= (limit + 1).bit_length():
        raise conversion_error(
            "OPERATION_LIMIT", "Dynamic dispatch exceeds the depth/work budget.", location
        )
    work = 2 if high else 0
    empty = Circuit(nq, num_clbits=nc)
    invalid = empty.gate("index_error")

    def visit(level, ones):
        nonlocal work
        work += 1
        if work > limit:
            raise conversion_error(
                "OPERATION_LIMIT", "Dynamic address expansion exceeds max_operations.", location
            )
        if level < len(bits):
            bit = bits[level]
            return empty.if_else(
                ClassicalCondition((bit,), 1),
                visit(level + 1, ones | {bit}),
                visit(level + 1, ones),
            )
        targets = []
        for group in groups:
            if isinstance(group, Address):
                index = sum(1 << i for b, i in group.digits if b in ones)
                if index >= group.extent:
                    return invalid if group.checked else empty
                targets.append((group.start + index,))
            else:
                targets.append(group)
        ops = []
        for op in emit(targets):
            work += 1
            if work > limit:
                raise conversion_error(
                    "OPERATION_LIMIT", "Expanded operands exceed the budget.", location
                )
            ops.append(op)
        return Circuit(nq, tuple(ops), nc)

    result = visit(0, frozenset())
    if high:
        result = empty.if_else(ClassicalCondition(tuple(sorted(high)), 0), result, invalid)
    yield from result.operations


@dataclass(frozen=True)
class Predicate:
    digits: tuple
    high: int | None
    low: int | None
    equality: ClassicalCondition | None = None
    inverted: bool = False
    bdd: object = None
    root: int = 0

    def select(self, yes, no, nq, nc, limit, depth, location):
        if self.equality is not None:
            if self.inverted:
                yes, no = no, yes
            return Circuit(nq, num_clbits=nc).if_else(self.equality, yes, no)
        return decision_circuit(self.bdd, self.root, (yes, no), nq, nc, limit, depth, location)


def comparison(text, registers, limit, depth, location):
    m = re.fullmatch(r"(.+?)\s*(==|!=|<=|>=|<|>)\s*(\d+|true|false)", text.strip(), re.S)
    if not m:
        raise conversion_error(
            "UNSUPPORTED_STATEMENT", "Expected an unsigned comparison with a literal.", location
        )
    digits = unsigned_word(m[1], registers, limit, location)
    op, v = m[2], (int(m[3] == "true") if m[3] in {"true", "false"} else int(m[3]))
    high = {"<": v, "<=": v + 1, "==": v + 1}.get(op)
    low = {">=": v, ">": v + 1, "!=": v + 1}.get(op)
    threshold = v + int(op in {"<=", ">"})
    if op in {"<", "<=", ">", ">="} and threshold > 0 and threshold & (threshold - 1) == 0:
        bits = tuple(bit for bit, i in digits if i >= threshold.bit_length() - 1)
        if bits:
            return Predicate(digits, high, low, ClassicalCondition(bits, 0), op in {">", ">="})
    bdd = _BDD(limit, depth, location)
    wires = {i: b for b, i in digits}
    equal, less = 1, 0
    if v.bit_length() > max(wires, default=-1) + 1:
        equal, less = 0, 1
    else:
        for i in range(max(wires, default=-1) + 1):
            bit = bdd.node(wires[i], 0, 1) if i in wires else 0
            one = (v >> i) & 1
            same = bit if one else bdd.apply("xor", bit, 1)
            less = bdd.apply(
                "or", bdd.apply("xor", bit, 1) if one else 0, bdd.apply("and", same, less)
            )
            equal = bdd.apply("and", same, equal)
    root = (
        equal if op in {"==", "!="} else less if op in {"<", ">="} else bdd.apply("or", less, equal)
    )
    if op in {"!=", ">=", ">"}:
        root = bdd.apply("xor", root, 1)
    return Predicate(digits, high, low, bdd=bdd, root=root)


def counter(start, width, value, subtract, nq, nc, limit, depth, location):
    value %= 1 << width
    if value == 0:
        return
    empty = Circuit(nq, num_clbits=nc)
    if value & (value - 1) == 0:
        first = value.bit_length() - 1
        length = width - first
        if length * (length + 1) // 2 + 4 * length - 1 > limit:
            raise conversion_error(
                "OPERATION_LIMIT", "Counter carry tests exceed the work budget.", location
            )
        for i in reversed(range(first, width)):
            bit = start + i
            toggle = empty.if_else(
                ClassicalCondition((bit,), 1),
                empty.classical_set(bit, 0),
                empty.classical_set(bit, 1),
            )
            if i > first:
                lower = tuple(range(start + first, bit))
                toggle = empty.if_else(
                    ClassicalCondition(lower, 0 if subtract else (1 << len(lower)) - 1), toggle
                )
            yield from toggle.operations
        return
    bdd = _BDD(limit, depth, location)
    bdd.charge(width)
    bits = tuple(bdd.node(start + i, 0, 1) for i in range(width))
    roots = _BitVectors(bdd).add(
        bits, tuple((value >> i) & 1 for i in range(width)), subtract=subtract
    )
    for i in reversed(range(width)):
        if bits[i] != roots[i]:
            body = decision_circuit(
                bdd,
                roots[i],
                (empty.classical_set(start + i, 1), empty.classical_set(start + i, 0)),
                nq,
                nc,
                limit,
                depth,
                location,
            )
            limit -= sum(1 for _ in body.walk_operations())
            yield from body.operations


@dataclass
class Switch:
    digits: tuple
    cases: dict = field(default_factory=dict)
    default: Circuit | None = None

    def add(self, labels, body, location):
        if labels is None:
            if self.default is not None:
                raise conversion_error("CLASSICAL_SWITCH_LABEL", "Duplicate default.", location)
            self.default = body
        else:
            for label in labels:
                if label in self.cases:
                    raise conversion_error(
                        "CLASSICAL_SWITCH_LABEL", "Duplicate switch label.", location
                    )
                self.cases[label] = body

    def lower(self, nq, nc, limit, depth, location):
        from ...ir.compiler import _check_circuit_limits

        empty = Circuit(nq, num_clbits=nc)
        result = self.default if self.default is not None else empty
        bits = tuple(b for b, _ in self.digits)
        if not bits:
            return self.cases.get(0, result)
        mask = sum(1 << i for _, i in self.digits)
        cases = [(v, b) for v, b in self.cases.items() if v & ~mask == 0]
        if len(cases) > depth:
            raise conversion_error(
                "CONTROL_DEPTH_LIMIT", "Switch expansion exceeds max_depth.", location
            )
        for value, body in reversed(cases):
            packed = sum(((value >> i) & 1) << j for j, (_, i) in enumerate(self.digits))
            result = empty.if_else(ClassicalCondition(bits, packed), body, result)
        _check_circuit_limits(result, max_operations=max(1, limit), max_depth=depth)
        return result
