"""Factored, exact density kernels for bounded quantum instruments.

Each sparse entry is a coefficient of an input matrix unit, quantum output
matrix unit and final classical record. Hidden histories are added, never
compared as trajectories. Coefficients are exponential polynomials over
Q(zeta_8), with exact rational arithmetic and conservative real atoms.

Independent components are factored using the UNION of both programs' quantum
and classical dependencies. No dense state or optional SDK is used. The worst
case is exponential; width, live terms, work and coefficient sizes are bounded.
See docs/semantic-kernels.md for the semantics, trust boundary and references.
"""

import json
from fractions import Fraction
from math import isqrt

from .errors import ResourceError
from .ir import CHANNEL_GATES, Operation
from .semantics import _PI, _angle, _bounded_coefficient

# zeta = exp(i*pi/4), zeta**4 = -1. Unique rational coordinates make comparison
# exact. In particular named H/T gates never acquire floating-point entries.
_ZERO = (Fraction(0),) * 4
_ONE = (Fraction(1),) + _ZERO[1:]
_I = (_ZERO[0], _ZERO[0], Fraction(1), _ZERO[0])
_MONOMIAL = ((), ())  # (phase exponent, real-atom powers)


def _cadd(a, b):
    return tuple(_bounded_coefficient(x + y) for x, y in zip(a, b, strict=True))


def _cmul(a, b):
    result = list(_ZERO)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x and y:
                k = i + j
                result[k % 4] += x * y * (-1 if k >= 4 else 1)
    return tuple(_bounded_coefficient(x) for x in result)


def _cconj(a):
    return a[0], -a[3], -a[2], -a[1]


def _zeta(power):
    power %= 8
    return tuple(Fraction((-1 if power >= 4 else 1) if i == power % 4 else 0) for i in range(4))


class _Algebra:
    def __init__(self, max_terms, max_work):
        self.max_terms, self.max_work, self.work = max_terms, max_work, 0

    def charge(self, count=1):
        self.work += count
        if self.work > self.max_work:
            raise ResourceError("Exact density-kernel proof exceeds max_work.")

    def admit(self, terms):
        if terms > self.max_terms:
            raise ResourceError("Exact density-kernel proof exceeds max_terms.")

    def constant(self, value=1):
        value = complex(value)
        coefficient = (Fraction(value.real), Fraction(0), Fraction(value.imag), Fraction(0))
        return {_MONOMIAL: coefficient} if any(coefficient) else {}

    def scale(self, value, factor):
        factor = Fraction(factor)
        if not factor:
            return {}
        return {key: tuple(_bounded_coefficient(x * factor) for x in c) for key, c in value.items()}

    def add(self, *values):
        result = {}
        for value in values:
            for key, c in value.items():
                self.charge()
                c = _cadd(result.get(key, _ZERO), c)
                if any(c):
                    result[key] = c
                    self.admit(len(result))
                else:
                    result.pop(key, None)
        return result

    def _phase(self, phase):
        phase = {atom: _bounded_coefficient(c) for atom, c in phase.items() if c}
        pi = phase.get(_PI, Fraction(0)) % 2
        coefficient = _ONE
        if (pi * 4).denominator == 1:
            coefficient = _zeta(int(pi * 4))
            phase.pop(_PI, None)
        elif pi:
            phase[_PI] = pi
        return tuple(sorted(phase.items())), coefficient

    def mul(self, a, b):
        result = {}
        for (phase_a, powers_a), ca in a.items():
            for (phase_b, powers_b), cb in b.items():
                self.charge(1 + len(phase_a) + len(phase_b) + len(powers_a) + len(powers_b))
                phase, powers = dict(phase_a), dict(powers_a)
                for atom, coefficient in phase_b:
                    phase[atom] = phase.get(atom, 0) + coefficient
                for atom, power in powers_b:
                    powers[atom] = powers.get(atom, 0) + power
                    if powers[atom].bit_length() > 4096:
                        raise ResourceError(
                            "Exact real-atom power exceeds the 4096-bit kernel limit."
                        )
                phase, root = self._phase(phase)
                key = phase, tuple(sorted(powers.items()))
                c = _cadd(result.get(key, _ZERO), _cmul(_cmul(ca, cb), root))
                if any(c):
                    result[key] = c
                    self.admit(len(result))
                else:
                    result.pop(key, None)
        return result

    def conj(self, value):
        result = {}
        for (phase, powers), c in value.items():
            self.charge(1 + len(phase))
            phase, root = self._phase({atom: -v for atom, v in phase})
            result[phase, powers] = _cmul(_cconj(c), root)
        return result

    def exp(self, value, factor=1):
        phase, c = self._phase({atom: v * factor for atom, v in _angle(value).items()})
        self.charge(1 + len(phase))
        return {(phase, ()): c}

    def atom(self, name):
        return {((), ((name, 1),)): _ONE}

    def real(self, expr):
        self.charge()
        if expr.op is None:
            result = self.constant(expr.constant)
            for name, coefficient in expr.terms:
                result = self.add(result, self.scale(self.atom(("parameter", name)), coefficient))
            return result
        if expr.op == "add":
            return self.add(*(self.real(arg) for arg in expr.args))
        if expr.op == "mul":
            return self.mul(*(self.real(arg) for arg in expr.args))
        if expr.op == "div" and expr.args[1].op is None and not expr.args[1].terms:
            denominator = expr.args[1].constant
            if denominator:
                return self.scale(self.real(expr.args[0]), 1 / Fraction(denominator))
        # Identical syntax has identical real denotation on the defined domain.
        # No algebraic independence or sampled evaluation is assumed.
        return self.atom(("expression", json.dumps(expr.to_dict(), sort_keys=True)))

    def sqrt_one_minus(self, expr):
        if expr.op is None and not expr.terms:
            value = 1 - Fraction(expr.constant)
            a, b = isqrt(value.numerator), isqrt(value.denominator)
            if a * a == value.numerator and b * b == value.denominator:
                return self.scale(self.constant(), Fraction(a, b))
        return self.atom(("sqrt_one_minus", json.dumps(expr.to_dict(), sort_keys=True)))


def _matrix_columns(matrix, alg):
    return tuple(
        tuple((r, alg.constant(row[c])) for r, row in enumerate(matrix) if row[c])
        for c in range(len(matrix))
    )


def _matrix(op, alg):
    """Sparse local unitary columns, in q0-most-significant order."""
    one, imag = alg.constant(), alg.constant(1j)

    def minus(x):
        return alg.scale(x, -1)

    name = op.name
    if name == "unitary":
        return _matrix_columns(op.matrix, alg)
    fixed = {
        "id": ((one, {}), ({}, one)),
        "x": (({}, one), (one, {})),
        "y": (({}, minus(imag)), (imag, {})),
        "z": ((one, {}), ({}, minus(one))),
        "s": ((one, {}), ({}, imag)),
        "sdg": ((one, {}), ({}, minus(imag))),
        "t": ((one, {}), ({}, {_MONOMIAL: _zeta(1)})),
        "tdg": ((one, {}), ({}, {_MONOMIAL: _zeta(-1)})),
    }
    root_half = {_MONOMIAL: (Fraction(0), Fraction(1, 2), Fraction(0), Fraction(-1, 2))}
    fixed["h"] = ((root_half, root_half), (root_half, minus(root_half)))
    plus, difference = (
        alg.scale(alg.add(one, imag), Fraction(1, 2)),
        alg.scale(alg.add(one, minus(imag)), Fraction(1, 2)),
    )
    fixed["sx"] = ((plus, difference), (difference, plus))
    fixed["sxdg"] = ((difference, plus), (plus, difference))
    if name in fixed:
        rows = fixed[name]
    elif name in {"rx", "ry", "rz", "p", "u", "rxx", "ryy", "rzz"}:
        angle = op.params[0]
        pos, neg = alg.exp(angle, Fraction(1, 2)), alg.exp(angle, Fraction(-1, 2))
        cosine = alg.scale(alg.add(pos, neg), Fraction(1, 2))
        sine = alg.mul(alg.scale(alg.add(pos, minus(neg)), Fraction(1, 2)), minus(imag))
        if name == "p":
            rows = ((one, {}), ({}, alg.exp(angle)))
        elif name == "rz":
            rows = ((neg, {}), ({}, pos))
        elif name == "rx":
            off = alg.mul(minus(imag), sine)
            rows = ((cosine, off), (off, cosine))
        elif name == "ry":
            rows = ((cosine, minus(sine)), (sine, cosine))
        elif name == "u":
            phi, lam = op.params[1:]
            rows = (
                (cosine, minus(alg.mul(alg.exp(lam), sine))),
                (alg.mul(alg.exp(phi), sine), alg.mul(alg.mul(alg.exp(phi), alg.exp(lam)), cosine)),
            )
        else:
            columns = []
            for c in range(4):
                if name == "rzz":
                    columns.append(((c, neg if c in {0, 3} else pos),))
                else:
                    sign = -1 if name == "ryy" and c in {0, 3} else 1
                    columns.append(
                        ((c, cosine), (c ^ 3, alg.scale(alg.mul(minus(imag), sine), sign)))
                    )
            return tuple(columns)
    elif name in {"cx", "cy", "cz", "ch", "cp", "crx", "cry", "crz"}:
        base = _matrix(Operation(name[1:], (0,), op.params), alg)
        return (((0, one),), ((1, one),)) + tuple(
            tuple((r + 2, value) for r, value in column) for column in base
        )
    elif name in {"swap", "iswap", "cswap", "ccx"}:
        permutation = {
            "swap": (0, 2, 1, 3),
            "iswap": (0, 2, 1, 3),
            "cswap": (0, 1, 2, 3, 4, 6, 5, 7),
            "ccx": (0, 1, 2, 3, 4, 5, 7, 6),
        }[name]
        return tuple(
            ((r, imag if name == "iswap" and c in {1, 2} else one),)
            for c, r in enumerate(permutation)
        )
    else:
        raise AssertionError(f"Missing exact unitary rule: {name}")
    return tuple(
        tuple((r, row[c]) for r, row in enumerate(rows) if row[c]) for c in range(len(rows))
    )


def _resources(op):
    if op.name == "barrier":
        return set()
    result = {(0, q) for q in op.qubits} | {(1, c) for c in op.clbits}
    if op.condition:
        result.update((1, bit) for bit in op.condition.bits)
        for block in op.blocks:
            for child in block.operations:
                result.update(_resources(child))
    return result


def _components(source, target, alg):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for circuit in (source, target):
        for op in circuit.operations:
            resources = sorted(_resources(op))
            alg.charge(1 + len(resources))
            if resources:
                root = find(resources[0])
                for resource in resources[1:]:
                    parent[find(resource)] = root
    groups = {}
    for resource in parent:
        groups.setdefault(find(resource), set()).add(resource)
    return tuple(sorted(tuple(sorted(g)) for g in groups.values()))


def _kernel(circuit, component, alg):
    quantum = {q: i for i, (kind, q) in enumerate(r for r in component if r[0] == 0)}
    classical = {q: i for i, (kind, q) in enumerate(r for r in component if r[0] == 1)}
    n, component = len(quantum), set(component)
    size = 1 << n
    alg.admit(size * size)
    alg.charge(size * size)
    state = {(r * size + c, r, c, 0): alg.constant() for r in range(size) for c in range(size)}

    def evolve(body, current):
        for op in body.operations:
            alg.charge()
            if not (_resources(op) & component):
                continue
            if op.name == "if_else":
                groups = ({}, {})
                for key, value in current.items():
                    record = key[3]
                    match = all(
                        bool(record & (1 << classical[bit])) == bool((op.condition.value >> i) & 1)
                        for i, bit in enumerate(op.condition.bits)
                    )
                    alg.charge(1 + len(op.condition.bits))
                    groups[0 if match else 1][key] = value
                a, b = (
                    evolve(block, group) if group else {}
                    for block, group in zip(op.blocks, groups, strict=True)
                )
                current = a
                for key, value in b.items():
                    current[key] = alg.add(current.get(key, {}), value)
                alg.admit(sum(len(v) for v in current.values()))
                continue
            output, count = {}, 0

            def put(key, value, output=output):
                nonlocal count
                previous = output.get(key, {})
                merged = alg.add(previous, value)
                count += len(merged) - len(previous)
                alg.admit(count)
                if merged:
                    output[key] = merged
                else:
                    output.pop(key, None)

            masks = tuple(1 << (n - quantum[q] - 1) for q in op.qubits)
            clear = sum(masks)

            def local(index, masks=masks):
                value = 0
                for mask in masks:
                    value = (value << 1) | bool(index & mask)
                return value

            def scatter(index, value, clear=clear, masks=masks):
                result = index & ~clear
                for i, mask in enumerate(masks):
                    if value & (1 << (len(masks) - i - 1)):
                        result |= mask
                return result

            matrices = None
            if op.name not in CHANNEL_GATES | {"measure", "reset"}:
                matrices = ((_matrix(op, alg), alg.constant()),)
            elif op.name == "kraus":
                matrices = tuple(
                    (_matrix_columns(matrix, alg), alg.constant()) for matrix in op.kraus
                )
            elif op.name in {"bit_flip", "phase_flip", "depolarizing"}:
                probability = alg.real(op.params[0])
                paulis = {
                    "bit_flip": ("x",),
                    "phase_flip": ("z",),
                    "depolarizing": ("x", "y", "z"),
                }[op.name]
                matrices = (
                    (
                        _matrix(Operation("id", (0,)), alg),
                        alg.add(alg.constant(), alg.scale(probability, -1)),
                    ),
                ) + tuple(
                    (
                        _matrix(Operation(gate, (0,)), alg),
                        alg.scale(probability, Fraction(1, len(paulis))),
                    )
                    for gate in paulis
                )
            if matrices is not None:
                # Prepare local superoperator columns once, instead of repeating
                # exact polynomial products for every input matrix unit/record.
                transitions, transition_terms = {}, 0
                for a in range(1 << len(masks)):
                    for b in range(1 << len(masks)):
                        terms = {}
                        for matrix, weight in matrices:
                            for r, ca in matrix[a]:
                                for c, cb in matrix[b]:
                                    terms[r, c] = alg.add(
                                        terms.get((r, c), {}),
                                        alg.mul(weight, alg.mul(ca, alg.conj(cb))),
                                    )
                        transitions[a, b] = tuple(
                            (r, c, value) for (r, c), value in terms.items() if value
                        )
                        transition_terms += sum(len(v) for _, _, v in transitions[a, b])
                        alg.admit(transition_terms)
            for (input_index, ket, bra, record), value in current.items():
                alg.charge(1 + len(value))
                a, b = local(ket), local(bra)
                if matrices is not None:
                    for r, c, coefficient in transitions[a, b]:
                        put(
                            (input_index, scatter(ket, r), scatter(bra, c), record),
                            alg.mul(value, coefficient),
                        )
                elif op.name in {"measure", "reset"}:
                    if a == b:
                        if op.name == "measure":
                            bit = 1 << classical[op.clbits[0]]
                            put((input_index, ket, bra, (record & ~bit) | (bit if a else 0)), value)
                        else:
                            put((input_index, scatter(ket, 0), scatter(bra, 0), record), value)
                else:
                    probability = alg.real(op.params[0])
                    if a != b:
                        root = alg.sqrt_one_minus(op.params[0])
                        put((input_index, ket, bra, record), alg.mul(value, root))
                    elif op.name == "amplitude_damping" and a == 1:
                        put(
                            (input_index, scatter(ket, 0), scatter(bra, 0), record),
                            alg.mul(value, probability),
                        )
                        put(
                            (input_index, ket, bra, record),
                            alg.mul(value, alg.add(alg.constant(), alg.scale(probability, -1))),
                        )
                    else:
                        put((input_index, ket, bra, record), value)
            current = output
        return current

    return evolve(circuit, state)


def instrument_form(source, target, max_terms, max_work, max_component_qubits):
    """Return a sufficient exact equality certificate and consumed work.

    Matrix units span ALL quantum inputs, including entangled inputs. Classical
    storage starts at zero and every final classical bit remains observable.
    Circuit and conditional-block global phases cancel in instrument semantics.
    """
    alg, normal = _Algebra(max_terms, max_work), []
    for component in _components(source, target, alg):
        width = sum(kind == 0 for kind, _ in component)
        if width > max_component_qubits:
            raise ResourceError("Exact density-kernel component exceeds max_component_qubits.")
        left = _kernel(source, component, alg)
        right = _kernel(target, component, alg)
        alg.admit(sum(len(v) for v in left.values()) + sum(len(v) for v in right.values()))
        if left != right:
            return None, alg.work
        normal.append(
            (component, tuple((key, tuple(sorted(v.items()))) for key, v in sorted(left.items())))
        )
    return tuple(normal), alg.work


def _operator_kernel(circuit, component, alg):
    """Exact amplitudes, including relative phases, for every input column."""
    wires = {q: i for i, (_, q) in enumerate(component)}
    n = len(wires)
    size = 1 << n
    alg.admit(size)
    state = {(column, column): alg.constant() for column in range(size)}
    for op in circuit.operations:
        alg.charge()
        if not op.qubits or op.name == "barrier" or op.qubits[0] not in wires:
            continue
        columns = _matrix(op, alg)
        masks = tuple(1 << (n - wires[q] - 1) for q in op.qubits)
        clear, output, count = sum(masks), {}, 0
        for (column, index), value in state.items():
            local = 0
            for mask in masks:
                local = (local << 1) | bool(index & mask)
            alg.charge(1 + len(masks))
            for row, coefficient in columns[local]:
                target = index & ~clear
                for i, mask in enumerate(masks):
                    if row & (1 << (len(masks) - i - 1)):
                        target |= mask
                key = column, target
                previous = output.get(key, {})
                merged = alg.add(previous, alg.mul(value, coefficient))
                count += len(merged) - len(previous)
                alg.admit(count)
                if merged:
                    output[key] = merged
                else:
                    output.pop(key, None)
        state = output
    return state


def operator_form(source, target, max_terms, max_work, max_component_qubits):
    """Sufficient phase-sensitive equality of independent operator factors.

    Cross-factor phase compensation is deliberately not normalized here.
    A mismatch can therefore remain inconclusive for equivalent circuits.
    """
    alg, normal = _Algebra(max_terms, max_work), []
    source_phase, target_phase = alg.exp(source.global_phase), alg.exp(target.global_phase)
    if source_phase != target_phase:
        return None, alg.work
    for component in _components(source, target, alg):
        if len(component) > max_component_qubits:
            raise ResourceError("Exact operator-kernel component exceeds max_component_qubits.")
        left = _operator_kernel(source, component, alg)
        right = _operator_kernel(target, component, alg)
        alg.admit(sum(len(v) for v in left.values()) + sum(len(v) for v in right.values()))
        if left != right:
            return None, alg.work
        normal.append(
            (component, tuple((key, tuple(sorted(v.items()))) for key, v in sorted(left.items())))
        )
    return (tuple(sorted(source_phase.items())), tuple(normal)), alg.work
