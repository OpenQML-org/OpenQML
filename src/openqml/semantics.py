"""Bounded algebraic circuit equivalence, separate from numerical verification.

The trusted kernel contains exact circuit identities, affine Boolean maps with
symbolic phase polynomials, and binary Pauli conjugation. No sampled parameter
bindings, tolerances, numerical state vectors or target-runtime calls establish
a proof. An unsuccessful normal-form comparison is inconclusive.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from fractions import Fraction

from .compiler import _check_circuit_limits
from .errors import ConversionError, Diagnostic, ResourceError, ValidationError
from .ir import integer
from .verification import _expression_work

_CLIFFORD = frozenset("id x y z h s sdg sx sxdg cx cy cz swap iswap".split())
_PHASE = frozenset("id x y z s sdg t tdg p rz cx cz cp swap rzz barrier".split())
_PI = ("pi",)
_ONE = ("constant",)


def _bounded_coefficient(value):
    if max(value.numerator.bit_length(), value.denominator.bit_length()) > 4096:
        raise ResourceError("Exact proof coefficient complexity exceeds the 4096-bit kernel limit.")
    return value


def _add(left, right, scale=Fraction(1)):
    result = dict(left)
    _accumulate(result, right, scale)
    return result


def _accumulate(result, right, scale=Fraction(1)):
    for atom, coefficient in right.items():
        value = _bounded_coefficient(result.get(atom, 0) + coefficient * scale)
        if value:
            result[atom] = value
        else:
            result.pop(atom, None)


def _scale(value, factor):
    return {
        key: _bounded_coefficient(coefficient * factor)
        for key, coefficient in value.items()
        if coefficient * factor
    }


def _angle(value):
    """Exact binary-rational coefficients; nonlinear functions remain opaque atoms."""
    if value.op is None:
        result = {("parameter", name): Fraction(coefficient) for name, coefficient in value.terms}
        if value.constant:
            result[_ONE] = Fraction(value.constant)
        return result
    if value.op == "add":
        return _add(_angle(value.args[0]), _angle(value.args[1]))
    if value.op in {"mul", "div"}:
        a, b = value.args
        if b.op is None and not b.terms and (value.op == "mul" or b.constant):
            factor = Fraction(b.constant)
            return _scale(_angle(a), factor if value.op == "mul" else 1 / factor)
        if value.op == "mul" and a.op is None and not a.terms:
            return _scale(_angle(b), Fraction(a.constant))
    # Equal atom syntax is sufficient for equality even when two different
    # expressions denote the same function. Never assume atoms independent.
    return {("expression", json.dumps(value.to_dict(), sort_keys=True)): Fraction(1)}


def _frozen_angle(value):
    result = dict(value)
    if _PI in result:
        result[_PI] %= 2
        if not result[_PI]:
            del result[_PI]
    return tuple(sorted(result.items()))


def _fingerprint(circuit):
    return hashlib.sha256(circuit.to_json(indent=None).encode()).hexdigest()


@dataclass(frozen=True)
class EquivalenceProof:
    proven: bool
    method: str
    semantics: str
    source_fingerprint: str
    target_fingerprint: str
    parameters: tuple[str, ...]
    checked_operations: int
    work: int
    normal_form_fingerprint: str | None
    diagnostics: tuple[Diagnostic, ...] = ()

    def raise_for_errors(self):
        if not self.proven:
            raise ConversionError(self.diagnostics)

    def check(self, source, target, **limits):
        """Recompute the exact decision; reject stale source/target associations."""
        checked = prove_equivalence(
            source, target, phase_sensitive=self.semantics != "projective_unitary", **limits
        )
        return (
            checked.proven
            and checked.source_fingerprint == self.source_fingerprint
            and checked.target_fingerprint == self.target_fingerprint
        )

    def to_dict(self):
        return {
            "scope": "IR_algebraic_equivalence",
            "parameter_domain": "all_real_bindings_where_expressions_are_defined",
            "numeric_constants": "exact_stored_binary_rationals",
            "requires_valid_IR_parameters": True,
            "coefficient_limit_bits": 4096,
            "named_gate_angles": "mathematical_pi",
            "global_phase": (
                "unobservable_for_quantum_instruments"
                if self.semantics == "quantum_instrument"
                else "ignored"
                if self.semantics == "projective_unitary"
                else "compared"
            ),
            "classical_initialization": "all_zero",
            "classical_outputs": "all_final_bits",
            "target_execution_checked": False,
            "inconclusive_is_inequivalent": False,
            **asdict(self),
        }

    def to_json(self, *, indent=2):
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)


def _digest(form):
    # Hashes identify a result for auditing. The decision compares full forms.
    return hashlib.sha256(repr(form).encode()).hexdigest()


def _phase_form(circuit, max_terms, max_work):
    wires = [(1 << q, 0) for q in range(circuit.num_qubits)]
    phases = {0: _angle(circuit.global_phase)}
    work = 0
    atom_count = len(phases[0])

    def accumulate(mask, angle, scale=Fraction(1)):
        nonlocal atom_count
        old = phases.setdefault(mask, {})
        size = len(old)
        _accumulate(old, angle, scale)
        atom_count += len(old) - size
        if len(phases) + atom_count > max_terms:
            raise ResourceError("Phase-polynomial proof exceeds max_terms.")

    def add(bit, angle):
        nonlocal work
        mask, offset = bit
        work += 1 + len(angle)
        if work > max_work:
            raise ResourceError("Phase-polynomial proof exceeds max_work.")
        if offset:
            accumulate(0, angle)
            angle = _scale(angle, -1)
        if mask:
            accumulate(mask, angle)

    for op in circuit.operations:
        if op.name not in _PHASE or op.condition or op.blocks:
            return None, work
        name, q = op.name, op.qubits
        work += 1
        if work > max_work:
            raise ResourceError("Phase-polynomial proof exceeds max_work.")
        if name in {"id", "barrier"}:
            continue
        a = wires[q[0]]
        if name in {"x", "y"}:
            if name == "y":
                accumulate(0, {_PI: Fraction(1, 2)})
                add(a, {_PI: Fraction(1)})
            wires[q[0]] = a[0], a[1] ^ 1
        elif name == "cx":
            b = wires[q[1]]
            wires[q[1]] = a[0] ^ b[0], a[1] ^ b[1]
        elif name == "swap":
            wires[q[0]], wires[q[1]] = wires[q[1]], a
        elif name in {"cz", "cp", "rzz"}:
            b = wires[q[1]]
            parity = a[0] ^ b[0], a[1] ^ b[1]
            angle = {_PI: Fraction(1)} if name == "cz" else _angle(op.params[0])
            if name == "rzz":
                accumulate(0, angle, Fraction(-1, 2))
                add(parity, angle)
            else:
                add(a, _scale(angle, Fraction(1, 2)))
                add(b, _scale(angle, Fraction(1, 2)))
                add(parity, _scale(angle, Fraction(-1, 2)))
        else:
            named = {
                "z": 1,
                "s": Fraction(1, 2),
                "sdg": Fraction(-1, 2),
                "t": Fraction(1, 4),
                "tdg": Fraction(-1, 4),
            }
            angle = {_PI: Fraction(named[name])} if name in named else _angle(op.params[0])
            if name == "rz":
                accumulate(0, angle, Fraction(-1, 2))
            add(a, angle)
    terms = tuple(
        (mask, frozen) for mask, angle in sorted(phases.items()) if (frozen := _frozen_angle(angle))
    )
    return (tuple(wires), terms), work


def _conjugate(x, z, phase, name, q):
    a = q[0]
    bit = 1 << a
    xa, za = (x >> a) & 1, (z >> a) & 1
    if name == "h":
        phase += 2 * xa * za
        if xa != za:
            x ^= bit
            z ^= bit
    elif name in {"s", "sdg"}:
        phase += xa * (1 if name == "s" else -1)
        if xa:
            z ^= bit
    elif name in {"x", "y", "z"}:
        phase += 2 * (za if name == "x" else xa if name == "z" else xa ^ za)
    elif name == "cx":
        b = q[1]
        x ^= xa << b
        z ^= ((z >> b) & 1) << a
    elif name == "swap":
        b = q[1]
        if xa != (x >> b) & 1:
            x ^= bit | (1 << b)
        if za != (z >> b) & 1:
            z ^= bit | (1 << b)
    elif name != "id":
        rules = (
            {
                "cz": (("h", (q[1],)), ("cx", q), ("h", (q[1],))),
                "cy": (("sdg", (q[1],)), ("cx", q), ("s", (q[1],))),
            }
            if len(q) == 2
            else {
                "sx": (("h", q), ("s", q), ("h", q)),
                "sxdg": (("h", q), ("sdg", q), ("h", q)),
            }
        )
        if name == "iswap":
            sequence = (("swap", q), ("s", (q[0],)), ("s", (q[1],)), ("cz", q))
        else:
            sequence = rules[name]
        for gate, wires in sequence:
            x, z, phase = _conjugate(x, z, phase, gate, wires)
    return x, z, phase % 4


def _tableau(circuit, max_work):
    operations = tuple(op for op in circuit.operations if op.name != "barrier")
    if any(op.name not in _CLIFFORD or op.condition or op.blocks for op in operations):
        return None, 0
    work = 2 * circuit.num_qubits * len(operations)
    if work > max_work:
        raise ResourceError("Clifford conjugation proof exceeds max_work.")
    images = []
    for generator in range(2 * circuit.num_qubits):
        q = generator % circuit.num_qubits
        x, z, phase = (1 << q, 0, 0) if generator < circuit.num_qubits else (0, 1 << q, 0)
        for op in operations:
            x, z, phase = _conjugate(x, z, phase, op.name, op.qubits)
        images.append((x, z, phase))
    return tuple(images), work


def _word(circuit, max_terms, max_work):
    """Exact rewrites plus disjoint-wire commutation and inverse cancellation.

    A token chain on each wire is a canonical representation of a partially
    commutative word. This is deliberately not a complete universal normal form.
    """
    from .ir import Operation

    pending = list(reversed(circuit.operations))
    tokens, stacks, phase, work = [], {}, _angle(circuit.global_phase), 0

    def gate(name, qubits, params=()):
        return Operation(name, tuple(qubits), tuple(params))

    while pending:
        op = pending.pop()
        work += 1
        if work > max_work or len(tokens) + len(pending) > max_terms:
            raise ResourceError("Algebraic word proof exceeds max_work or max_terms.")
        q, p, name = op.qubits, op.params, op.name
        if name in {"barrier", "id"}:
            continue
        rules = None
        if name in {"sx", "sxdg"}:
            rules = (gate("h", q), gate("s" if name == "sx" else "sdg", q), gate("h", q))
        elif name in {"cz", "cy"}:
            a, b = q
            first, last = ("h", "h") if name == "cz" else ("sdg", "s")
            rules = (gate(first, (b,)), gate("cx", (a, b)), gate(last, (b,)))
        elif name == "swap":
            a, b = q
            rules = (gate("cx", (a, b)), gate("cx", (b, a)), gate("cx", (a, b)))
        elif name == "iswap":
            a, b = q
            rules = (gate("swap", q), gate("s", (a,)), gate("s", (b,)), gate("cz", q))
        elif name == "rx":
            rules = (gate("h", q), gate("rz", q, p), gate("h", q))
        elif name == "ry":
            rules = (gate("sdg", q), gate("h", q), gate("rz", q, p), gate("h", q), gate("s", q))
        elif name in {"rxx", "ryy", "rzz"}:
            a, b = q
            center = (gate("cx", q), gate("rz", (b,), p), gate("cx", q))
            if name == "rxx":
                outer = (gate("h", (a,)), gate("h", (b,)))
                rules = outer + center + outer
            elif name == "ryy":
                before = (gate("sdg", (a,)), gate("sdg", (b,)), gate("h", (a,)), gate("h", (b,)))
                after = (gate("h", (a,)), gate("h", (b,)), gate("s", (a,)), gate("s", (b,)))
                rules = before + center + after
            else:
                rules = center
        elif name == "u":
            rules = (gate("p", q, (p[2],)), gate("ry", q, (p[0],)), gate("p", q, (p[1],)))
        elif name in {"crx", "cry", "crz", "cp"}:
            a, b = q
            if name == "crx":
                rules = (gate("h", (b,)), gate("crz", q, p), gate("h", (b,)))
            elif name == "cp":
                half = p[0] / 2
                rules = (
                    gate("p", (a,), (half,)),
                    gate("p", (b,), (half,)),
                    gate("cx", q),
                    gate("p", (b,), (-half,)),
                    gate("cx", q),
                )
            else:
                axis = name[1:]
                half = p[0] / 2
                rules = (
                    gate(axis, (b,), (half,)),
                    gate("cx", q),
                    gate(axis, (b,), (-half,)),
                    gate("cx", q),
                )
        if rules is not None:
            pending.extend(reversed(rules))
            continue
        if op.condition or op.blocks:
            return None, work
        if name == "p":
            _accumulate(phase, _angle(p[0]), Fraction(1, 2))
            name = "rz"
        angle = _angle(p[0]) if name == "rz" else None
        if name == "rz" and not angle:
            continue
        token = name, q, angle, op.matrix
        previous = [stacks.get(wire, [])[-1] if stacks.get(wire) else None for wire in q]
        inverse = {"s": "sdg", "sdg": "s", "t": "tdg", "tdg": "t"}
        if previous and previous[0] is not None and all(index == previous[0] for index in previous):
            index = previous[0]
            old = tokens[index]
            if name == "rz" and old is not None and old[0] == "rz" and old[1] == q:
                merged = old[2]
                _accumulate(merged, angle)
                tokens[index] = ("rz", q, merged, None) if merged else None
                if not merged:
                    stacks[q[0]].pop()
                continue
            cancel = (
                old is not None
                and old[1:] == token[1:]
                and (
                    name in {"h", "x", "y", "z", "cx", "ccx", "ch", "cswap"}
                    and old[0] == name
                    or old[0] == inverse.get(name)
                )
            )
            if cancel:
                tokens[index] = None
                for wire in q:
                    stacks[wire].pop()
                continue
        index = len(tokens)
        tokens.append(token)
        for wire in q:
            stacks.setdefault(wire, []).append(index)
    # Independent token numbering must not affect the canonical word. A
    # deterministic topological sort preserves every shared-wire dependency.
    import heapq

    tokens = [
        None
        if token is None
        else (
            token[0],
            token[1],
            _frozen_angle(token[2]) if token[2] is not None else None,
            token[3],
        )
        for token in tokens
    ]
    successors, degree = {}, {}
    for stack in stacks.values():
        for index in stack:
            degree.setdefault(index, 0)
        for a, b in zip(stack, stack[1:], strict=False):
            edges = successors.setdefault(a, set())
            if b not in edges:
                edges.add(b)
                degree[b] += 1
    ready = [(repr(tokens[index]), index) for index, value in degree.items() if value == 0]
    heapq.heapify(ready)
    result = []
    while ready:
        _, index = heapq.heappop(ready)
        result.append(tokens[index])
        for next_index in successors.get(index, ()):
            degree[next_index] -= 1
            if not degree[next_index]:
                heapq.heappush(ready, (repr(tokens[next_index]), next_index))
    return (tuple(result), _frozen_angle(phase)), work


def prove_equivalence(
    source,
    target,
    *,
    phase_sensitive=True,
    max_qubits=4096,
    max_operations=100_000,
    max_terms=100_000,
    max_work=2_000_000,
    max_component_qubits=6,
):
    """Prove a supported identity for every binding; never promote a numeric check.

    Stored floats are exact binary rationals. Near multiples of pi are never
    rounded to Clifford gates. Nonlinear expression atoms require identical
    syntax. Failure is inconclusive, including differently normalized but equal
    circuits. phase_sensitive=False additionally enables projective Clifford
    proofs, which intentionally do not certify a circuit's global phase.
    Effectful programs use factored exact density kernels with initially zero
    classical storage. All final bits and the full quantum output are retained.
    Each connected component has an explicit width/term/work budget.
    """
    if type(phase_sensitive) is not bool:
        raise ValidationError("phase_sensitive must be a boolean.")
    for value, label in ((max_terms, "max_terms"), (max_work, "max_work")):
        integer(value, label, 1)
    integer(max_component_qubits, "max_component_qubits", 1)
    if max_component_qubits > 8:
        raise ValidationError("max_component_qubits must be at most 8.")
    for circuit in (source, target):
        _check_circuit_limits(circuit, max_operations=max_operations)
        if circuit.num_qubits > min(integer(max_qubits, "max_qubits", 1), 8192):
            raise ResourceError(
                "Equivalence proof exceeds its width limit or the 8192-wire kernel ceiling."
            )
    # Preflight expression expansion before serialization/fingerprinting.
    work = _expression_work(source, max_work)
    work += _expression_work(target, max_work - work)
    if work >= max_work:
        raise ResourceError("Equivalence proof exceeds max_work.")
    semantics = "quantum_instrument" if source.has_effects or target.has_effects else "unitary"
    semantics = (
        "projective_unitary" if not phase_sensitive and semantics == "unitary" else semantics
    )
    notes = []
    method, normal = "inconclusive", None
    dimensions = source.num_qubits == target.num_qubits and source.num_clbits == target.num_clbits
    if not dimensions:
        notes.append(
            Diagnostic("SEMANTIC_REGISTER_MISMATCH", "Quantum or classical register widths differ.")
        )
    elif source == target:
        method, normal = "structural_identity", (source.num_qubits, _fingerprint(source))
    elif source.has_effects or target.has_effects:
        from ._exact_instrument import instrument_form

        normal, cost = instrument_form(
            source, target, max_terms, max_work - work, max_component_qubits
        )
        work += cost
        if normal is not None:
            method = "factored_density_kernel"
    elif not source.has_effects and not target.has_effects:
        methods = [("phase_polynomial", _phase_form), ("algebraic_word", _word)]
        for name, builder in methods:
            left, cost = builder(source, max_terms, max_work - work)
            work += cost
            right, cost = builder(target, max_terms, max_work - work)
            work += cost
            if left is not None and right is not None:
                if not phase_sensitive:
                    left = (
                        (left[0], tuple(t for t in left[1] if t[0] != 0))
                        if name == "phase_polynomial"
                        else (left[0], ())
                    )
                    right = (
                        (right[0], tuple(t for t in right[1] if t[0] != 0))
                        if name == "phase_polynomial"
                        else (right[0], ())
                    )
                if left == right:
                    method, normal = name, left
                    break
        if normal is None and not phase_sensitive:
            left, cost = _tableau(source, max_work - work)
            work += cost
            right, cost = _tableau(target, max_work - work)
            work += cost
            if left is not None and left == right:
                method, normal = "clifford_conjugation", left
        if normal is None:
            from ._exact_instrument import instrument_form, operator_form

            builder = operator_form if phase_sensitive else instrument_form
            normal, cost = builder(source, target, max_terms, max_work - work, max_component_qubits)
            work += cost
            if normal is not None:
                method = (
                    "factored_operator_kernel" if phase_sensitive else "factored_projective_kernel"
                )
    if normal is None:
        notes.append(
            Diagnostic(
                "SEMANTIC_PROOF_INCONCLUSIVE",
                "Supported exact normal forms did not establish equivalence; "
                "this is not evidence of inequivalence.",
                hint="Use bounded native differential checks at explicit bindings, "
                "or retain a supported exact rewrite path.",
            )
        )
    return EquivalenceProof(
        normal is not None,
        method,
        semantics,
        _fingerprint(source),
        _fingerprint(target),
        tuple(sorted(set(source.parameters) | set(target.parameters))),
        sum(len(c.operations) for c in (source, target)),
        work,
        _digest(normal) if normal is not None else None,
        tuple(notes),
    )


@dataclass(frozen=True)
class CertifiedConversion:
    value: object
    proof: EquivalenceProof
    source_format: str
    target_format: str


def semantic_coverage():
    """Report representative export-admissibility probes without importing SDKs.

    These rows cover the displayed IR features, not every dialect or arbitrary
    native object. Runtime and algebraic verification remain independent layers.
    """
    from .conversion import diagnose
    from .ir import Circuit, ClassicalCondition
    from .parameters import Parameter

    probes = {
        "clifford_unitary": Circuit(2).h(0).sx(1).cx(0, 1),
        "symbolic_rotations": Circuit(2).ry(Parameter("theta"), 0).rzz(Parameter("phi"), 0, 1),
        "explicit_global_phase": Circuit(1).h(0).with_global_phase(Parameter("phase")),
        "terminal_measurements": Circuit(2).h(0).measure_all(),
        "measurement_feedback": Circuit(2)
        .h(0)
        .measure(0, 0)
        .if_else(ClassicalCondition((0,), 1), Circuit(2).x(1), Circuit(2).z(1)),
        "reset": Circuit(1).reset(0),
        "explicit_noise_channel": Circuit(1).amplitude_damping(0.2, 0),
        "matrix_defined_unitary": Circuit(1).unitary(((0, 1), (1, 0)), 0),
    }
    formats = ("ir", "json", "qiskit", "pennylane", "cirq", "openqasm2", "openqasm3")
    rows = []
    for feature, circuit in probes.items():
        cells = {}
        for target in formats:
            notes = diagnose(circuit, target)
            cells[target] = {
                "supported": not any(note.severity == "error" for note in notes),
                "diagnostics": [note.code for note in notes],
            }
        rows.append({"feature": feature, "targets": cells})
    return {
        "scope": "representative_IR_feature_export_admissibility",
        "native_execution_checked": False,
        "source_programs": "circuit_objects_only",
        "external_backend_context": "not_represented",
        "proof_methods": [
            "structural_identity",
            "phase_polynomial",
            "algebraic_word",
            "projective_clifford_conjugation",
            "factored_density_kernel",
            "factored_operator_kernel",
            "factored_projective_kernel",
        ],
        "rows": rows,
    }


def _certify_plan(
    plan,
    *,
    phase_sensitive=True,
    max_terms=100_000,
    max_work=2_000_000,
    max_component_qubits=6,
    source_options=None,
):
    from .compiler import rename_parameters
    from .conversion import plan_conversion

    exported = plan.export()
    restored = plan_conversion(
        exported.value,
        source=plan.target_format,
        target="ir",
        source_options=source_options,
        max_operations=plan.max_operations,
        max_depth=plan.max_depth,
    ).circuit
    if plan.parameter_map:
        restored = rename_parameters(
            restored, {value: key for key, value in plan.parameter_map.items()}
        )
    proof = prove_equivalence(
        plan.circuit,
        restored,
        phase_sensitive=phase_sensitive,
        max_operations=plan.max_operations,
        max_terms=max_terms,
        max_work=max_work,
        max_component_qubits=max_component_qubits,
    )
    proof.raise_for_errors()
    return CertifiedConversion(exported.value, proof, plan.source_format, plan.target_format)


def compile_verified(
    value,
    *,
    target,
    phase_sensitive=True,
    max_terms=100_000,
    max_work=2_000_000,
    max_component_qubits=6,
    **conversion_options,
):
    """Export only when the imported artifact has an algebraic IR certificate.

    The adapter/IR interpretation is the trust boundary. Certificates do not
    establish native SDK or physical-hardware behavior. Mutating a native
    artifact invalidates the original association; certify it again after edits.
    """
    from .conversion import plan_conversion

    return _certify_plan(
        plan_conversion(value, target=target, **conversion_options),
        phase_sensitive=phase_sensitive,
        max_terms=max_terms,
        max_work=max_work,
        max_component_qubits=max_component_qubits,
    )
