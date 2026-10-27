"""Bounded numerical checks of conversion round trips over complete input bases.

For unitary programs we compare every operator column, including global phase.
For effects we propagate every matrix unit through each retained classical
record. Linearity then covers arbitrary quantum inputs at the checked binding,
with initially zero classical storage. These are numerical adapter round-trip
checks, not independent target-runtime execution or symbolic proofs.

The standard linear-map representation is also documented at
https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.quantum_info.SuperOp
Our internal operator basis uses row-major q0-most-significant indexing.
"""

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np

from .compiler import _check_circuit_limits
from .errors import ConversionError, Diagnostic, ResourceError, ValidationError
from .ir import integer
from .parameters import real


@dataclass(frozen=True)
class ConversionCheck:
    """Complete input-basis numerical comparison at one parameter binding.

    No circuits, parameter values, quantum arrays or native objects are retained.
    A mismatch includes its input/output basis indices and classical record;
    record bits are represented by the tuple of positions holding one.
    """

    matched: bool
    target_format: str
    semantics: str
    num_qubits: int
    num_clbits: int
    checked_inputs: int
    checked_entries: int
    mismatched_entries: int
    max_absolute_error: float
    worst_input: int | None
    worst_output: int | None
    worst_record: tuple[int, ...] | None
    atol: float
    rtol: float

    @property
    def diagnostics(self):
        if self.matched:
            return ()
        return (
            Diagnostic(
                "CONVERSION_ROUNDTRIP_MISMATCH",
                f"Round trip changed {self.mismatched_entries} basis-map entries; "
                f"maximum absolute error is {self.max_absolute_error:g}.",
                f"input[{self.worst_input}].output[{self.worst_output}].record{self.worst_record}",
                hint="Inspect wire order, phase, channel conventions and classical records.",
            ),
        )

    def raise_for_errors(self):
        if not self.matched:
            raise ConversionError(self.diagnostics)

    def to_dict(self):
        return {
            "scope": "adapter_roundtrip_at_bindings",
            "quantum_inputs": "complete_basis",
            "classical_input": "all_zero",
            "wire_order": "q0_most_significant",
            "basis_indexing": (
                "row_major_matrix_units"
                if self.semantics == "quantum_instrument"
                else "basis_states"
            ),
            **asdict(self),
            "diagnostics": [asdict(d) for d in self.diagnostics],
        }

    def to_json(self, *, indent=2):
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)


def _expression_work(circuit, limit):
    """Reject expanded expression trees before binding recursively evaluates them."""
    memo, total, pending_bodies = {}, 0, [circuit]
    while pending_bodies:
        body = pending_bodies.pop()
        expressions = [body.global_phase]
        for op in body.operations:
            expressions.extend(op.params)
            pending_bodies.extend(op.blocks)
        for root in expressions:
            pending = [(root, False)]
            while pending:
                value, ready = pending.pop()
                if id(value) in memo:
                    continue
                if len(memo) + len(pending) > limit:
                    raise ResourceError("Conversion verification expressions exceed max_work.")
                if value.op is None:
                    cost, depth = 1 + 2 * len(value.terms), 0
                elif not ready:
                    pending.append((value, True))
                    pending.extend((arg, False) for arg in reversed(value.args))
                    continue
                else:
                    cost = min(limit + 1, 1 + sum(memo[id(arg)][0] for arg in value.args))
                    depth = 1 + max(memo[id(arg)][1] for arg in value.args)
                if depth > 128:
                    raise ResourceError(
                        "Conversion verification expressions exceed nesting limits."
                    )
                memo[id(value)] = cost, depth
            total += memo[id(root)][0]
            if total > limit:
                raise ResourceError("Conversion verification expressions exceed max_work.")
    return total


def _operation_work(circuit, instrument):
    total = 1
    for _, op in circuit.walk_operations():
        if op.name in {"barrier", "if_else"}:
            continue
        if instrument:
            count = len(op.kraus) if op.name == "kraus" else 4
            total += 4 * count * (1 << len(op.qubits))
        else:
            total += 1 << len(op.qubits)
    return total


def _evolve_basis(circuit, basis, instrument, records):
    from .density import _evolve
    from .simulator import _evolve_unitary, _TensorOps

    ops = _TensorOps()
    if instrument:
        return _evolve(circuit, {frozenset(): basis}, {}, ops, records)
    values = _evolve_unitary(basis, circuit.operations, {}, circuit.num_qubits, ops, None)
    phase = circuit.global_phase.evaluate({})
    return {frozenset(): values * np.exp(1j * phase)}


def _available_bytes(circuits, budget):
    local_matrix = max(
        (1 << (2 * len(op.qubits)) for c in circuits for _, op in c.walk_operations()),
        default=1,
    )
    numerator, denominator = budget.as_integer_ratio()
    return numerator * 2**20 // denominator - 32 * local_matrix


def _basis_capacity(circuits, peaks, size, budget):
    available = _available_bytes(circuits, budget)
    per_input = 128 * size * sum(peaks)
    capacity = available // per_input
    if capacity < 1:
        raise ResourceError("Conversion verification exceeds max_memory_mb.")
    return capacity


def _bounded_records(circuit, max_branches, remaining_work, unit_work, size, budget, other_peak):
    from .planning import _record_plan

    capacity = _available_bytes((circuit,), budget) // (128 * size) - other_peak
    limit = min(max_branches, remaining_work // unit_work, capacity)
    if limit < 1:
        raise ResourceError("Conversion verification exceeds max_work or max_memory_mb.")
    try:
        return _record_plan(circuit, True, None, limit)
    except ResourceError as exc:
        if limit < max_branches:
            raise ResourceError(
                "Conversion verification live records exceed max_work or max_memory_mb."
            ) from exc
        raise


def _verify_conversion(
    plan,
    bindings,
    *,
    atol,
    rtol,
    max_qubits,
    max_memory_mb,
    max_work,
    max_branches,
    basis_chunk_size,
    source_options,
):
    from .conversion import plan_conversion

    atol, rtol = real(atol), real(rtol)
    budget = real(max_memory_mb)
    if min(atol, rtol) < 0 or budget <= 0:
        raise ValidationError("Use nonnegative tolerances and a positive memory budget.")
    max_qubits = integer(max_qubits, "max_qubits", 1)
    max_work = integer(max_work, "max_work", 1)
    max_branches = integer(max_branches, "max_branches", 1)
    chunk = integer(basis_chunk_size, "basis_chunk_size", 1)
    circuit = plan.circuit
    if circuit.num_qubits > max_qubits:
        raise ResourceError("Conversion verification exceeds max_qubits.")
    _check_circuit_limits(circuit, max_operations=plan.max_operations, max_depth=plan.max_depth)
    work = _expression_work(circuit, max_work)
    if bindings is not None:
        if not isinstance(bindings, Mapping) or set(bindings) != set(circuit.parameters):
            raise ValidationError("Verification requires all source parameter bindings.")
        values = {name: real(value) for name, value in bindings.items()}
        plan = plan_conversion(
            circuit,
            target=plan.target_format,
            bindings=values,
            max_operations=plan.max_operations,
            max_depth=plan.max_depth,
            max_output_statements=plan.max_output_statements,
        )
        circuit = plan.circuit
    if circuit.parameters:
        raise ValidationError("Bind all parameters before conversion verification.")
    plan.raise_for_errors()
    prepared = plan.prepared_circuit or circuit
    _check_circuit_limits(prepared, max_operations=plan.max_operations, max_depth=plan.max_depth)
    work += _expression_work(prepared, max_work - work)
    # Bound the source before native export or allocating basis arrays. A caller
    # raising max_qubits still cannot cause unbounded integer exponentiation.
    instrument = circuit.has_effects
    exponent = (4 if instrument else 2) * circuit.num_qubits
    if exponent >= max_work.bit_length():
        raise ResourceError("Conversion verification exceeds max_work.")
    size = 1 << ((2 if instrument else 1) * circuit.num_qubits)
    source_work = (1 << exponent) * _operation_work(circuit, instrument)
    source_records = (
        _bounded_records(circuit, max_branches, max_work - work, source_work, size, budget, 1)
        if instrument
        else None
    )
    source_peak = source_records.peak_branches if instrument else 1
    work += source_work * source_peak
    if work > max_work:
        raise ResourceError("Conversion verification exceeds max_work.")
    _basis_capacity((circuit,), (source_peak, source_peak), size, budget)
    exported = plan.export()
    restored = plan_conversion(
        exported.value,
        source=plan.target_format,
        target="ir",
        source_options=source_options,
        max_operations=plan.max_operations,
        max_depth=plan.max_depth,
    ).circuit
    if (restored.num_qubits, restored.num_clbits) != (circuit.num_qubits, circuit.num_clbits):
        raise ValidationError("Conversion round trip changed quantum or classical register width.")
    if not instrument and restored.has_effects:
        raise ValidationError("Conversion round trip introduced effects into a unitary program.")
    if restored.parameters:
        raise ValidationError("Conversion round trip left unbound parameters.")
    work += _expression_work(restored, max_work - work)
    target_work = (1 << exponent) * _operation_work(restored, instrument)
    target_records = (
        _bounded_records(
            restored, max_branches, max_work - work, target_work, size, budget, source_peak
        )
        if instrument
        else None
    )
    target_peak = target_records.peak_branches if instrument else 1
    work += target_work * target_peak
    if work > max_work:
        raise ResourceError("Conversion verification exceeds max_work.")
    # Stream input basis chunks. Source outputs remain live during target
    # evolution; eight buffers per peak branch bound primal arrays conservatively.
    capacity = _basis_capacity((circuit, restored), (source_peak, target_peak), size, budget)
    chunk = min(chunk, capacity, size)
    checked = mismatched = 0
    maximum, worst_input, worst_output, worst_record = 0.0, None, None, None
    for start in range(0, size, chunk):
        rows = min(chunk, size - start)
        basis = np.zeros((rows, size), dtype=complex)
        basis[np.arange(rows), np.arange(start, start + rows)] = 1
        expected = _evolve_basis(circuit, basis, instrument, source_records)
        actual = _evolve_basis(restored, basis, instrument, target_records)
        for record in sorted(expected.keys() | actual.keys(), key=lambda k: tuple(sorted(k))):
            a, b = actual.get(record), expected.get(record)
            if a is None:
                a = np.zeros_like(b)
            if b is None:
                b = np.zeros_like(a)
            errors = np.abs(a - b)
            if not np.isfinite(errors).all():
                raise ValidationError("Conversion verification produced nonfinite values.")
            checked += errors.size
            mismatched += int(np.count_nonzero(~np.isclose(a, b, atol=atol, rtol=rtol)))
            index = np.unravel_index(int(np.argmax(errors)), errors.shape)
            error = float(errors[index])
            if error > maximum:
                maximum = error
                worst_input, worst_output = start + int(index[0]), int(index[1])
                worst_record = tuple(sorted(record))
            del a, b, errors
        del expected, actual, basis
    return ConversionCheck(
        not mismatched,
        plan.target_format,
        "quantum_instrument" if instrument else "unitary_including_global_phase",
        circuit.num_qubits,
        circuit.num_clbits,
        size,
        checked,
        mismatched,
        maximum,
        worst_input,
        worst_output,
        worst_record,
        atol,
        rtol,
    )
