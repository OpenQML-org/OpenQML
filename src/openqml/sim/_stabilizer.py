"""Exact Clifford readouts and terminal sampling on an all-zero input."""

from dataclasses import dataclass, replace

import numpy as np

from ..compute.kernel import get_compute_kernel
from ..errors import ResourceError, ValidationError
from ..ir._clifford import CLIFFORD_GATES, PackedPaulis, packed_rows
from ..ir.observables import Hamiltonian
from ..runtime.backends import BackendCapabilities


class LayoutTooLarge(ResourceError):
    """A preflight rejection before sampling begins."""


def supported(circuit, *, sampling=False):
    measured = False
    for op in circuit.operations:
        if op.condition or op.blocks:
            return False
        if op.name == "measure" and sampling:
            measured = True
        elif op.name != "barrier" and (measured or op.name not in CLIFFORD_GATES):
            return False
    return True


def readout_workspace(width, terms):
    # Two Python bigint columns per wire (30 bits per digit on 64-bit CPython).
    return 1024 + width * (256 + 8 * ((terms + 29) // 30)) + 128 * terms


def readouts(circuit, observables):
    tableau = PackedPaulis()
    ends, coefficients = [], []
    for observable in observables:
        terms = observable.terms if isinstance(observable, Hamiltonian) else (observable,)
        for term in terms:
            bit = 1 << len(coefficients)
            coefficients.append(term.coefficient)
            for q, letter in enumerate(term.paulis):
                if letter != "I":
                    x, z = tableau.columns.get(q, (0, 0))
                    tableau.columns[q] = (
                        x | (bit if letter in "XY" else 0),
                        z | (bit if letter in "YZ" else 0),
                    )
        ends.append(len(coefficients))
    tableau.propagate(circuit.operations, inverse=True)
    flips = 0
    for x, _ in tableau.columns.values():
        flips |= x
    # |0> has zero expectation for any Pauli that flips an input bit.
    result, start = [], 0
    for end in ends:
        value = 0.0
        for i in range(start, end):
            if not (flips >> i) & 1:
                value += -coefficients[i] if (tableau.sign >> i) & 1 else coefficients[i]
        result.append(value)
        start = end
    return tuple(result)


def _affine_support(circuit, wires):
    """Return each measured bit as (parity mask, offset) on uniform free bits."""
    tableau = PackedPaulis()
    tableau.columns = {q: (0, 1 << i) for i, q in enumerate(wires)}
    tableau.propagate(circuit.operations, inverse=True)
    pivots, expressions = {}, {}
    for wire, (x, z, phase) in zip(wires, tableau.rows(tableau.columns, len(wires)), strict=True):
        mask = 0
        # Eliminate the X support of commuting U† Z_q U. A residual Z-only
        # word is deterministic on |0>; each new X pivot is an independent bit.
        while x:
            lead = x.bit_length()
            pivot = pivots.get(lead)
            if pivot is None:
                bit = 1 << len(pivots)
                pivots[lead] = x, z, phase, bit
                mask ^= bit
                phase = 0
                break
            px, pz, pp, bit = pivot
            phase = (phase + pp + 2 * (z & px).bit_count()) % 4
            x ^= px
            z ^= pz
            mask ^= bit
        expressions[wire] = mask, phase // 2
    return expressions, len(pivots)


def _draw(expressions, rank, shots, seed, budget):
    width = len(expressions)
    if rank == 0:
        return {"".join(str(offset) for _, offset in expressions): shots}
    rng = np.random.default_rng(seed)
    # Draw a histogram directly when the support is small relative to shots.
    if rank < min(shots.bit_length(), 17) and (32 << rank) + width < budget:
        size = 1 << rank
        frequencies = rng.multinomial(shots, np.full(size, 1 / size))
        counts = {}
        for i in np.flatnonzero(frequencies):
            if (len(counts) + 1) * (128 + width) + 32 * size > budget:
                raise ResourceError("Stabilizer count strings exceed max_memory_mb.")
            key = "".join(
                str(((int(i) & mask).bit_count() & 1) ^ offset) for mask, offset in expressions
            )
            counts[key] = int(frequencies[i])
        return counts

    counts = {}
    remaining = shots
    while remaining:
        # Pack independent shots along each column. Only a bounded strip is
        # transposed to form the returned q0-/c0-first strings.
        free_bytes = budget - len(counts) * (128 + width)
        chunk = min(256, remaining, free_bytes // (160 + 5 * width + rank))
        if chunk < 1:
            raise ResourceError("Stabilizer sampling workspace exceeds max_memory_mb.")
        octets = (chunk + 7) // 8
        bits = rng.bytes(rank * octets)
        variables = [
            int.from_bytes(bits[i : i + octets], "little") for i in range(0, len(bits), octets)
        ]
        ones = (1 << chunk) - 1
        columns = []
        for mask, offset in reversed(expressions):
            column = ones if offset else 0
            while mask:
                bit = mask & -mask
                column ^= variables[bit.bit_length() - 1]
                mask ^= bit
            columns.append(column & ones)
        for value in packed_rows(columns, chunk):
            key = format(value, f"0{width}b")
            if (
                key not in counts
                and (len(counts) + 1) * (128 + width) + chunk * (32 + 4 * width + rank) > budget
            ):
                raise ResourceError("Stabilizer count strings exceed max_memory_mb.")
            counts[key] = counts.get(key, 0) + 1
        remaining -= chunk
    return counts


def sample_counts(circuit, shots, options, classical_bits):
    """Inputs have been bound and validated before dispatch."""
    if options.max_qubits is not None and circuit.num_qubits > options.max_qubits:
        raise ResourceError("Circuit exceeds the requested qubit limit.")
    budget = get_compute_kernel()._prepare_budget(options.max_memory_mb).max_bytes
    if classical_bits == ():
        return {"": shots}
    measured = circuit.has_measurements
    width = (
        (circuit.num_clbits if measured else circuit.num_qubits)
        if classical_bits is None
        else len(classical_bits)
    )
    if 1024 + 128 * width > budget:
        raise LayoutTooLarge("Stabilizer output layout exceeds max_memory_mb.")
    if measured:
        selected = None if classical_bits is None else set(classical_bits)
        writes = {
            op.clbits[0]: op.qubits[0]
            for op in circuit.operations
            if op.name == "measure" and (selected is None or op.clbits[0] in selected)
        }
        outputs = tuple(
            writes.get(c) for c in (range(width) if selected is None else classical_bits)
        )
    else:
        outputs = range(width)
    # Idle wires start in |0>. For a projection, expand backwards only through
    # gates touching the requested measurement cone before sizing its tableau.
    needed = set(writes.values()) if measured else None
    active_wires = set()
    for op in reversed(circuit.operations):
        if op.name not in CLIFFORD_GATES or op.name == "id":
            continue
        if needed is not None:
            if needed.isdisjoint(op.qubits):
                continue
            needed.update(op.qubits)
        active_wires.update(op.qubits)
        if 256 * len(active_wires) + 128 * width + 1024 > budget:
            raise LayoutTooLarge("Stabilizer wire bookkeeping exceeds max_memory_mb.")
    if not active_wires:
        return {"0" * width: shots}
    wires = (
        tuple(dict.fromkeys(q for q in outputs if q in active_wires))
        if measured
        else tuple(sorted(active_wires))
    )
    lanes, active = len(wires), len(active_wires)
    workspace = (
        1024
        + 128 * width
        + 256 * lanes
        + 256 * active
        + (active + lanes) * (16 + (lanes + 1) // 2)
        + 2 * min(256, lanes) * active
    )
    if workspace >= budget:
        raise LayoutTooLarge("Packed stabilizer workspace exceeds max_memory_mb.")
    forms, rank = _affine_support(circuit, wires)
    expressions = [forms.get(q, (0, 0)) for q in outputs]
    return _draw(expressions, rank, shots, options.seed, budget - workspace)


@dataclass(frozen=True)
class StabilizerBackend:
    name: str = "stabilizer"
    capabilities = BackendCapabilities(
        gates=CLIFFORD_GATES,
        native_expectations=True,
        deterministic_expectations=True,
        exact_expectations=True,
        native_sampling=True,
    )

    def expectations_batch(self, circuits, observables, *, options):
        from .adaptive import plan_execution

        return np.asarray(
            [
                plan_execution(
                    c,
                    observables,
                    method="stabilizer",
                    max_qubits=options.max_qubits,
                    max_memory_mb=options.max_memory_mb,
                ).evaluate()
                for c in circuits
            ]
        )

    def sample_batch(self, circuits, shots, *, options):
        counts = []
        for i, circuit in enumerate(circuits):
            if not supported(circuit, sampling=True):
                raise ValidationError(
                    "Stabilizer sampling requires Clifford gates and terminal measurements."
                )
            local = options if options.seed is None else replace(options, seed=options.seed + i)
            counts.append(sample_counts(circuit, shots, local, None))
        return counts
