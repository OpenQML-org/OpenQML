"""Sparse classical liveness and static resource planning for density execution."""

import math
from dataclasses import dataclass, replace
from functools import lru_cache

from .errors import ResourceError, ValidationError
from .ir import Circuit, integer


@dataclass(frozen=True)
class _RecordStep:
    live_after: frozenset[int]
    blocks: tuple = ()


@dataclass(frozen=True)
class _RecordPlan:
    live_in: frozenset[int]
    live_out: frozenset[int]
    steps: tuple
    peak_branches: int = 0
    output_branches: int = 0


def _topology(circuit):
    # Cache topology only; tensor values cannot affect the plan.
    return tuple(
        ("measure", op.clbits[0])
        if op.name == "measure"
        else ("if_else", op.condition, tuple(_topology(b) for b in op.blocks))
        if op.name == "if_else"
        else None
        for op in circuit.operations
    )


def _written(topology):
    bits = set()
    for node in topology:
        if node is not None:
            if node[0] == "measure":
                bits.add(node[1])
            else:
                for block in node[2]:
                    bits.update(_written(block))
    return frozenset(bits)


def _build_records(topology, live_out, possible_writes):
    live, reverse = live_out, []
    for node in reversed(topology):
        if node is None:
            reverse.append(None)
        elif node[0] == "measure":
            reverse.append(_RecordStep(live))
            live = live - {node[1]}
        else:
            blocks = tuple(_build_records(b, live, possible_writes) for b in node[2])
            reverse.append(_RecordStep(live, blocks))
            live = (frozenset(node[1].bits) & possible_writes).union(*(b.live_in for b in blocks))
    return _RecordPlan(live, live_out, tuple(reversed(reverse)))


@lru_cache(maxsize=64)
def _cached_records(topology, retain_classical, classical_bits, max_branches):
    possible = _written(topology)
    if retain_classical and (classical_bits is None or possible.issubset(classical_bits)):
        # Full-record key sets only grow; reject over-cap prefixes before snapshots.
        _, possible = _full_record_preflight(topology, {frozenset()}, max_branches)
    live = (
        (possible if classical_bits is None else frozenset(classical_bits) & possible)
        if retain_classical
        else frozenset()
    )
    records = _build_records(topology, live, possible)
    outputs, peak = _branch_plan(topology, {frozenset()}, max_branches, records)
    return replace(records, peak_branches=peak, output_branches=len(outputs))


def _record_plan(circuit, retain_classical=False, classical_bits=None, max_branches=256):
    topology = _topology(circuit)
    remaining, pending = 4096, [topology]
    while pending and remaining >= 0:
        for node in pending.pop():
            remaining -= 1
            if node is not None and node[0] == "if_else":
                remaining -= len(node[1].bits)
                pending.extend(node[2])
            if remaining < 0:
                break
    build = _cached_records if remaining >= 0 else _cached_records.__wrapped__
    return build(topology, retain_classical, classical_bits, max_branches)


def _full_record_preflight(topology, initial, max_branches):
    keys, written = set(initial), set()
    for node in topology:
        if node is None:
            continue
        if node[0] == "measure":
            bit, output = node[1], set()
            written.add(bit)
            for key in keys:
                output.update((key - {bit}, key | {bit}))
                if len(output) > max_branches:
                    raise ResourceError("Final classical record count exceeds max_branches.")
        else:
            groups, output = [set(), set()], set()
            for key in keys:
                groups[0 if node[1].matches(key) else 1].add(key)
            for i, group in enumerate(groups):
                if group:
                    child_keys, child_written = _full_record_preflight(
                        node[2][i], group, max_branches
                    )
                    output.update(child_keys)
                    written.update(child_written)
                    if len(output) > max_branches:
                        raise ResourceError("Final classical record count exceeds max_branches.")
        keys = output
    return keys, frozenset(written)


def _branch_plan(topology, initial, max_branches, records):
    keys = {key & records.live_in for key in initial}
    peak = len(keys)

    def check(values):
        if len(values) > max_branches:
            raise ResourceError("Live classical branch count exceeds max_branches.")

    check(keys)
    for node, step in zip(topology, records.steps, strict=True):
        if step is None:
            continue
        next_keys = set()
        if node[0] == "measure":
            bit = node[1]
            for key in keys:
                key = key & (step.live_after - {bit})
                next_keys.add(key)
                if bit in step.live_after:
                    next_keys.add(key | {bit})
                check(next_keys)
        else:
            groups = [set(), set()]
            for key in keys:
                branch = 0 if node[1].matches(key) else 1
                groups[branch].add(key & step.blocks[branch].live_in)
            for i, group in enumerate(groups):
                if group:
                    child_keys, child_peak = _branch_plan(
                        node[2][i], group, max_branches, step.blocks[i]
                    )
                    peak = max(peak, child_peak)
                    next_keys.update(child_keys)
                    check(next_keys)
        keys = next_keys
        peak = max(peak, len(keys))
    return keys, peak


@dataclass(frozen=True)
class DensityPlan:
    """Static plan; no quantum arrays or numerical probability pruning are used.

    Counts include both control bodies. Memory estimates cover four complex128
    buffers per batch item/live assignment, not AD graphs or total process RSS.
    """

    num_qubits: int
    num_operations: int
    peak_branches: int
    output_branches: int
    max_live_bits: int
    recorded_measurements: int
    unrecorded_measurements: int
    retains_classical: bool
    classical_bits: tuple[int, ...] | None = None

    def estimated_memory_mb(self, batch_size=1):
        integer(batch_size, "batch_size")
        try:
            return math.ldexp(
                float(max(1, batch_size) * self.peak_branches), 2 * self.num_qubits - 14
            )
        except OverflowError:
            return math.inf


def plan_density(circuit, *, retain_classical=False, classical_bits=None, max_branches=256):
    """Plan density execution without allocating quantum states.

    retain_classical keeps final storage; classical_bits selects an ordered
    record marginal. Memory estimates exclude autodiff graphs and process RSS.
    """
    if not isinstance(circuit, Circuit) or type(retain_classical) is not bool:
        raise ValidationError("Use a Circuit and boolean retain_classical.")
    integer(max_branches, "max_branches", 1)
    from ._validation import validate_classical_bits

    classical_bits = validate_classical_bits(circuit, classical_bits)
    retain_classical = retain_classical or classical_bits is not None
    records = _record_plan(circuit, retain_classical, classical_bits, max_branches)
    total = recorded = unrecorded = live_bits = 0

    def walk(body, plan):
        nonlocal total, recorded, unrecorded, live_bits
        live_bits = max(live_bits, len(plan.live_in), len(plan.live_out))
        for op, step in zip(body.operations, plan.steps, strict=True):
            total += 1
            if step is not None:
                live_bits = max(live_bits, len(step.live_after))
                if op.name == "measure":
                    recorded += op.clbits[0] in step.live_after
                    unrecorded += op.clbits[0] not in step.live_after
                for block, child in zip(op.blocks, step.blocks, strict=True):
                    walk(block, child)

    walk(circuit, records)
    return DensityPlan(
        circuit.num_qubits,
        total,
        records.peak_branches,
        records.output_branches,
        live_bits,
        recorded,
        unrecorded,
        retain_classical,
        classical_bits,
    )
