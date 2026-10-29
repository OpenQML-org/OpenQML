"""Exact dense template interning and bounded native execution lanes.

Keys include ordered operands and exact matrix entries. Parameter expressions
are already interned in the plan's shared scalar graph; equal slot sequences
identify equal quantum functions. No values, tolerances or tensor caches are
used to decide reuse.
"""

import hashlib
import json
import math
from dataclasses import dataclass, field, replace

import numpy as np

from .ir import Circuit
from .observables import PauliString
from .parameters import Parameter


def _matrix_key(matrix):
    if matrix is None:
        return None
    # Compare complete canonical payloads, never just their digest. A compact
    # binary snapshot avoids millions of per-entry Python strings for matrices.
    values = np.array(matrix, dtype="<c16", copy=True, order="C")
    parts = values.view("<f8")
    parts[parts == 0] = 0.0
    return values.tobytes()


def _template_id(key):
    digest = hashlib.sha256(b"openqml.dense_template\0")
    digest.update(str(key[0]).encode() + b"\0")
    for name, qubits, params, matrix in key[1]:
        header = json.dumps((name, qubits, params), separators=(",", ":")).encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(b"\0" if matrix is None else b"\1" + len(matrix).to_bytes(8, "big"))
        if matrix is not None:
            digest.update(matrix)
    return digest.hexdigest()


@dataclass(frozen=True)
class ReadoutRoute:
    """One local Pauli readout, only the lanes requesting it, and source factors."""

    readout_index: int
    lanes: tuple[int, ...]
    factor_indices: tuple[tuple[int, ...], ...]

    def to_dict(self):
        return {
            "readout_index": self.readout_index,
            "lanes": list(self.lanes),
            "factor_indices": [list(v) for v in self.factor_indices],
        }


@dataclass(frozen=True, eq=False)
class ExecutionBatch:
    """One dense template with distinct parameter lanes and factor destinations.

    Identical parameter lanes execute once even across different source wires.
    Other lanes share native batched gate applications, bounded by the plan's
    max_dense_batch and the runtime working budget. Source groups remain in
    ExpectationPlan.groups. This template is not a replacement source program.
    template_id covers ordered gates, operands and exact matrix payloads;
    parameter_slots and readouts complete the execution identity.
    """

    template_id: str
    circuit: Circuit = field(repr=False)
    names: tuple[str, ...]
    observables: tuple[PauliString, ...]
    parameter_slots: tuple[tuple[int, ...], ...]
    source_groups: tuple[tuple[int, ...], ...]
    factor_maps: tuple[tuple[tuple[int, int], ...], ...]
    working_bytes_per_lane: int
    estimated_work_per_lane: int
    readout_routes: tuple[ReadoutRoute, ...] = ()

    @property
    def estimated_work(self):
        gate_work = len(self.parameter_slots) * sum(
            1 << len(op.qubits) for op in self.circuit.operations
        )
        queries = sum(len(route.lanes) for route in self.readout_routes)
        return 2 * (1 << self.circuit.num_qubits) * (gate_work + queries)

    def to_dict(self):
        return {
            "template_id": self.template_id,
            "num_qubits": self.circuit.num_qubits,
            "num_operations": len(self.circuit.operations),
            "readouts": [p.paulis for p in self.observables],
            "parameter_slots": [list(p) for p in self.parameter_slots],
            "source_groups": [list(p) for p in self.source_groups],
            "factor_maps": [[list(p) for p in lane] for lane in self.factor_maps],
            "working_bytes_per_lane": self.working_bytes_per_lane,
            "estimated_work_per_lane": self.estimated_work_per_lane,
            "estimated_work": self.estimated_work,
            "readout_routes": [r.to_dict() for r in self.readout_routes],
            "requested_readout_pairs": sum(len(r.lanes) for r in self.readout_routes),
            "unrouted_readout_pairs": len(self.parameter_slots) * len(self.observables),
        }


def _schedule_dense(groups, charge):
    templates = {}
    for group_index, group in enumerate(groups):
        program = group.dense_program
        if program is None:
            continue
        circuit = program.circuit
        charge(
            len(circuit.operations)
            + len(group.observables)
            + sum(len(op.params) + sum(map(len, op.matrix or ())) for op in circuit.operations)
        )
        key = (
            circuit.num_qubits,
            tuple(
                (op.name, op.qubits, len(op.params), _matrix_key(op.matrix))
                for op in circuit.operations
            ),
        )
        if key not in templates:
            names, operations = [], []
            for op in circuit.operations:
                params = []
                for _ in op.params:
                    name = f"_execution_angle_{len(names)}"
                    names.append(name)
                    params.append(Parameter(name))
                operations.append(replace(op, params=tuple(params)) if params else op)
            templates[key] = {
                "circuit": Circuit(circuit.num_qubits, tuple(operations)),
                "names": tuple(names),
                "readouts": {},
                "lanes": {},
                "bytes": 0,
            }
        template = templates[key]
        positions = dict(zip(program.names, group.program_indices, strict=True))
        slots = tuple(positions[p.terms[0][0]] for op in circuit.operations for p in op.params)
        sources, destinations = template["lanes"].setdefault(slots, ([], []))
        sources.append(group_index)
        for ob, factor in zip(group.observables, group.factor_indices, strict=True):
            readout = template["readouts"].setdefault(ob.paulis, len(template["readouts"]))
            destinations.append((factor, readout))
        template["bytes"] = max(template["bytes"], group.working_bytes_per_row)
    batches = []
    for key, template in templates.items():
        readouts = tuple(PauliString(p) for p in template["readouts"])
        circuit, lanes = template["circuit"], template["lanes"]
        charge(len(lanes) * len(template["names"]) + sum(len(v[1]) for v in lanes.values()))
        routes = {}
        for lane, (_, destinations) in enumerate(lanes.values()):
            for factor, readout in destinations:
                routes.setdefault(readout, {}).setdefault(lane, []).append(factor)
        routes = tuple(
            ReadoutRoute(readout, tuple(values), tuple(tuple(v) for v in values.values()))
            for readout, values in routes.items()
        )
        work = (
            2
            * (1 << circuit.num_qubits)
            * (len(readouts) + sum(1 << len(op.qubits) for op in circuit.operations))
        )
        batches.append(
            ExecutionBatch(
                _template_id(key),
                circuit,
                template["names"],
                readouts,
                tuple(lanes),
                tuple(tuple(v[0]) for v in lanes.values()),
                tuple(tuple(v[1]) for v in lanes.values()),
                template["bytes"]
                + 32 * (1 << circuit.num_qubits)
                + 16 * len(readouts)
                + 8 * len(template["names"]),
                work,
                routes,
            )
        )
    return tuple(batches)


def _evaluate_dense(plan, coefficients, factors, ops, shape):
    from .simulator import _evolve_unitary, _expectation_from_state, _initial_state

    rows = max(1, math.prod(shape))
    available = plan._memory_bytes - plan._retained_bytes_per_row * rows
    for batch in plan.dense_batches:
        width = min(
            plan.max_dense_batch,
            max(1, int(available // (batch.working_bytes_per_lane * rows))),
        )
        # Bucket static destinations once, so small runtime packs do not scan
        # every unrelated readout route. No bindings or tensors enter metadata.
        packed = [{} for _ in range(0, len(batch.parameter_slots), width)]
        for route in batch.readout_routes:
            for lane, destinations in zip(route.lanes, route.factor_indices, strict=True):
                packed[lane // width].setdefault(route.readout_index, []).append(
                    (lane % width, destinations)
                )
        for start in range(0, len(batch.parameter_slots), width):
            lanes = batch.parameter_slots[start : start + width]
            # Keep a singleton on the ordinary path to avoid unnecessary stacks.
            if len(lanes) == 1:
                bindings = {
                    name: coefficients[..., slot]
                    for name, slot in zip(batch.names, lanes[0], strict=True)
                }
                local_shape = shape
            else:
                bindings = {}
                for column, name in enumerate(batch.names):
                    arrays = tuple(coefficients[..., lane[column]] for lane in lanes)
                    bindings[name] = (
                        ops.xp.stack(arrays, dim=0)
                        if ops.backend == "torch"
                        else ops.xp.stack(arrays, axis=0)
                    )
                local_shape = (len(lanes),) + shape
            # Prepared templates have zero phase and validated local operands.
            # The plan preflights rows and every pack against its working budget.
            state = _evolve_unitary(
                _initial_state(batch.circuit.num_qubits, local_shape, ops),
                batch.circuit.operations,
                bindings,
                batch.circuit.num_qubits,
                ops,
                None,
            )
            for readout, active in packed[start // width].items():
                selected = state
                single = len(active) == 1
                if single and len(lanes) > 1:
                    selected = state[active[0][0]]
                elif len(active) < len(lanes):
                    indices = tuple(lane for lane, _ in active)
                    selected = (
                        ops.xp.index_select(
                            state, 0, ops.xp.tensor(indices, dtype=ops.xp.long, device=ops.device)
                        )
                        if ops.backend == "torch"
                        else ops.xp.take(state, ops.xp.asarray(indices), axis=0)
                    )
                values = _expectation_from_state(
                    selected, batch.observables[readout], batch.circuit.num_qubits, ops
                )
                for offset, (_, destinations) in enumerate(active):
                    value = values if single else values[offset]
                    for factor in destinations:
                        factors[factor] = value
                del selected, values
            del state, bindings
