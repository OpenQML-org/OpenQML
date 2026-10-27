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

from .ir import Circuit
from .observables import PauliString
from .parameters import Parameter


def _matrix_key(matrix):
    if matrix is None:
        return None
    # Normalize signed zero because it compares equal in exact Python keys.
    return tuple(
        tuple(((z.real or 0.0).hex(), (z.imag or 0.0).hex()) for z in row) for row in matrix
    )


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
        charge(len(lanes) * (len(readouts) + len(template["names"])))
        work = (
            2
            * (1 << circuit.num_qubits)
            * (len(readouts) + sum(1 << len(op.qubits) for op in circuit.operations))
        )
        digest = hashlib.sha256(json.dumps(key, separators=(",", ":")).encode()).hexdigest()
        batches.append(
            ExecutionBatch(
                digest,
                circuit,
                template["names"],
                readouts,
                tuple(lanes),
                tuple(tuple(v[0]) for v in lanes.values()),
                tuple(tuple(v[1]) for v in lanes.values()),
                template["bytes"] + 16 * len(readouts) + 8 * len(template["names"]),
                work,
            )
        )
    return tuple(batches)


def _evaluate_dense(plan, coefficients, factors, ops, shape, context):
    from .simulator import expectations

    rows = max(1, math.prod(shape))
    available = plan._memory_bytes - plan._retained_bytes_per_row * rows
    for batch in plan.dense_batches:
        width = min(
            plan.max_dense_batch,
            max(1, int(available // (batch.working_bytes_per_lane * rows))),
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
            values = expectations(
                batch.circuit,
                batch.observables,
                bindings,
                backend=ops.backend,
                like=context,
                batch_shape=local_shape,
                max_qubits=plan.max_dense_qubits,
                max_memory_mb=plan.max_memory_mb,
            )
            copy = any(
                len({r for _, r in destinations}) < len(batch.observables)
                for destinations in batch.factor_maps[start : start + len(lanes)]
            )
            for offset in range(len(lanes)):
                local = values if len(lanes) == 1 else values[offset]
                destinations = batch.factor_maps[start + offset]
                selected = {}
                for factor, readout in destinations:
                    if readout not in selected:
                        value = local[..., readout]
                        # A sparse destination must not retain the entire union
                        # of lane/readout arrays through an eager slice view.
                        if copy:
                            value = value.clone() if ops.backend == "torch" else ops.xp.copy(value)
                        selected[readout] = value
                    factors[factor] = selected[readout]
