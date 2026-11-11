"""Group readout workloads by structure while keeping stochastic rows distinct."""

from collections import OrderedDict
from dataclasses import replace
from threading import RLock

import numpy as np

from .compute import memory_budget
from .errors import ResourceError
from .ir import UNITARY_GATES, integer
from .parameters import Expression, Parameter, expression


def parameter_batches(circuits, max_memory_mb=256):
    """Yield (original rows, symbolic prototype, occurrence-value arrays).

    Rotation occurrences, rather than source expressions, become independent
    controls. Fixed matrices, Kraus operators and channel probabilities stay in
    the structure key. Global phases are irrelevant only for these readouts.
    """
    groups = {}
    for row, circuit in enumerate(circuits):
        signature = (
            circuit.num_qubits,
            circuit.num_clbits,
            tuple(
                (
                    path,
                    op.name,
                    op.qubits,
                    op.clbits,
                    op.matrix,
                    op.kraus,
                    op.condition,
                    len(op.params) if op.name in UNITARY_GATES else op.params,
                )
                for path, op in circuit.walk_operations()
            ),
        )
        if signature not in groups:
            groups[signature] = []
        groups[signature].append(row)
    for rows in groups.values():
        first = circuits[rows[0]]
        controls = [
            (path, j)
            for path, op in first.walk_operations()
            if op.name in UNITARY_GATES
            for j in range(len(op.params))
        ]
        if 8 * len(rows) * len(controls) > memory_budget(max_memory_mb) * 2**20:
            raise ResourceError("Batched parameter values exceed max_memory_mb.")
        prototype, bindings = first, {}
        for i, (path, op) in enumerate(first.walk_operations()):
            if op.name not in UNITARY_GATES or not op.params:
                continue
            parameters = []
            for j in range(len(op.params)):
                name = f"_oq_{i:08d}_{j:04d}"
                parameters.append(expression(Parameter(name)))
                bindings[name] = np.asarray(
                    [circuits[r].operation_at(path).params[j].constant for r in rows], dtype=float
                )
            prototype = prototype.replace_operation(path, replace(op, params=tuple(parameters)))

        def clear_phase(block):
            return replace(
                block,
                global_phase=Expression(),
                operations=tuple(
                    replace(op, blocks=tuple(clear_phase(b) for b in op.blocks))
                    if op.blocks
                    else op
                    for op in block.operations
                ),
            )

        yield tuple(rows), clear_phase(prototype), bindings


class CircuitCache:
    """Bounded, instance-local compiled topology cache; never caches readouts."""

    def __init__(self, size):
        self.size = integer(size, "cache_size")
        self._entries, self._lock = OrderedDict(), RLock()

    def get(self, key, compile_circuit):
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                return self._entries[key]
            value = compile_circuit()
            if self.size:
                self._entries[key] = value
                while len(self._entries) > self.size:
                    self._entries.popitem(last=False)
            return value

    def clear(self):
        with self._lock:
            self._entries.clear()
