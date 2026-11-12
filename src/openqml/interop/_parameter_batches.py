"""Group readout workloads by structure while keeping stochastic rows distinct."""

from collections.abc import Mapping
from dataclasses import replace

import numpy as np

from ..compute._cache import SingleFlightCache
from ..compute.kernel import memory_budget
from ..errors import ResourceError
from ..ir.core import UNITARY_GATES, integer
from ..ir.parameters import Expression, Parameter, expression


class _Bindings(Mapping):
    """Column views backed by one contiguous row-major parameter matrix."""

    def __init__(self, names, values):
        self.matrix = values
        self._columns = {name: values[:, i] for i, name in enumerate(names)}

    def __getitem__(self, name):
        return self._columns[name]

    def __iter__(self):
        return iter(self._columns)

    def __len__(self):
        return len(self._columns)


def binding_matrix(bindings, names=None, *, rows=0):
    """Reuse the packed buffer when a native engine uses canonical control order."""
    names = tuple(bindings) if names is None else tuple(names)
    if isinstance(bindings, _Bindings) and names == tuple(bindings):
        return bindings.matrix
    return np.stack([bindings[name] for name in names], axis=-1) if names else np.empty((rows, 0))


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
            (path, j, f"_oq_{i:08d}_{j:04d}")
            for i, (path, op) in enumerate(first.walk_operations())
            if op.name in UNITARY_GATES
            for j in range(len(op.params))
        ]
        if 8 * len(rows) * len(controls) > memory_budget(max_memory_mb) * 2**20:
            raise ResourceError("Batched parameter values exceed max_memory_mb.")
        values = np.empty((len(rows), len(controls)), dtype=float)
        for column, (path, j, _) in enumerate(controls):
            for row, original in enumerate(rows):
                values[row, column] = circuits[original].operation_at(path).params[j].constant
        names = tuple(name for _, _, name in controls)
        occurrences = iter(names)

        def prepare(block, occurrences=occurrences):
            operations = []
            for op in block.operations:
                if op.name in UNITARY_GATES and op.params:
                    op = replace(
                        op,
                        params=tuple(expression(Parameter(next(occurrences))) for _ in op.params),
                    )
                if op.blocks:
                    op = replace(op, blocks=tuple(prepare(child) for child in op.blocks))
                operations.append(op)
            return replace(
                block,
                global_phase=Expression(),
                operations=tuple(operations),
            )

        yield tuple(rows), prepare(first), _Bindings(names, values)


class CircuitCache:
    """Instance-local SDK topology cache with per-key compilation admission."""

    def __init__(self, size):
        self.size = integer(size, "cache_size")
        self._cache = SingleFlightCache(self.size)

    def get(self, key, compile_circuit):
        return self._cache.get(key, compile_circuit)

    def clear(self):
        self._cache.clear()
