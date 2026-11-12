"""Bounded single/two-wire fusion plans containing only immutable metadata."""

from dataclasses import dataclass

import numpy as np

from ..compute.kernel import get_compute_kernel
from ..errors import ResourceError


@dataclass(frozen=True)
class FixedMatrix:
    values: tuple


@dataclass(frozen=True)
class GateGroup:
    qubits: tuple
    operations: tuple
    pieces: tuple
    parameters: tuple


@dataclass(frozen=True)
class FusionProgram:
    # Retain the source, making its identity a safe cache key until eviction.
    source: tuple
    groups: tuple


def embed_numpy(matrix, qubits, target):
    if tuple(qubits) == tuple(target):
        return matrix
    if len(qubits) == 1:
        identity = np.eye(2, dtype=matrix.dtype)
        if qubits[0] == target[0]:
            value = np.einsum("...ab,cd->...acbd", matrix, identity)
        else:
            value = np.einsum("ab,...cd->...acbd", identity, matrix)
        return value.reshape(matrix.shape[:-2] + (4, 4))
    rank = matrix.ndim - 2
    return (
        matrix.reshape(matrix.shape[:-2] + (2, 2, 2, 2))
        .transpose(tuple(range(rank)) + (rank + 1, rank, rank + 3, rank + 2))
        .reshape(matrix.shape)
    )


def _group(operations):
    from .dense import _gate_matrix, _TensorOps

    operations = tuple(operations)
    qubits = tuple(sorted(set(q for operation in operations for q in operation.qubits)))
    parameters = tuple(
        sorted({name for op in operations for p in op.params for name in p.parameters})
    )
    if len(qubits) > 2:
        return GateGroup(operations[0].qubits, operations, operations, parameters)
    pieces, combined, kernel = [], None, get_compute_kernel()
    ops = _TensorOps()

    def flush():
        nonlocal combined
        if combined is not None:
            pieces.append(FixedMatrix(tuple(tuple(complex(v) for v in row) for row in combined)))
            combined = None

    try:
        for operation in operations:
            if any(p.parameters for p in operation.params):
                flush()
                pieces.append(operation)
                continue
            matrix = _gate_matrix(
                operation.name, [p.evaluate({}) for p in operation.params], ops, operation.matrix
            )
            matrix = embed_numpy(matrix, operation.qubits, qubits)
            combined = matrix if combined is None else kernel.matmul(matrix, combined)
        flush()
    except ResourceError:
        # Fusion is optional: a small budget may admit each original gate while
        # excluding a temporary local matrix product.
        pieces = list(operations)
    return GateGroup(qubits, operations, tuple(pieces), parameters)


def fusion_program(operations, num_qubits):
    operations = operations if isinstance(operations, tuple) else tuple(operations)

    def build():
        groups, pending, wires = [], [], set()

        def flush():
            nonlocal pending, wires
            if pending:
                groups.append(_group(pending))
                pending, wires = [], set()

        for operation in operations:
            if operation.name == "barrier":
                flush()
                continue
            combined = wires.union(operation.qubits)
            if len(combined) > 2 or len(pending) >= 64:
                flush()
            if len(operation.qubits) > 2:
                groups.append(_group((operation,)))
                continue
            pending.append(operation)
            wires.update(operation.qubits)
        flush()
        return FusionProgram(operations, tuple(groups))

    return get_compute_kernel().plan("gate-fusion", (id(operations), num_qubits), build)
