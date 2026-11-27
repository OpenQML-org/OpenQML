"""Bounded local-matrix and diagonal fusion plans for statevector evolution."""

from dataclasses import dataclass
from itertools import groupby
from typing import NamedTuple

import numpy as np

from ..compute.kernel import _torch_compiling, get_compute_kernel
from ..errors import ResourceError


@dataclass(frozen=True)
class FixedMatrix:
    values: tuple


class GateGroup(NamedTuple):
    qubits: tuple
    operations: tuple
    pieces: tuple
    parameters: tuple


class FusionProgram(NamedTuple):
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


def embed_native(matrix, qubits, target, ops):
    """Embed a local differentiable matrix using only native tensor views."""
    if tuple(qubits) == tuple(target):
        return matrix
    if len(qubits) == 1:
        identity = ops.array(((1, 0), (0, 1)))
        value = (
            matrix[..., :, None, :, None] * identity[None, :, None, :]
            if qubits[0] == target[0]
            else identity[:, None, :, None] * matrix[..., None, :, None, :]
        )
        return value.reshape(tuple(matrix.shape[:-2]) + (4, 4))
    rank = matrix.ndim - 2
    return ops.transpose(
        matrix.reshape(tuple(matrix.shape[:-2]) + (2, 2, 2, 2)),
        tuple(range(rank)) + (rank + 1, rank, rank + 3, rank + 2),
    ).reshape(matrix.shape)


def _group(operations, qubits, *, fold_constants=True):
    from .dense import _gate_matrix, _TensorOps

    operations = tuple(operations)
    qubits = tuple(qubits)
    if not fold_constants:
        # Native compilers fold these small constant products themselves. Keep
        # Python cache locks and NumPy evaluation outside the captured graph.
        return GateGroup(qubits, operations, operations, ())
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


def _order_window(gates):
    """Choose ready gates sharing the current local operator, preserving wire order."""
    if len(gates) < 3 or len({q for op in gates for q in op.qubits}) <= 2:
        return gates
    previous, successors, indegree, ready = {}, [[] for _ in gates], [], set()
    supports = [frozenset(op.qubits) for op in gates]
    for i, support in enumerate(supports):
        parents = {previous[q] for q in support if q in previous}
        indegree.append(len(parents))
        if not parents:
            ready.add(i)
        for parent in parents:
            successors[parent].append(i)
        for q in support:
            previous[q] = i
    ordered, active = [], frozenset()
    while ready:
        # Prefer no expansion, then one additional wire. Source order resolves ties.
        index = min(ready, key=lambda i: (len(active | supports[i]), i)) if active else min(ready)
        if len(active | supports[index]) > 2:
            index, active = min(ready), frozenset()
        active = active | supports[index]
        ordered.append(gates[index])
        ready.remove(index)
        for child in successors[index]:
            indegree[child] -= 1
            if not indegree[child]:
                ready.add(child)
    return ordered


def _dependency_order(operations):
    window = []
    for operation in operations:
        if operation.name == "barrier" or len(operation.qubits) > 2:
            yield from _order_window(window)
            window.clear()
            yield operation
        else:
            window.append(operation)
            if len(window) == 64:
                yield from _order_window(window)
                window.clear()
    yield from _order_window(window)


def _program(operations, *, fold_constants=True):
    groups, pending, wires = [], [], []

    def flush():
        nonlocal pending, wires
        if pending:
            groups.append(_group(pending, sorted(wires), fold_constants=fold_constants))
            pending, wires = [], []

    for operation in _dependency_order(operations) if fold_constants else operations:
        if operation.name == "barrier":
            flush()
            continue
        combined = wires + [q for q in operation.qubits if q not in wires]
        if len(combined) > 2 or len(pending) >= 64:
            flush()
        if len(operation.qubits) > 2:
            groups.append(_group((operation,), operation.qubits, fold_constants=fold_constants))
            continue
        pending.append(operation)
        for qubit in operation.qubits:
            if qubit not in wires:
                wires.append(qubit)
    flush()
    return FusionProgram(operations, tuple(groups))


class DiagonalGroup(NamedTuple):
    qubits: tuple
    operations: tuple
    parameters: tuple


_DIAGONAL = frozenset("z s sdg t tdg p rz cz cp crz rzz".split())


def _diagonal_group(wires, operations):
    parameters = {p for op in operations for expr in op.params for p in expr.parameters}
    return DiagonalGroup(tuple(sorted(wires)), tuple(operations), tuple(sorted(parameters)))


def _phase_program(operations):
    # Keep the ordinary scheduler for short runs. Longer diagonal runs commute
    # and need only one amplitude multiply per block, including distant wires.
    groups, pending = [], []
    for diagonal, run in groupby(operations, key=lambda op: op.name in _DIAGONAL):
        gates = tuple(run)
        if not diagonal or len(gates) < 4 or len({q for op in gates for q in op.qubits}) <= 2:
            pending.extend(gates)
            continue
        if pending:
            groups.extend(_program(tuple(pending)).groups)
            pending.clear()
        wires, block = set(), []
        for op in gates:
            if len(wires.union(op.qubits)) > 8 or len(block) >= 256:
                groups.append(_diagonal_group(wires, block))
                wires, block = set(), []
            wires.update(op.qubits)
            block.append(op)
        groups.append(_diagonal_group(wires, block))
    if pending:
        groups.extend(_program(tuple(pending)).groups)
    return FusionProgram(operations, tuple(groups))


def fusion_program(operations, num_qubits, *, phase_blocks=False):
    operations = operations if isinstance(operations, tuple) else tuple(operations)
    if _torch_compiling():
        return _program(operations, fold_constants=False)

    return get_compute_kernel().plan(
        "gate-fusion",
        (id(operations), num_qubits, phase_blocks),
        lambda: _phase_program(operations) if phase_blocks else _program(operations),
    )
