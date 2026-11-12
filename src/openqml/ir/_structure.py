"""Sparse quantum/classical dependency graphs and semantics-preserving wire maps."""

from dataclasses import replace

from .core import Circuit, ClassicalCondition


def operation_resources(op):
    if op.name == "barrier":
        return set()
    resources = {(0, q) for q in op.qubits} | {(1, c) for c in op.clbits}
    if op.condition is not None:
        resources.update((1, c) for c in op.condition.bits)
        for block in op.blocks:
            for child in block.operations:
                resources.update(operation_resources(child))
    return resources


def connected_components(circuits, charge=lambda work: None):
    """Include classical writers/readers and both atomic conditional bodies."""
    parent = {}

    def find(resource):
        parent.setdefault(resource, resource)
        root = resource
        while parent[root] != root:
            root = parent[root]
        while parent[resource] != resource:
            previous = parent[resource]
            parent[resource] = root
            resource = previous
        return root

    for circuit in circuits:
        for op in circuit.operations:
            resources = sorted(operation_resources(op))
            charge(1 + len(resources))
            if resources:
                root = find(resources[0])
                for resource in resources[1:]:
                    parent[find(resource)] = root
    groups = {}
    for resource in parent:
        groups.setdefault(find(resource), []).append(resource)
    return tuple(sorted(tuple(sorted(group)) for group in groups.values()))


def remap_circuit(circuit, quantum, classical, *, operations=None, keep_phase=True):
    """No array or list is indexed by the declared source register width."""
    width = max(1, len(quantum))
    mapped = []
    for op in circuit.operations if operations is None else operations:
        if op.name == "barrier":
            continue
        condition = (
            None
            if op.condition is None
            else ClassicalCondition(
                tuple(classical[c] for c in op.condition.bits), op.condition.value
            )
        )
        mapped.append(
            replace(
                op,
                qubits=tuple(quantum[q] for q in op.qubits),
                clbits=tuple(classical[c] for c in op.clbits),
                condition=condition,
                blocks=tuple(
                    remap_circuit(b, quantum, classical, keep_phase=keep_phase) for b in op.blocks
                ),
            )
        )
    return Circuit(width, tuple(mapped), len(classical), circuit.global_phase if keep_phase else 0)


def compact_pair(source, target, charge):
    groups = connected_components((source, target), charge)
    resources = {resource for group in groups for resource in group}
    quantum = {q: i for i, q in enumerate(sorted(q for kind, q in resources if kind == 0))}
    classical = {c: i for i, c in enumerate(sorted(c for kind, c in resources if kind == 1))}
    return remap_circuit(source, quantum, classical), remap_circuit(target, quantum, classical)


def split_circuit(circuit, charge=lambda work: None):
    groups = connected_components((circuit,), charge)
    owners = {resource: i for i, group in enumerate(groups) for resource in group}
    operations = [[] for _ in groups]
    for op in circuit.operations:
        resources = operation_resources(op)
        charge(1 + len(resources))
        if resources:
            operations[owners[next(iter(resources))]].append(op)
    result = []
    for group, ops in zip(groups, operations, strict=True):
        wires = tuple(q for kind, q in group if kind == 0)
        bits = tuple(c for kind, c in group if kind == 1)
        if wires:
            child = remap_circuit(
                circuit,
                {q: i for i, q in enumerate(wires)},
                {c: i for i, c in enumerate(bits)},
                operations=ops,
                keep_phase=False,
            )
            result.append((wires, bits, child))
    return tuple(result)
