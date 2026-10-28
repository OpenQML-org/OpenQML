"""Batched numeric execution, with native engine dispatch and vectorized NumPy fallback."""

from dataclasses import replace

import numpy as np

from .backends import ExecutionOptions, get_backend
from .errors import BackendError, ResourceError, ValidationError
from .ir import Circuit
from .observables import Hamiltonian, PauliString
from .parameters import Parameter


def _bound(circuits, bindings):
    if isinstance(circuits, Circuit):
        rows = [{}] if bindings is None else list(bindings)
        return [circuits.bind(row) for row in rows]
    values = list(circuits)
    if any(not isinstance(c, Circuit) for c in values):
        raise ValidationError("Circuit batches must contain Circuit objects.")
    rows = [{} for _ in values] if bindings is None else list(bindings)
    if len(values) != len(rows):
        raise ValidationError("Circuit/binding batch lengths differ.")
    return [c.bind(row) for c, row in zip(values, rows, strict=True)]


def _limits(circuits, device, max_qubits):
    for circuit in circuits:
        limit = device.capabilities.max_qubits
        if (limit is not None and circuit.num_qubits > limit) or (
            max_qubits is not None and circuit.num_qubits > max_qubits
        ):
            raise ResourceError(f"Circuit exceeds backend/requested qubit limit for {device.name}.")


def _vectorized_states(circuits, options):
    from .parameters import expression
    from .simulator import statevector

    first = circuits[0]

    def signature(c):
        return (
            c.num_qubits,
            tuple(
                (op.name, op.qubits, op.clbits, op.matrix, len(op.params)) for op in c.operations
            ),
        )

    if any(signature(c) != signature(first) for c in circuits[1:]):
        return [
            statevector(c, max_qubits=options.max_qubits, max_memory_mb=options.max_memory_mb)
            for c in circuits
        ]
    operations, bindings = [], {}
    for i, op in enumerate(first.operations):
        params = []
        for j in range(len(op.params)):
            name = f"_batch_{i}_{j}"
            params.append(expression(Parameter(name)))
            bindings[name] = np.array([c.operations[i].params[j].constant for c in circuits])
        operations.append(replace(op, params=tuple(params)) if params else op)
    phase_name = "_batch_global_phase"
    bindings[phase_name] = np.array([c.global_phase.constant for c in circuits])
    prototype = replace(
        first, operations=tuple(operations), global_phase=expression(Parameter(phase_name))
    )
    return statevector(
        prototype,
        bindings,
        max_qubits=options.max_qubits,
        max_memory_mb=options.max_memory_mb,
    )


def _observables(observables, circuits):
    observables = tuple(observables)
    if not observables:
        raise ValidationError("At least one observable is required.")
    for observable in observables:
        terms = observable.terms if isinstance(observable, Hamiltonian) else (observable,)
        if any(not isinstance(term, PauliString) for term in terms):
            raise ValidationError("Expected PauliString or Hamiltonian observables.")
        if any(len(term.paulis) != c.num_qubits for c in circuits for term in terms):
            raise ValidationError("Observable width differs from the circuit width.")
    return observables


def _vectorized_densities(circuits, options):
    from .density import density_matrix
    from .parameters import expression

    first = circuits[0]

    def signature(c):
        return (
            c.num_qubits,
            c.num_clbits,
            tuple(
                (
                    path,
                    op.name,
                    op.qubits,
                    op.clbits,
                    op.matrix,
                    op.kraus,
                    op.condition,
                    len(op.params),
                )
                for path, op in c.walk_operations()
            ),
        )

    if any(signature(c) != signature(first) for c in circuits[1:]):
        return [
            density_matrix(
                c,
                max_qubits=options.max_qubits or 20,
                max_memory_mb=options.max_memory_mb,
                max_branches=options.max_branches,
            )
            for c in circuits
        ]
    prototype, bindings = first, {}
    for index, (path, op) in enumerate(first.walk_operations()):
        params = []
        for j in range(len(op.params)):
            name = f"_density_{index}_{j}"
            params.append(expression(Parameter(name)))
            bindings[name] = np.array([c.operation_at(path).params[j].constant for c in circuits])
        if params:
            prototype = prototype.replace_operation(path, replace(op, params=tuple(params)))
    return density_matrix(
        prototype,
        bindings,
        batch_shape=(len(circuits),),
        max_qubits=options.max_qubits or 20,
        max_memory_mb=options.max_memory_mb,
        max_branches=options.max_branches,
    )


def _dense_densities(circuits, device, options):
    from .density import _planning_limit, _resource_check, density_matrix
    from .planning import plan_density

    peak = 1
    for c in circuits:
        branches = (
            plan_density(
                c,
                max_branches=_planning_limit(
                    c,
                    options.max_qubits or 20,
                    options.max_memory_mb,
                    max_branches=options.max_branches,
                ),
            ).peak_branches
            if device is get_backend("numpy")
            else 1
        )
        _resource_check(c, options.max_qubits or 20, options.max_memory_mb, branches=branches)
        peak = max(peak, branches)
    width = max(c.num_qubits for c in circuits)
    chunk = max(1, min(128, int(options.max_memory_mb * 2**20 / (64 * 4**width * peak))))
    for start in range(0, len(circuits), chunk):
        group = circuits[start : start + chunk]
        if device is get_backend("numpy"):
            states = _vectorized_densities(group, options)
        else:
            states = [
                density_matrix(
                    c,
                    backend=device,
                    max_qubits=options.max_qubits or 20,
                    max_memory_mb=options.max_memory_mb,
                    max_branches=options.max_branches,
                )
                for c in group
            ]
        yield group, states


def _lower(circuits, device):
    from .compiler import decompose

    return [
        decompose(
            replace(c, operations=tuple(op for op in c.operations if op.name != "barrier")),
            basis=device.capabilities.gates,
        )
        for c in circuits
    ]


def _dense_states(circuits, device, options):
    from ._validation import invoke, validate_states
    from .simulator import _resource_check

    limit = options.max_qubits
    for circuit in circuits:
        _resource_check(circuit, limit, options.max_memory_mb)
    options = replace(options, max_qubits=limit)
    width = max(c.num_qubits for c in circuits)
    # Bound working buffers independently of the number of retained output states.
    chunk = max(1, min(128, int(options.max_memory_mb * 2**20 / (64 * 2**width))))
    for start in range(0, len(circuits), chunk):
        group = circuits[start : start + chunk]
        if device is get_backend("numpy"):
            states = _vectorized_states(group, options)
        elif callable(getattr(device, "run_batch", None)):
            states = invoke(device, "run_batch", group, options=options)
        else:
            states = [invoke(device, "run", c, options=options) for c in group]
        yield group, validate_states(states, group)


def expectations_batch(
    circuits,
    observables,
    bindings=None,
    *,
    backend="numpy",
    max_qubits=None,
    max_memory_mb=256,
    seed=None,
    max_branches=256,
):
    from ._validation import invoke, validate_expectations
    from .simulator import _expectations_from_state, _TensorOps

    options = ExecutionOptions(max_qubits, max_memory_mb, seed, max_branches)
    values = _bound(circuits, bindings)
    observables = _observables(observables, values)
    tensor = isinstance(backend, str) and backend in {"torch", "jax"}
    device = None if tensor else get_backend(backend)
    if not values:
        return np.empty((0, len(observables)))
    if tensor:
        from .simulator import expectations

        output = np.asarray(
            [
                np.asarray(
                    expectations(
                        c,
                        observables,
                        backend=backend,
                        max_qubits=max_qubits,
                        max_memory_mb=max_memory_mb,
                        max_branches=max_branches,
                    )
                )
                for c in values
            ]
        )
    else:
        _limits(values, device, max_qubits)
        from .backends import check_effects

        for c in values:
            check_effects(c, device, expectation=True)
        values = _lower(values, device)
        if device.capabilities.native_expectations:
            output = invoke(device, "expectations_batch", values, observables, options=options)
        elif any(c.has_effects for c in values) or (
            device.capabilities.density_matrix and not callable(getattr(device, "run", None))
        ):
            from .density import _expectations_from_density

            rows, ops = [], _TensorOps()
            for group, states in _dense_densities(values, device, options):
                stacked = np.stack(states)
                rows.extend(
                    _expectations_from_density(stacked, observables, group[0].num_qubits, ops)
                )
            output = np.asarray(rows)
        else:
            ops, rows = _TensorOps(), []
            for group, states in _dense_states(values, device, options):
                if len({c.num_qubits for c in group}) == 1:
                    stacked = np.stack(states)
                    rows.extend(
                        _expectations_from_state(stacked, observables, group[0].num_qubits, ops)
                    )
                else:
                    rows.extend(
                        [
                            _expectations_from_state(state, observables, c.num_qubits, ops)
                            for c, state in zip(group, states, strict=True)
                        ]
                    )
            output = np.asarray(rows)
    return validate_expectations(output, len(values), observables)


def execute_batch(
    circuits,
    bindings=None,
    *,
    backend="numpy",
    shots=None,
    seed=None,
    max_qubits=None,
    max_memory_mb=256,
    max_branches=256,
    classical_bits=None,
):
    from ._validation import invoke, project_counts, validate_classical_bits, validate_shots
    from .simulator import ExecutionResult, _validate_counts, execute

    options = ExecutionOptions(max_qubits, max_memory_mb, seed, max_branches)
    if shots is not None:
        shots = validate_shots(shots)
    classical_bits = validate_classical_bits(None, classical_bits)
    if classical_bits is not None and shots is None:
        raise ValidationError("Set shots when selecting classical count bits.")
    values = _bound(circuits, bindings)
    device = get_backend(backend)
    if not values:
        return []
    for c in values:
        validate_classical_bits(c, classical_bits)
    _limits(values, device, max_qubits)
    from .backends import check_effects

    for c in values:
        check_effects(c, device)
    if shots is not None and device.capabilities.native_sampling:
        counts = invoke(device, "sample_batch", values, shots, options=options)
        if not isinstance(counts, (list, tuple)) or len(counts) != len(values):
            raise BackendError("Invalid native sampler batch length.")
        for row, circuit in zip(counts, values, strict=True):
            _validate_counts(row, circuit, shots)
        return [
            ExecutionResult(
                device.name,
                None,
                None,
                project_counts(row, classical_bits),
                classical_bits=classical_bits,
            )
            for row in counts
        ]
    if (
        shots is not None
        or any(c.has_effects for c in values)
        or (
            device.capabilities.density_matrix
            and (
                not callable(getattr(device, "run", None))
                or getattr(device, "method", None) == "density_matrix"
            )
        )
    ):
        return [
            execute(
                c,
                backend=device,
                shots=shots,
                seed=None if seed is None else seed + i,
                max_qubits=max_qubits,
                max_memory_mb=max_memory_mb,
                max_branches=max_branches,
                classical_bits=classical_bits,
            )
            for i, c in enumerate(values)
        ]
    values = _lower(values, device)
    results = []
    for _, states in _dense_states(values, device, options):
        results.extend(ExecutionResult(device.name, np.abs(state) ** 2, state) for state in states)
    return results
