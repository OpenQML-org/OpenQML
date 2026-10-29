"""Differentiable density evolution with exact classical branch bookkeeping."""

import math

import numpy as np

from .errors import ResourceError, ValidationError
from .ir import CHANNEL_GATES, UNITARY_GATES, integer
from .parameters import real
from .planning import _record_plan


def _resource_check(circuit, max_qubits, max_memory_mb, batch_size=1, branches=1):
    if max_qubits is not None:
        integer(max_qubits, "max_qubits", 1)
    budget = real(max_memory_mb)
    if budget <= 0:
        raise ValidationError("max_memory_mb must be positive.")
    numerator, denominator = budget.as_integer_ratio()
    capacity = ((numerator << 20) // denominator) >> (2 * circuit.num_qubits + 6)
    if max(1, batch_size) * branches > capacity or (
        max_qubits is not None and circuit.num_qubits > max_qubits
    ):
        raise ResourceError(
            "Density simulation exceeds the qubit/working-memory budget (4**n per branch)."
        )


def _planning_limit(circuit, max_qubits, max_memory_mb, batch_size=1, max_branches=256):
    """Bound metadata enumeration by memory before allocating branch-key sets."""
    integer(max_branches, "max_branches", 1)
    _resource_check(circuit, max_qubits, max_memory_mb, batch_size)
    numerator, denominator = real(max_memory_mb).as_integer_ratio()
    capacity = ((numerator << 20) // (denominator * 64 * max(1, batch_size))) >> (
        2 * circuit.num_qubits
    )
    return max(1, min(max_branches, capacity))


def _probability(value, ops):
    p = ops.array(value, real_value=True)
    valid = ops.xp.isfinite(p) & (p >= 0) & (p <= 1)
    if ops.backend == "numpy":
        if not np.all(valid):
            raise ValidationError("Channel probability must lie in [0, 1].")
    elif ops.backend == "torch":
        if not getattr(ops.xp.compiler, "is_compiling", lambda: False)() and not bool(
            ops.xp.all(valid)
        ):
            raise ValidationError("Channel probability must lie in [0, 1].")
    else:
        import jax

        if not isinstance(p, jax.core.Tracer) and not np.all(np.asarray(valid)):
            raise ValidationError("Channel probability must lie in [0, 1].")
    # Traced invalid values propagate NaN instead of a plausible nonphysical map.
    return ops.xp.where(valid, p, float("nan"))


def channel_kraus(op, bindings=None):
    """Bound, q0-MSB Kraus operators for SDK export; named probabilities are canonical."""
    if op.name == "kraus":
        return tuple(np.asarray(matrix, dtype=complex) for matrix in op.kraus)
    if op.name == "reset":
        return (np.array([[1, 0], [0, 0]], complex), np.array([[0, 1], [0, 0]], complex))
    from .simulator import _X, _Y, _Z, _TensorOps

    p = float(_probability(op.params[0].evaluate(bindings or {}), _TensorOps()))
    if op.name == "amplitude_damping":
        return (
            np.diag([1, np.sqrt(1 - p)]).astype(complex),
            np.array([[0, np.sqrt(p)], [0, 0]], complex),
        )
    if op.name == "phase_damping":
        return (
            np.diag([1, np.sqrt(1 - p)]).astype(complex),
            np.diag([0, np.sqrt(p)]).astype(complex),
        )
    paulis = {"bit_flip": (_X,), "phase_flip": (_Z,), "depolarizing": (_X, _Y, _Z)}[op.name]
    return (np.sqrt(1 - p) * np.eye(2, dtype=complex),) + tuple(
        np.sqrt(p / len(paulis)) * a for a in paulis
    )


def _conjugate(rho, matrix, qubits, n, ops):
    from .simulator import _apply_matrix

    left = _apply_matrix(rho, matrix, qubits, 2 * n, ops)
    return _apply_matrix(left, ops.conj(matrix), tuple(q + n for q in qubits), 2 * n, ops)


def _apply_channel(rho, op, bindings, n, ops):
    from .simulator import _X, _Y, _Z, _apply_matrix

    if op.name == "kraus":
        return sum((_conjugate(rho, ops.array(k), op.qubits, n, ops) for k in op.kraus), rho * 0)
    if op.name == "reset":
        matrix = ops.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        return _apply_matrix(rho, matrix, (op.qubits[0], n + op.qubits[0]), 2 * n, ops)
    p = _probability(op.params[0].evaluate(bindings), ops)
    if op.name in {"amplitude_damping", "phase_damping"}:
        s = ops.xp.sqrt(1 - p)
        if op.name == "amplitude_damping":
            matrix = ops.matrix([[1, 0, 0, p], [0, s, 0, 0], [0, 0, s, 0], [0, 0, 0, 1 - p]])
        else:
            matrix = ops.matrix([[1, 0, 0, 0], [0, s, 0, 0], [0, 0, s, 0], [0, 0, 0, 1]])
        return _apply_matrix(rho, matrix, (op.qubits[0], n + op.qubits[0]), 2 * n, ops)
    paulis = {"bit_flip": (_X,), "phase_flip": (_Z,), "depolarizing": (_X, _Y, _Z)}[op.name]
    transformed = sum(
        (_conjugate(rho, ops.array(a), op.qubits, n, ops) for a in paulis), rho * 0
    ) / len(paulis)
    return (1 - p)[..., None] * rho + p[..., None] * transformed


def _forget(states, live):
    output = {}
    for key, rho in states.items():
        target = key & live
        output[target] = output[target] + rho if target in output else rho
    return output


def _map_states(states, function, ops):
    if len(states) == 1:
        key, rho = next(iter(states.items()))
        return {key: function(rho)}
    keys = tuple(states)
    values = tuple(states.values())
    stacked = (
        ops.xp.stack(values, dim=0) if ops.backend == "torch" else ops.xp.stack(values, axis=0)
    )
    result = function(stacked)
    return {key: result[i] for i, key in enumerate(keys)}


def _evolve(circuit, states, bindings, ops, records):
    """Trampoline nested blocks so parent call stacks cannot retain density buffers."""
    stack = [_evolve_steps(circuit, states, bindings, ops, records)]
    del states
    value = None
    while stack:
        try:
            task = stack[-1].send(value)
        except StopIteration as completed:
            value = completed.value
            stack.pop()
        else:
            value = None
            body, branch, plan = task
            stack.append(_evolve_steps(body, branch, bindings, ops, plan))
            # CPython 3.10 retains evaluated call arguments until the callee
            # returns. Yielding tasks avoids one retained input per nesting level.
            del task, branch
    return value


def _evolve_steps(circuit, states, bindings, ops, records):
    from .simulator import _apply_matrix, _gate_matrix

    n = circuit.num_qubits
    states = _forget(states, records.live_in)
    for op, step in zip(circuit.operations, records.steps, strict=True):
        if op.name == "barrier":
            continue
        if op.name == "if_else":
            groups, output = [{}, {}], {}
            # Choose the body before forgetting condition bits. All maps inside
            # that selected body are linear in the unnormalized branch state.
            for key, rho in states.items():
                i = 0 if op.condition.matches(key) else 1
                target = key & step.blocks[i].live_in
                groups[i][target] = groups[i][target] + rho if target in groups[i] else rho
            # Release parent references before descending into the child.
            states = {}
            del key, rho
            for i in range(2):
                if groups[0]:
                    branch = yield op.blocks[i], groups.pop(0), step.blocks[i]
                    for key, rho in branch.items():
                        output[key] = output[key] + rho if key in output else rho
                    del branch, key, rho
                else:
                    groups.pop(0)
            states = output
            continue
        if op.name == "measure":
            bit = op.clbits[0]
            states = _forget(states, step.live_after - {bit})
            if bit not in step.live_after:
                projection = ops.array(np.diag([1, 0, 0, 1]))
                axes = (op.qubits[0], n + op.qubits[0])
                states = _map_states(
                    states,
                    lambda rho, projection=projection, axes=axes: _apply_matrix(
                        rho, projection, axes, 2 * n, ops
                    ),
                    ops,
                )
            else:
                output = {}
                for value in (0, 1):
                    projection = ops.array([[1 - value, 0], [0, value]])
                    collapsed = _map_states(
                        states,
                        lambda rho, projection=projection, qubits=op.qubits: _conjugate(
                            rho, projection, qubits, n, ops
                        ),
                        ops,
                    )
                    for key, rho in collapsed.items():
                        output[key | {bit} if value else key] = rho
                states = output
        elif op.name in CHANNEL_GATES | {"reset"}:
            states = _map_states(
                states, lambda rho, op=op: _apply_channel(rho, op, bindings, n, ops), ops
            )
        elif op.name in UNITARY_GATES:
            matrix = _gate_matrix(
                op.name, [p.evaluate(bindings) for p in op.params], ops, op.matrix
            )
            states = _map_states(
                states,
                lambda rho, matrix=matrix, qubits=op.qubits: _conjugate(
                    rho, matrix, qubits, n, ops
                ),
                ops,
            )
        else:
            raise ValidationError(f"No density execution rule for {op.name!r}.")
    return states


def _simulate(
    circuit,
    bindings=None,
    *,
    backend="numpy",
    like=None,
    max_qubits=20,
    max_memory_mb=256,
    batch_shape=None,
    max_branches=256,
    retain_classical=False,
    classical_bits=None,
):
    from .simulator import _TensorOps

    integer(max_branches, "max_branches", 1)
    bindings = {} if bindings is None else bindings
    if set(bindings) != set(circuit.parameters):
        raise ValidationError(f"Expected exactly these parameter bindings: {circuit.parameters}.")
    if backend == "numpy":
        clean = {}
        for name, value in bindings.items():
            value = np.asarray(value)
            if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
                raise ValidationError("Bindings must be finite real scalars or arrays.")
            clean[name] = value.astype(float, copy=False)
        bindings = clean
    if like is None and backend != "numpy":
        like = next((v for v in bindings.values() if hasattr(v, "dtype")), None)
    ops = _TensorOps(backend, like)
    try:
        inferred = np.broadcast_shapes(*(tuple(getattr(v, "shape", ())) for v in bindings.values()))
        shape = (
            inferred if batch_shape is None else np.broadcast_shapes(inferred, tuple(batch_shape))
        )
    except ValueError as exc:
        raise ValidationError("Parameter batch shapes cannot be broadcast together.") from exc
    branch_limit = _planning_limit(
        circuit, max_qubits, max_memory_mb, math.prod(shape), max_branches
    )
    try:
        records = _record_plan(circuit, retain_classical, classical_bits, branch_limit)
        peak = records.peak_branches
    except ResourceError as exc:
        if branch_limit < max_branches:
            raise ResourceError("Density live branches exceed the working-memory budget.") from exc
        raise
    _resource_check(circuit, max_qubits, max_memory_mb, math.prod(shape), peak)
    size = 4**circuit.num_qubits
    if backend == "torch":
        rho = ops.xp.zeros(size, dtype=ops.dtype, device=ops.device)
        rho[0] = 1
    elif backend == "jax":
        rho = ops.xp.zeros(size, dtype=ops.dtype).at[0].set(1)
    else:
        rho = np.zeros(size, dtype=complex)
        rho[0] = 1
    rho = ops.xp.broadcast_to(rho, shape + (size,))
    states = _evolve(circuit, {frozenset(): rho}, bindings, ops, records)
    return states, ops, shape


def density_matrix(
    circuit,
    bindings=None,
    *,
    backend="numpy",
    like=None,
    max_qubits=20,
    max_memory_mb=256,
    batch_shape=None,
    max_branches=256,
):
    """Final unconditional density, including measurements/reset/atomic classical branches.

    Classical decisions are evaluated once on entering each block. Unnormalized
    outcome branches preserve quantum/classical correlations and merge on bit writes.
    Records are marginalized after their last read; no probability-based pruning
    changes the graph at zero-weight branches.
    """
    integer(max_branches, "max_branches", 1)
    if not isinstance(backend, str):
        from .backends import get_backend

        if backend is get_backend("numpy"):
            backend = "numpy"
    if not (isinstance(backend, str) and backend in {"numpy", "torch", "jax"}):
        from ._validation import invoke, validate_density
        from .backends import ExecutionOptions, get_backend

        if like is not None or batch_shape is not None:
            raise ValidationError(
                "External density backends require scalar bindings without tensor context."
            )
        device = get_backend(backend)
        if not device.capabilities.density_matrix:
            raise ValidationError(f"Backend {device.name!r} has no density-matrix contract.")
        options = ExecutionOptions(max_qubits, max_memory_mb, max_branches=max_branches)
        _resource_check(circuit, max_qubits, max_memory_mb)
        if (
            device.capabilities.max_qubits is not None
            and circuit.num_qubits > device.capabilities.max_qubits
        ):
            raise ResourceError("Circuit exceeds backend qubit limit.")
        bound = circuit.bind({} if bindings is None else bindings)
        from .backends import check_effects

        check_effects(bound, device, expectation=True)
        if bound.has_measurements and not device.capabilities.measurement_expectations:
            if (
                getattr(device, "noise", None) is not None
                or getattr(device, "noise_model", None) is not None
            ):
                raise ValidationError(
                    "Measurement-channel lowering cannot preserve native noise placement. "
                    "Use explicit IR channels with numpy or a native ShotBackend."
                )
            # Discarded records require the nonselective projective channel.
            from dataclasses import replace

            from .ir import Operation

            projectors = ([[1, 0], [0, 0]], [[0, 0], [0, 1]])
            bound = replace(
                bound,
                operations=tuple(
                    Operation("kraus", op.qubits, kraus=projectors) if op.name == "measure" else op
                    for op in bound.operations
                ),
                num_clbits=0,
            )
        from .compiler import decompose

        bound = decompose(bound, basis=device.capabilities.gates)
        return validate_density(invoke(device, "run_density", bound, options=options), circuit)
    states, ops, shape = _simulate(
        circuit,
        bindings,
        backend=backend,
        like=like,
        max_qubits=max_qubits,
        max_memory_mb=max_memory_mb,
        batch_shape=batch_shape,
        max_branches=max_branches,
    )
    rho = sum(states.values())
    # Retain zero derivatives for global phases and inactive branch parameters.
    zero = sum(
        (ops.array(value, real_value=True).sum() * 0 for value in (bindings or {}).values()), 0
    )
    dimension = 2**circuit.num_qubits
    return (rho + zero).reshape(shape + (dimension, dimension))


def _density_pauli_expectation(rho, paulis, num_qubits, ops):
    from .simulator import _pauli_plan, _torch_pauli_plan

    if ops.backend == "torch":
        source, phase = _torch_pauli_plan(paulis, ops.device, ops.dtype)
        indices = ops.xp.arange(2**num_qubits, device=ops.device)
    else:
        source, phase = _pauli_plan(paulis)
        source, phase = ops.xp.asarray(source), ops.array(phase)
        indices = ops.xp.arange(2**num_qubits)
    return ops.real((rho[..., source, indices] * phase).sum(axis=-1))


def _expectations_from_density(rho, observables, num_qubits, ops):
    from ._readouts import _evaluate_readouts

    def diagonal():
        value = (
            rho.diagonal(dim1=-2, dim2=-1)
            if ops.backend == "torch"
            else rho.diagonal(axis1=-2, axis2=-1)
        )
        return ops.real(value)

    return _evaluate_readouts(
        observables,
        num_qubits,
        ops,
        rho[..., 0, 0],
        diagonal,
        lambda word: _density_pauli_expectation(rho, word, num_qubits, ops),
    )


def expectation_from_density(rho, observable, num_qubits, ops):
    return _expectations_from_density(rho, (observable,), num_qubits, ops)[..., 0]


def execute_density(
    circuit,
    bindings=None,
    *,
    shots=None,
    seed=None,
    max_qubits=20,
    max_memory_mb=256,
    max_branches=256,
    classical_bits=None,
):
    from ._validation import validate_classical_bits
    from .simulator import ExecutionResult

    classical_bits = validate_classical_bits(circuit, classical_bits)

    states, _, _ = _simulate(
        circuit,
        bindings,
        max_qubits=max_qubits,
        max_memory_mb=max_memory_mb,
        max_branches=max_branches,
        retain_classical=shots is not None and circuit.has_measurements,
        classical_bits=classical_bits,
    )
    dimension = 2**circuit.num_qubits
    rho = sum(states.values()).reshape(dimension, dimension)
    probabilities = np.maximum(0, np.real(np.diag(rho)))
    counts = None
    if shots is not None:
        rng = np.random.default_rng(seed)
        if circuit.has_measurements:
            width = circuit.num_clbits if classical_bits is None else len(classical_bits)
            if width > int(real(max_memory_mb) * 2**20):
                raise ResourceError("Classical count strings exceed the working-memory budget.")
            keys = list(states)
            weights = np.maximum(
                0,
                np.asarray([np.trace(states[k].reshape(dimension, dimension)).real for k in keys]),
            )
            histogram = rng.multinomial(shots, weights / weights.sum())
            counts = {
                "".join(
                    "1" if bit in key else "0"
                    for bit in (range(width) if classical_bits is None else classical_bits)
                ): int(count)
                for key, count in zip(keys, histogram, strict=True)
                if count
            }
        else:
            histogram = rng.multinomial(shots, probabilities / probabilities.sum())
            counts = {
                format(int(i), f"0{circuit.num_qubits}b"): int(histogram[i])
                for i in np.flatnonzero(histogram)
            }
    return ExecutionResult("numpy", probabilities, None, counts, rho, classical_bits)


def external_density(circuit, backend):
    if backend == "qiskit":
        from qiskit.quantum_info import DensityMatrix

        from .adapters.qiskit import to_qiskit

        rho = np.asarray(DensityMatrix.from_instruction(to_qiskit(circuit)).data)
        n = circuit.num_qubits
        axes = tuple(reversed(range(n))) + tuple(reversed(range(n, 2 * n)))
        return rho.reshape((2,) * (2 * n)).transpose(axes).reshape(rho.shape)
    if backend == "cirq":
        import cirq

        from .adapters.cirq import to_cirq

        return (
            cirq.DensityMatrixSimulator(dtype=np.complex128)
            .simulate(
                to_cirq(circuit.with_global_phase(0)),
                qubit_order=cirq.LineQubit.range(circuit.num_qubits),
            )
            .final_density_matrix
        )
    if backend == "pennylane":
        import pennylane as qml

        from .adapters.pennylane import to_pennylane

        tape = to_pennylane(circuit)
        dev = qml.device("default.mixed", wires=circuit.num_qubits)

        @qml.qnode(dev)
        def node():
            for op in tape.operations:
                qml.apply(op)
            return qml.state()

        return np.asarray(node())
    raise ValidationError(f"No density executor for {backend!r}.")
