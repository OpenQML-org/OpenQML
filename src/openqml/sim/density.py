"""Differentiable density evolution with exact classical branch bookkeeping."""

import math
from dataclasses import dataclass

import numpy as np

from ..compute.kernel import _broadcast_shape, compute_operation, get_compute_kernel, memory_budget
from ..errors import ResourceError, ValidationError, _require_mapping
from ..ir.core import CHANNEL_GATES, UNITARY_GATES, _require_circuit, integer
from .planning import _record_plan


def _resource_check(circuit, max_qubits, max_memory_mb, batch_size=1, branches=1):
    _require_circuit(circuit)
    if max_qubits is not None:
        integer(max_qubits, "max_qubits", 1)
    budget = get_compute_kernel()._prepare_budget(max_memory_mb)
    capacity = budget.max_bytes >> (2 * circuit.num_qubits + 6)
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
    numerator, denominator = memory_budget(max_memory_mb).as_integer_ratio()
    capacity = ((numerator << 20) // (denominator * 64 * max(1, batch_size))) >> (
        2 * circuit.num_qubits
    )
    return max(1, min(max_branches, capacity))


def _probability(value, ops):
    p = ops.array(value, real_value=True)
    valid = ops.xp.isfinite(p) & (p >= 0) & (p <= 1)
    return ops.check(p, valid, "Channel probability must lie in [0, 1].")


def channel_kraus(op, bindings=None):
    """Bound, q0-MSB Kraus operators for SDK export; named probabilities are canonical."""
    if op.name == "kraus":
        return tuple(np.asarray(matrix, dtype=complex) for matrix in op.kraus)
    if op.name == "reset":
        return (np.array([[1, 0], [0, 0]], complex), np.array([[0, 1], [0, 0]], complex))
    from .dense import _X, _Y, _Z, _TensorOps

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
    from .dense import _apply_matrix

    left = _apply_matrix(rho, matrix, qubits, 2 * n, ops)
    return _apply_matrix(left, ops.conj(matrix), tuple(q + n for q in qubits), 2 * n, ops)


def _apply_channel(rho, op, bindings, n, ops):
    from .dense import _apply_matrix, _gate_matrix

    if op.name == "kraus":
        result = rho * 0
        for matrix in op.kraus:
            result = result + _conjugate(rho, ops.array(matrix), op.qubits, n, ops)
        return result
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
    paulis = {"bit_flip": ("x",), "phase_flip": ("z",), "depolarizing": ("x", "y", "z")}[op.name]
    transformed = rho * 0
    for name in paulis:
        transformed = transformed + _conjugate(rho, _gate_matrix(name, (), ops), op.qubits, n, ops)
    transformed = transformed / len(paulis)
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
    from .dense import _apply_matrix, _gate_matrix

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


@dataclass(frozen=True)
class _DensityInstruction:
    kind: str
    operation: object
    sources: tuple[int, ...]
    targets: tuple[int, ...]
    drops: tuple[int, ...] = ()


@dataclass(frozen=True)
class _DensityProgram:
    instructions: tuple[_DensityInstruction, ...]
    outputs: tuple[int, ...]
    slots: int
    peak_branches: int


def _density_program(circuit, records):
    """Resolve classical bookkeeping before native graph tracing."""
    from dataclasses import replace

    instructions, count = [], 1

    def emit(kind, operation, sources, width=1):
        nonlocal count
        targets = tuple(range(count, count + width))
        count += width
        instructions.append(_DensityInstruction(kind, operation, tuple(sources), targets))
        return targets

    def merge(pairs):
        groups = {}
        for key, slot in pairs:
            groups.setdefault(key, []).append(slot)
        return {
            key: slots[0] if len(slots) == 1 else emit("sum", None, slots)[0]
            for key, slots in groups.items()
        }

    def walk(body, states, plan):
        states = merge((key & plan.live_in, slot) for key, slot in states.items())
        for operation, step in zip(body.operations, plan.steps, strict=True):
            name = operation.name
            if name == "barrier":
                continue
            if name == "if_else":
                groups = [{}, {}]
                for key, slot in states.items():
                    groups[0 if operation.condition.matches(key) else 1][key] = slot
                outputs = []
                for index, group in enumerate(groups):
                    if group:
                        outputs.extend(
                            walk(operation.blocks[index], group, step.blocks[index]).items()
                        )
                states = merge(outputs)
            elif name == "measure":
                bit = operation.clbits[0]
                states = merge(
                    (key & (step.live_after - {bit}), slot) for key, slot in states.items()
                )
                if bit not in step.live_after:
                    targets = emit("dephase", operation, states.values(), len(states))
                    states = dict(zip(states, targets, strict=True))
                else:
                    outputs = []
                    for value in (0, 1):
                        targets = emit(str(value), operation, states.values(), len(states))
                        outputs.extend(
                            (key | {bit} if value else key, slot)
                            for key, slot in zip(states, targets, strict=True)
                        )
                    states = dict(outputs)
            else:
                targets = emit("gate", operation, states.values(), len(states))
                states = dict(zip(states, targets, strict=True))
        return states

    outputs = tuple(walk(circuit, {frozenset(): 0}, records).values())
    last_use = {}
    for index, instruction in enumerate(instructions):
        for source in instruction.sources:
            last_use[source] = index
    retained = set(outputs)
    drops = {}
    for source, index in last_use.items():
        if source not in retained:
            drops.setdefault(index, []).append(source)
    return _DensityProgram(
        tuple(replace(step, drops=tuple(drops.get(i, ()))) for i, step in enumerate(instructions)),
        outputs,
        count,
        records.peak_branches,
    )


def _compiled_density(circuit, program, bindings, ops, max_qubits, max_memory_mb):
    """Trace a static register program; no classical sets or generators in the graph."""
    from .dense import _apply_matrix, _gate_matrix, _initial_state

    n = circuit.num_qubits
    shape = _broadcast_shape(tuple(ops.parameter_shape(value) for value in bindings.values()))
    _resource_check(circuit, max_qubits, max_memory_mb, math.prod(shape), program.peak_branches)
    slots = [None] * program.slots
    slots[0] = _initial_state(2 * n, shape, ops)
    for step in program.instructions:
        if step.kind == "sum":
            rho = slots[step.sources[0]]
            for source in step.sources[1:]:
                rho = rho + slots[source]
            slots[step.targets[0]] = rho
        else:
            values = tuple(slots[source] for source in step.sources)
            rho = (
                values[0]
                if len(values) == 1
                else ops.xp.stack(values, dim=0)
                if ops.backend == "torch"
                else ops.xp.stack(values, axis=0)
            )
            op = step.operation
            if step.kind == "dephase":
                projection = ops.array(((1, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 1)))
                rho = _apply_matrix(rho, projection, (op.qubits[0], n + op.qubits[0]), 2 * n, ops)
            elif step.kind == "0" or step.kind == "1":
                value = int(step.kind)
                rho = _conjugate(rho, ops.array(((1 - value, 0), (0, value))), op.qubits, n, ops)
            elif op.name in CHANNEL_GATES or op.name == "reset":
                rho = _apply_channel(rho, op, bindings, n, ops)
            else:
                matrix = _gate_matrix(
                    op.name, [p.evaluate(bindings) for p in op.params], ops, op.matrix
                )
                rho = _conjugate(rho, matrix, op.qubits, n, ops)
            for index, target in enumerate(step.targets):
                slots[target] = rho if len(values) == 1 else rho[index]
        for source in step.drops:
            slots[source] = None
    return _finish_density(
        {index: slots[index] for index in program.outputs}, ops, shape, bindings, n
    )


def _simulate(
    circuit,
    bindings=None,
    *,
    backend="numpy",
    like=None,
    max_qubits=None,
    max_memory_mb=256,
    batch_shape=None,
    max_branches=256,
    retain_classical=False,
    classical_bits=None,
):
    from .dense import _TensorOps

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
        inferred = _broadcast_shape(tuple(ops.parameter_shape(v) for v in bindings.values()))
        shape = (
            inferred if batch_shape is None else _broadcast_shape((inferred, tuple(batch_shape)))
        )
    except ValueError as exc:
        raise ValidationError("Parameter batch shapes cannot be broadcast together.") from exc
    branch_limit = _planning_limit(
        circuit, max_qubits, max_memory_mb, math.prod(shape), max_branches
    )
    try:
        records = _record_plan(circuit, retain_classical, classical_bits, branch_limit)
    except ResourceError as exc:
        if branch_limit < max_branches:
            raise ResourceError("Density live branches exceed the working-memory budget.") from exc
        raise
    peak = records.peak_branches
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


@compute_operation
def density_matrix(
    circuit,
    bindings=None,
    *,
    backend="auto",
    like=None,
    max_qubits=None,
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
    _require_circuit(circuit)
    if bindings is not None:
        _require_mapping(bindings)
    integer(max_branches, "max_branches", 1)
    if not isinstance(backend, str):
        from ..runtime.backends import get_backend

        if backend is get_backend("numpy"):
            backend = "numpy"
    if not (isinstance(backend, str) and backend in {"numpy", "torch", "jax"}):
        from ..runtime._validation import invoke, validate_density
        from ..runtime.backends import ExecutionOptions, get_backend
        from ..runtime.scheduler import CostAwareBackend, PriorityBackend, _ObservedBackend

        device = get_backend(backend)
        if isinstance(device, PriorityBackend):
            options = ExecutionOptions(max_qubits, max_memory_mb, max_branches=max_branches)
            if isinstance(device, CostAwareBackend):
                route, rows = device.native_plan(
                    circuit, bindings, batch_shape, task="density", options=options
                )
            else:
                route = device.plan([circuit], task="density", options=options)
                rows = 1
            if route.device in {"CPU", "GPU"}:
                context, target = None, "numpy"
                if route.device == "GPU":
                    import torch

                    target, context = "torch", torch.empty((), dtype=torch.float64, device="cuda")

                def native_execute():
                    return density_matrix(
                        circuit,
                        bindings,
                        backend=target,
                        like=context,
                        max_qubits=max_qubits,
                        max_memory_mb=max_memory_mb,
                        max_branches=max_branches,
                        batch_shape=batch_shape,
                    )

                result = (
                    route._backend.run_native(native_execute, circuit, rows, options=options)
                    if isinstance(route._backend, _ObservedBackend)
                    else native_execute()
                )
                return result if target == "numpy" else result.detach().cpu().numpy()
        if like is not None or batch_shape is not None:
            raise ValidationError(
                "External density backends require scalar bindings without tensor context."
            )
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
        from ..runtime.scheduler import resolve_backend

        device = resolve_backend(device, "density", [bound], options=options)
        from ..runtime.backends import check_effects

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

            from ..ir.core import Operation

            projectors = ([[1, 0], [0, 0]], [[0, 0], [0, 1]])
            bound = replace(
                bound,
                operations=tuple(
                    Operation("kraus", op.qubits, kraus=projectors) if op.name == "measure" else op
                    for op in bound.operations
                ),
                num_clbits=0,
            )
        from ..ir.compiler import decompose

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
    return _finish_density(states, ops, shape, bindings, circuit.num_qubits)


def _finish_density(states, ops, shape, bindings, num_qubits):
    rho = sum(states.values())
    # Retain zero derivatives for global phases and inactive branch parameters.
    zero = 0
    for value in (bindings or {}).values():
        zero = zero + ops.array(value, real_value=True).sum() * 0
    dimension = 2**num_qubits
    return (rho + zero).reshape(shape + (dimension, dimension))


def _density_pauli_expectation(rho, paulis, num_qubits, ops):
    from .dense import _native_pauli_plan, _pauli_plan, _torch_pauli_plan

    if ops.backend == "torch":
        from ..compute.kernel import _torch_compiling

        source, phase = (
            _native_pauli_plan(paulis, ops)
            if _torch_compiling()
            else _torch_pauli_plan(paulis, ops.device, ops.dtype)
        )
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


@compute_operation
def execute_density(
    circuit,
    bindings=None,
    *,
    shots=None,
    seed=None,
    max_qubits=None,
    max_memory_mb=256,
    max_branches=256,
    classical_bits=None,
):
    from ..runtime._validation import validate_classical_bits
    from .dense import ExecutionResult

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
            if width > int(memory_budget(max_memory_mb) * 2**20):
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

        from ..interop.adapters.qiskit import to_qiskit

        rho = np.asarray(DensityMatrix.from_instruction(to_qiskit(circuit)).data)
        n = circuit.num_qubits
        axes = tuple(reversed(range(n))) + tuple(reversed(range(n, 2 * n)))
        return rho.reshape((2,) * (2 * n)).transpose(axes).reshape(rho.shape)
    if backend == "cirq":
        import cirq

        from ..interop.adapters.cirq import to_cirq

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

        from ..interop.adapters.pennylane import to_pennylane

        tape = to_pennylane(circuit)
        dev = qml.device("default.mixed", wires=circuit.num_qubits)

        @qml.qnode(dev)
        def node():
            for op in tape.operations:
                qml.apply(op)
            return qml.state()

        return np.asarray(node())
    raise ValidationError(f"No density executor for {backend!r}.")
