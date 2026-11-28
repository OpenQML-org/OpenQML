"""Absorbing quantum instruments on a finite classical control graph."""

import math
import secrets
from collections import defaultdict

import numpy as np

from ..compute.kernel import _torch_compiling
from ..errors import ResourceError, ValidationError
from ..ir._control import INDEX_RANGE_ERROR, RuntimeIndexError, control_program
from ..ir.core import CHANNEL_GATES, integer

_NONTERMINATING = frozenset({-2})


def _graph(circuit, max_nodes, max_configurations, max_branches, retain, selected):
    entry, code = control_program(circuit, max_configurations)
    routes = {}

    def route(pc, key):
        path = set()
        while pc >= 0:
            state = pc, key
            if state in routes:
                target = routes[state]
                break
            if state in path:
                target = -3, _NONTERMINATING
                break
            op, following, other = code[pc]
            if op.name == "index_error":
                target = -2, None
                break
            if op.name not in {"if_else", "while_loop", "classical_set", "barrier"}:
                target = state
                break
            path.add(state)
            if len(routes) + len(path) > max_configurations:
                raise ResourceError("Runtime control exceeds the classical configuration budget.")
            pc = following
            if op.name in {"if_else", "while_loop"}:
                if op.condition is not None and not op.condition.matches(key):
                    pc = other
            elif op.name == "classical_set":
                bit = op.clbits[0]
                key = key | {bit} if op.params[0].constant else key - {bit}
        else:
            record = key & selected if selected is not None else key
            target = -1, record if retain else frozenset()
        routes.update(dict.fromkeys(path, target))
        return target

    start = route(entry, frozenset())
    if start[1] == _NONTERMINATING:
        raise ResourceError("A recurrent classical path cannot terminate.")
    positions = {start: 0}
    nodes = [start]
    edges, terminals = [], {}
    widths = defaultdict(int)
    widths[start[0]] = 1
    for source, (pc, key) in enumerate(nodes):
        if pc < 0:
            terminals[key] = source
            continue
        op, following, _ = code[pc]
        if op.name == "measure":
            bit = op.clbits[0]
            choices = ((0, key - {bit}), (1, key | {bit}))
        else:
            choices = ((None, key),)
        for outcome, storage in choices:
            target = route(following, storage)
            if target not in positions:
                widths[target[0]] += 1
                if len(nodes) >= max_nodes or widths[target[0]] > max_branches:
                    raise ResourceError(
                        "Runtime control graph exceeds the memory or branch budget."
                    )
                positions[target] = len(nodes)
                nodes.append(target)
            edges.append((positions[target], source, op, outcome))
    if not terminals.keys() - {_NONTERMINATING}:
        raise ResourceError("A recurrent runtime program has no terminating path.")
    return len(nodes), edges, terminals


def _is_traced(value, ops):
    if ops.backend == "torch":
        return _torch_compiling()
    if ops.backend == "jax":
        from jax.core import Tracer

        return isinstance(value, Tracer)
    return False


def absorbing_instrument(
    circuit,
    bindings,
    ops,
    shape,
    *,
    complete_basis=False,
    max_qubits=None,
    max_memory_mb=256,
    max_branches=256,
    max_loop_steps=100_000,
    retain_classical=False,
    classical_bits=None,
    charge=None,
):
    """Double the horizon of a finite Markov instrument and certify absorption."""
    from .dense import _gate_matrix
    from .density import _apply_channel, _conjugate

    n = circuit.num_qubits
    max_loop_steps = integer(max_loop_steps, "max_loop_steps", 1)
    if max_qubits is not None and n > integer(max_qubits, "max_qubits", 1):
        raise ResourceError("Runtime circuit exceeds max_qubits.")
    budget = ops.kernel._prepare_budget(max_memory_mb).max_bytes
    width = math.isqrt(budget // (128 * max(1, math.prod(shape))))
    max_nodes = width >> (2 * n)
    if max_nodes < 1:
        raise ResourceError("Runtime operator workspace exceeds max_memory_mb.")
    size = 1 << (2 * n)
    count, edges, terminals = _graph(
        circuit,
        max_nodes,
        min(max_loop_steps, budget // 256),
        max_branches,
        retain_classical and circuit.has_classical_outputs,
        None if classical_bits is None else frozenset(classical_bits),
    )
    dimension = count * size
    if charge is not None:
        charge((len(edges) + count) * size * size)
    identity = (
        ops.xp.eye(size, dtype=ops.dtype, device=ops.device)
        if ops.backend == "torch"
        else ops.array(ops.xp.eye(size, dtype=ops.dtype))
    )
    basis = ops.xp.broadcast_to(identity, shape + (size, size))
    zero = identity * 0
    blocks = [[zero] * count for _ in range(count)]
    for position in terminals.values():
        blocks[position][position] = identity
    params = {
        name: ops.array(value, real_value=True)[..., None] for name, value in bindings.items()
    }
    for target, source, op, outcome in edges:
        if outcome is not None:
            projector = ((1, 0), (0, 0)) if outcome == 0 else ((0, 0), (0, 1))
            image = _conjugate(basis, ops.array(projector), op.qubits, n, ops)
        elif op.name == "reset" or op.name in CHANNEL_GATES:
            image = _apply_channel(basis, op, params, n, ops)
        else:
            matrix = _gate_matrix(op.name, [p.evaluate(params) for p in op.params], ops, op.matrix)
            image = _conjugate(basis, matrix, op.qubits, n, ops)
        blocks[target][source] = blocks[target][source] + image.swapaxes(-1, -2)
    transition = ops.stack([ops.stack(row, axis=-2) for row in blocks], axis=-4)
    del blocks
    transition = ops.xp.broadcast_to(
        transition.reshape(transition.shape[:-4] + (dimension, dimension)),
        shape + (dimension, dimension),
    )
    initial_columns = size if complete_basis else 1
    diag = (1 << n) + 1
    expected = identity[:, ::diag].sum(axis=-1) if complete_basis else 1
    tolerance = max(1e-12, 32 * float(ops.xp.finfo(ops.real_dtype).eps))

    def balance(matrix):
        # Each transition is trace-preserving on the joint quantum/classical
        # space. Remove accumulated diagonal trace roundoff before doubling.
        view = matrix.reshape(shape + (count, size, count, size))
        traces = view[..., :, ::diag, :, :].sum(axis=-4).sum(axis=-3)
        roots = ops.xp.sqrt(traces[..., ::diag].real)
        scale = (roots[..., :, None] * roots[..., None, :]).reshape(shape + (dimension,))
        return matrix / scale[..., None, :]

    def read(matrix):
        rows = {
            key: matrix[..., i * size : (i + 1) * size, :initial_columns].swapaxes(-1, -2)
            for key, i in terminals.items()
            if key != _NONTERMINATING
        }
        trace = sum(value[..., ::diag].sum(axis=-1) for value in rows.values())
        valid = (ops.xp.abs(trace - expected) <= tolerance).all(axis=-1)
        return rows, valid

    # A terminal block is the identity, so squaring also accumulates exits.
    transition = balance(transition)
    rounds = 60 if max_loop_steps == 100_000 else (max_loop_steps.bit_length() - 1)
    rows, valid = read(transition)
    traced = _is_traced(valid, ops)
    if traced and ops.backend == "jax":
        from jax import lax

        transition = lax.fori_loop(
            0, rounds, lambda _, matrix: balance(matrix @ matrix), transition
        )
        rows, valid = read(transition)
    else:
        for _ in range(rounds):
            if not traced and bool(ops.xp.all(valid)):
                break
            if charge is not None:
                charge(dimension**3 + dimension**2)
            transition = balance(transition @ transition)
            rows, valid = read(transition)
    rows = {
        key: ops.kernel.check(
            value,
            valid[..., None, None],
            "A recurrent runtime path has no certified probability-one termination.",
            xp=ops.xp,
            backend=ops.backend,
            error=ResourceError,
        )
        for key, value in rows.items()
    }
    invalid = rows.pop(None, None)
    if complete_basis:
        if invalid is not None:
            rows[frozenset({-1})] = invalid
        return rows
    if invalid is not None:
        valid = (invalid == 0).all(axis=-1).all(axis=-1)
        if not rows:
            rows[frozenset()] = invalid * 0
        rows = {
            key: ops.kernel.check(
                value,
                valid[..., None, None],
                INDEX_RANGE_ERROR,
                xp=ops.xp,
                backend=ops.backend,
                error=RuntimeIndexError,
            )
            for key, value in rows.items()
        }
    return {key: value[..., 0, :] for key, value in rows.items()}


def execute_tensor_control(
    circuit, bindings, backend, options, shots, classical_bits, max_loop_steps, like, counts_only
):
    from .dense import ExecutionResult
    from .density import _simulate

    recorded = shots is not None and circuit.has_classical_outputs
    states, ops, shape = _simulate(
        circuit,
        bindings,
        backend=backend,
        like=like,
        max_qubits=options.max_qubits,
        max_memory_mb=options.max_memory_mb,
        max_branches=options.max_branches,
        max_loop_steps=max_loop_steps,
        retain_classical=recorded,
        classical_bits=classical_bits,
    )
    if shape:
        raise ValidationError("execute requires a single scalar binding for runtime control.")
    size = 1 << circuit.num_qubits
    rho = sum(states.values()).reshape((size, size))
    xp = ops.xp
    probabilities = xp.maximum(ops.array(0, real_value=True), rho.diagonal().real)
    counts = None
    if shots is not None:
        width = circuit.num_qubits
        if recorded:
            width = circuit.num_clbits if classical_bits is None else len(classical_bits)
        budget = ops.kernel._prepare_budget(options.max_memory_mb).max_bytes
        support = len(states) if recorded else size
        if (128 + width) * support > budget:
            raise ResourceError("Runtime count strings exceed max_memory_mb.")
        weights = (
            ops.stack([state[:: size + 1].real.sum() for state in states.values()]).real
            if recorded
            else probabilities
        )
        weights = xp.maximum(ops.array(0, real_value=True), weights)
        weights = weights / weights.sum()
        seed = options.seed if options.seed is not None else secrets.randbits(63)
        frequencies = _sample_native(weights, shots, seed, max(1, budget // 64), ops)
        if recorded:
            bits = range(width) if classical_bits is None else classical_bits
            counts = {
                "".join("1" if bit in key else "0" for bit in bits): frequency
                for key, frequency in zip(states, frequencies, strict=True)
                if frequency
            }
        else:
            counts = {
                format(key, f"0{width}b"): frequency
                for key, frequency in enumerate(frequencies)
                if frequency
            }
    if counts_only:
        return ExecutionResult(backend, None, None, counts, classical_bits=classical_bits)
    if backend == "torch":
        rho, probabilities = rho.detach().cpu(), probabilities.detach().cpu()
    return ExecutionResult(
        backend, np.asarray(probabilities), None, counts, np.asarray(rho), classical_bits
    )


def _sample_native(weights, shots, seed, chunk, ops):
    xp = ops.xp
    frequencies = [0] * len(weights)
    if ops.backend == "torch":
        weights = weights.detach()
        generator = xp.Generator(device=weights.device).manual_seed(int(seed) % (1 << 63))
    else:
        from jax import random

        key = random.key(int(seed) % (1 << 32))
        logits = xp.log(weights)
    for start in range(0, shots, min(chunk, 65_536)):
        count = min(shots - start, chunk, 65_536)
        if ops.backend == "torch":
            samples = xp.multinomial(weights, count, replacement=True, generator=generator)
            histogram = xp.bincount(samples, minlength=len(weights))
        else:
            key, draw = random.split(key)
            samples = random.categorical(draw, logits, shape=(count,))
            histogram = xp.bincount(samples, length=len(weights))
        for index, frequency in enumerate(histogram.tolist()):
            frequencies[index] += frequency
    return frequencies
