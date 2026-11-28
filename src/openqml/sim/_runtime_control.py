"""Exact finite control paths and grouped measurement trajectories on NumPy."""

import hashlib
import math

import numpy as np

from ..compute.kernel import get_compute_kernel
from ..errors import ResourceError, ValidationError
from ..ir._control import control_program
from ..ir.core import CHANNEL_GATES, integer


def _operator_digest(state):
    components = np.ascontiguousarray(state).view(state.real.dtype)
    magnitude = np.max(np.abs(components), initial=0)
    if not magnitude:
        return None
    # The remaining program acts linearly, so a common scale cannot affect
    # which measurement branches of this operator can reach the loop again.
    normalized = components / magnitude
    normalized[normalized == 0] = 0
    return hashlib.blake2b(normalized.view(np.uint8), digest_size=16).digest()


def _scale_operator(state, exponent):
    result = np.empty_like(state)
    np.ldexp(state.real, exponent, out=result.real)
    np.ldexp(state.imag, exponent, out=result.imag)
    return result


def run_control(
    circuit,
    bindings,
    ops,
    shape,
    *,
    shots=None,
    seed=None,
    max_qubits=None,
    max_memory_mb=256,
    max_branches=256,
    max_loop_steps=100_000,
    retain_classical=False,
    classical_bits=None,
    input_operator=None,
    charge=None,
):
    from .dense import _apply_matrix, _gate_matrix
    from .density import _apply_channel, _conjugate

    limit = integer(max_loop_steps, "max_loop_steps", 1)
    max_branches = integer(max_branches, "max_branches", 1)
    if ops.backend != "numpy":
        raise ValidationError("Runtime control uses NumPy or a native control-flow sampler.")
    n = circuit.num_qubits
    if max_qubits is not None and n > integer(max_qubits, "max_qubits", 1):
        raise ResourceError("Runtime circuit exceeds max_qubits.")
    budget = get_compute_kernel()._prepare_budget(max_memory_mb).max_bytes
    sampled = shots is not None
    pure = sampled and not any(op.name in CHANNEL_GATES for _, op in circuit.walk_operations())
    power = n if pure else 2 * n
    if max(1, math.prod(shape)) > budget >> (power + 6):
        raise ResourceError("Runtime quantum state exceeds max_memory_mb.")
    entry, code = control_program(circuit, min(100_000, budget // 160))
    record = retain_classical and circuit.has_classical_outputs
    selected = None if classical_bits is None else frozenset(classical_bits)
    width = (circuit.num_clbits if selected is None else len(selected)) if record else n
    if sampled and 128 + width > budget:
        raise ResourceError("Runtime count strings exceed max_memory_mb.")
    size = 1 << n
    if input_operator is None:
        state = np.zeros(shape + (1 << power,), dtype=complex)
        state[..., 0] = 1
    else:
        state = input_operator
        if sampled or state.shape != shape + (size * size,):
            raise ValidationError("An exact runtime input must contain full operator basis rows.")
    state_bytes = state.nbytes
    available = budget - 160 * len(code) - 3 * state_bytes
    rng = np.random.default_rng(seed) if sampled else None
    projectors = (np.diag([1.0, 0.0]), np.diag([0.0, 1.0]))
    pending = [(entry, frozenset(), state, shots, 0, set(), 0)]
    outputs = {}
    steps = pending_bytes = output_bytes = 0

    def check(live, extra=0):
        if live > max_branches:
            raise ResourceError("Runtime execution exceeds max_branches.")
        if live * state_bytes + extra + pending_bytes + output_bytes > available:
            raise ResourceError("Runtime execution exceeds max_memory_mb.")

    def add_count(key, num):
        nonlocal output_bytes
        if key not in outputs:
            output_bytes += 128 + width
            check(len(pending) + 1)
        outputs[key] = outputs.get(key, 0) + num

    check(1)
    while pending:
        pc, key, state, count, exponent, visited, history_bytes = pending.pop()
        pending_bytes -= history_bytes
        while pc != -1:
            steps += 1
            if steps > limit:
                raise ResourceError("Runtime control exceeds max_loop_steps before termination.")
            op, following, other = code[pc]
            name = op.name
            if charge is not None:
                unit = (
                    1
                    if name in {"while_loop", "if_else", "classical_set", "barrier"}
                    else 4 * (len(op.kraus) if name == "kraus" else 4) * (1 << len(op.qubits))
                )
                charge(unit * state.size)
            if name in {"while_loop", "if_else"}:
                take = op.condition is None or op.condition.matches(key)
                if name == "while_loop" and take and not sampled:
                    digest = _operator_digest(state)
                    if digest is None:
                        state = None
                        break
                    signature = pc, key, digest
                    if signature in visited:
                        raise ResourceError(
                            "A recurrent loop path has no finite termination proof. "
                            "Use sample() for trajectories."
                        )
                    visited.add(signature)
                    history_bytes += 512 + 32 * len(key)
                    check(len(pending) + len(outputs) + 1, history_bytes)
                pc = following if take else other
                continue
            pc = following
            if name == "barrier":
                continue
            if name == "classical_set":
                bit = op.clbits[0]
                key = key | {bit} if op.params[0].constant else key - {bit}
            elif name == "measure" or (pure and name == "reset"):
                apply = _apply_matrix if pure else _conjugate
                children = [apply(state, p, op.qubits, n, ops) for p in projectors]
                if sampled:
                    weights = np.array(
                        [
                            np.vdot(v, v).real if pure else np.trace(v.reshape(size, size)).real
                            for v in children
                        ]
                    )
                    weights = np.maximum(weights, 0)
                    if not np.isfinite(weights).all() or weights.sum() <= 0:
                        raise ValidationError("Invalid quantum measurement probabilities.")
                    ones = int(rng.binomial(count, weights[1] / weights.sum()))
                    counts = count - ones, ones
                branches = []
                for value, child in enumerate(children):
                    num, child_exponent = None, exponent
                    if sampled:
                        num = counts[value]
                        if not num:
                            continue
                        child = child / (math.sqrt(weights[value]) if pure else weights[value])
                    else:
                        magnitude = float(np.max(np.abs(child), initial=0))
                        if not magnitude:
                            continue
                        # Track the scale separately so a recurrent branch
                        # cannot disappear through probability underflow.
                        _, shift = math.frexp(magnitude)
                        if shift:
                            child = _scale_operator(child, -shift)
                            child_exponent += shift
                    target = key
                    if name == "measure":
                        bit = op.clbits[0]
                        target = key | {bit} if value else key - {bit}
                    elif value:
                        child = _apply_matrix(child, _gate_matrix("x", (), ops), op.qubits, n, ops)
                    branches.append((pc, target, child, num, child_exponent))
                check(
                    len(pending) + len(branches) + (0 if sampled else len(outputs)),
                    history_bytes * len(branches),
                )
                if len(branches) == 2:
                    pending_bytes += history_bytes
                    pending.append((*branches.pop(), visited.copy(), history_bytes))
                if not branches:
                    state = None
                    break
                pc, key, state, count, exponent = branches[0]
                del children, child, branches
            elif name == "reset" or name in CHANNEL_GATES:
                state = _apply_channel(state, op, bindings, n, ops)
            else:
                matrix = _gate_matrix(
                    name, [p.evaluate(bindings) for p in op.params], ops, op.matrix
                )
                state = (_apply_matrix if pure else _conjugate)(state, matrix, op.qubits, n, ops)
        if state is None:
            continue
        if not sampled:
            target = (key if selected is None else key & selected) if record else frozenset()
            if target not in outputs:
                output_bytes += 256 + 32 * len(target)
            check(len(pending) + len(outputs) + int(target not in outputs), history_bytes)
            if exponent:
                state = _scale_operator(state, exponent)
            outputs[target] = outputs[target] + state if target in outputs else state
        elif record:
            bits = range(width) if classical_bits is None else classical_bits
            add_count("".join("1" if b in key else "0" for b in bits), count)
        else:
            weights = (
                abs(state) ** 2
                if pure
                else np.maximum(0, state.reshape(size, size).diagonal().real)
            )
            histogram = rng.multinomial(count, weights / weights.sum())
            for i in np.flatnonzero(histogram):
                add_count(format(int(i), f"0{n}b"), int(histogram[i]))
    return outputs
