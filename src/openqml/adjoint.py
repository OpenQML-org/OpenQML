"""Reversible NumPy differentiation of real quantum expectation values.

Inverse gates reconstruct intermediate states instead of retaining a gate tape.
The approach follows Jones and Gacon: https://arxiv.org/abs/2009.02823.
"""

import math
from collections.abc import Mapping, Set

import numpy as np

from .errors import ResourceError, ValidationError
from .ir import Circuit, integer
from .observables import Hamiltonian, PauliString
from .parameters import parameter_name, real
from .simulator import _X, _Y, _Z, _apply_matrix, _gate_matrix, _TensorOps


def _check_adjoint(circuit, max_qubits, max_memory_mb):
    if not isinstance(circuit, Circuit):
        raise ValidationError("Adjoint differentiation requires an OpenQML Circuit.")
    if circuit.has_effects:
        raise ValidationError("Adjoint differentiation requires a unitary circuit without effects.")
    integer(max_qubits, "max_qubits", 1)
    budget = real(max_memory_mb)
    if budget <= 0:
        raise ValidationError("max_memory_mb must be positive.")
    width = circuit.num_qubits
    local = max((len(op.qubits) for op in circuit.operations if op.name != "barrier"), default=1)
    log_bytes = math.log2(budget) + 20
    # Bound transient working arrays as well as the two persistent states.
    # Circuit storage, symbolic metadata and requested result arrays are separate.
    if (
        width >= 63
        or width > max_qubits
        or width + math.log2(12 * 16) > log_bytes
        or 2 * local + math.log2(8 * 16) > log_bytes
    ):
        raise ResourceError("Adjoint state or local matrix exceeds the configured working budget.")
    required = (12 * 16 * (1 << width) + 8 * 16 * (1 << (2 * local))) / (1 << 20)
    if required > budget:
        raise ResourceError("Adjoint state and local matrices exceed max_memory_mb.")


def _prepare(circuit, observables, bindings_batch, parameters, max_qubits, max_memory_mb):
    _check_adjoint(circuit, max_qubits, max_memory_mb)
    if isinstance(parameters, (str, bytes, Mapping, Set)):
        raise ValidationError("Gradient parameters require ordered circuit parameter names.")
    if isinstance(observables, (str, bytes, Mapping, Set)):
        raise ValidationError("Adjoint observables require an ordered iterable.")
    try:
        names = (
            circuit.parameters
            if parameters is None
            else tuple(parameter_name(n) for n in parameters)
        )
        obs = tuple(observables)
        rows = list(bindings_batch)
    except TypeError as exc:
        raise ValidationError(
            "Adjoint inputs require parameter, observable and binding iterables."
        ) from exc
    symbols = set(circuit.parameters)
    if len(set(names)) != len(names) or set(names) - symbols:
        raise ValidationError("Gradient parameters must be unique circuit parameter names.")
    if not obs:
        raise ValidationError("Adjoint gradients require observables.")
    for observable in obs:
        terms = observable.terms if isinstance(observable, Hamiltonian) else (observable,)
        if any(
            not isinstance(t, PauliString) or len(t.paulis) != circuit.num_qubits for t in terms
        ):
            raise ValidationError("Gradient observable width must match the circuit.")
    clean = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != symbols:
            raise ValidationError(
                "Adjoint bindings must match the circuit parameter names exactly."
            )
        values = {name: real(value) for name, value in row.items()}
        # Reject undefined expressions in every row before constructing a state.
        circuit.global_phase.evaluate(values)
        for op in circuit.operations:
            for angle in op.params:
                angle.evaluate(values)
        clean.append(values)
    positions = {name: i for i, name in enumerate(names)}
    derivatives = tuple(
        tuple(
            (
                index,
                tuple(
                    (positions[name], angle.derivative(name))
                    for name in sorted(angle.parameters & positions.keys())
                ),
            )
            for index, angle in enumerate(op.params)
            if angle.parameters & positions.keys()
        )
        for op in circuit.operations
    )
    return names, obs, clean, derivatives


def _real_array(value, shape, label):
    try:
        array = np.asarray(value)
        if array.dtype.kind not in "iuf":
            raise ValueError("Expected real entries.")
        array = array.astype(float)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError("Invalid shape or entries.")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{label} must be a finite real array with shape {shape}.") from exc
    return array


def _derivative_matrix(name, params, index, matrix):
    if name == "u":
        theta, phi, lam = params
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        a, b, ab = np.exp(1j * phi), np.exp(1j * lam), np.exp(1j * (phi + lam))
        return np.asarray(
            (
                [[-s / 2, -b * c / 2], [a * c / 2, -ab * s / 2]],
                [[0, 0], [1j * a * s, 1j * ab * c]],
                [[0, -1j * b * s], [0, 1j * ab * c]],
            )[index],
            dtype=complex,
        )
    if name in {"p", "cp"}:
        generator = np.zeros_like(matrix)
        generator[-1, -1] = 1j
    elif name in {"rx", "ry", "rz"}:
        generator = -0.5j * {"rx": _X, "ry": _Y, "rz": _Z}[name]
    elif name in {"crx", "cry", "crz"}:
        generator = np.zeros_like(matrix)
        generator[2:, 2:] = -0.5j * {"crx": _X, "cry": _Y, "crz": _Z}[name]
    elif name in {"rxx", "ryy", "rzz"}:
        pauli = {"rxx": _X, "ryy": _Y, "rzz": _Z}[name]
        generator = -0.5j * np.kron(pauli, pauli)
    else:
        raise ValidationError(f"No adjoint derivative for gate {name!r}.")
    return generator @ matrix


def _forward(circuit, bindings):
    ops = _TensorOps()
    state = np.zeros(1 << circuit.num_qubits, dtype=complex)
    state[0] = 1
    for op in circuit.operations:
        if op.name != "barrier":
            matrix = _gate_matrix(
                op.name, [p.evaluate(bindings) for p in op.params], ops, op.matrix
            )
            state = _apply_matrix(state, matrix, op.qubits, circuit.num_qubits, ops)
    # A common global phase cancels from real expectation values and gradients.
    return state


def _costate(state, observables, cotangents):
    terms = {}
    for observable, weight in zip(observables, cotangents, strict=True):
        for term in observable.terms if isinstance(observable, Hamiltonian) else (observable,):
            terms[term.paulis] = real(terms.get(term.paulis, 0.0) + weight * term.coefficient)
    result = np.zeros_like(state)
    for paulis, coefficient in terms.items():
        if not coefficient:
            continue
        index = np.arange(state.size, dtype=np.int64)
        index ^= sum(1 << (len(paulis) - q - 1) for q, p in enumerate(paulis) if p in "XY")
        transformed = state[index] * (1j ** paulis.count("Y"))
        for q, p in enumerate(paulis):
            if p in "YZ":
                transformed *= 1 - 2 * ((index >> (len(paulis) - q - 1)) & 1)
        result += coefficient * transformed
    return result


def _reverse(circuit, bindings, state, costate, derivatives, size):
    ops, result = _TensorOps(), np.zeros(size)
    for op, angles in zip(reversed(circuit.operations), reversed(derivatives), strict=True):
        if op.name == "barrier":
            continue
        params = [p.evaluate(bindings) for p in op.params]
        matrix = _gate_matrix(op.name, params, ops, op.matrix)
        before = _apply_matrix(state, matrix.conj().T, op.qubits, circuit.num_qubits, ops)
        for index, coefficients in angles:
            tangent = _apply_matrix(
                before,
                _derivative_matrix(op.name, params, index, matrix),
                op.qubits,
                circuit.num_qubits,
                ops,
            )
            local = 2 * np.vdot(costate, tangent).real
            for position, coefficient in coefficients:
                result[position] += local * coefficient.evaluate(bindings)
        costate = _apply_matrix(costate, matrix.conj().T, op.qubits, circuit.num_qubits, ops)
        state = before
    if not np.isfinite(result).all():
        raise ValidationError("Adjoint derivatives exceed finite real precision.")
    return result


def adjoint_vjp(
    circuit,
    observables,
    bindings_batch,
    cotangents,
    *,
    parameters=None,
    max_qubits=20,
    max_memory_mb=256,
):
    """Return (batch, parameters) reverse products with NumPy, without shifted circuits.

    Each row uses one forward and one reverse sweep with a constant number of
    state vectors. Shared/nonlinear symbols use the exact expression chain rule.
    Measurements, channels and feedback are rejected. This is a first-order
    numerical simulator API; it does not differentiate quantum hardware results.
    The working budget excludes existing circuit/metadata and output storage.
    """
    names, obs, rows, derivatives = _prepare(
        circuit, observables, bindings_batch, parameters, max_qubits, max_memory_mb
    )
    cotangents = _real_array(cotangents, (len(rows), len(obs)), "Cotangents")
    result = np.zeros((len(rows), len(names)))
    if not any(derivatives):
        return result
    for i, row in enumerate(rows):
        if np.any(cotangents[i]):
            state = _forward(circuit, row)
            result[i] = _reverse(
                circuit, row, state, _costate(state, obs, cotangents[i]), derivatives, len(names)
            )
    return result


def adjoint_jvp(
    circuit,
    observables,
    bindings_batch,
    tangents,
    *,
    parameters=None,
    max_qubits=20,
    max_memory_mb=256,
):
    """Return (batch, observables) directional products without a full Jacobian.

    Each row shares one forward state; each output uses a reverse sweep. Tangents
    follow parameters or circuit.parameters. Scope and budgets match adjoint_vjp.
    """
    names, obs, rows, derivatives = _prepare(
        circuit, observables, bindings_batch, parameters, max_qubits, max_memory_mb
    )
    tangents = _real_array(tangents, (len(rows), len(names)), "Tangents")
    result = np.zeros((len(rows), len(obs)))
    if not any(derivatives):
        return result
    for i, row in enumerate(rows):
        if np.any(tangents[i]):
            state = _forward(circuit, row)
            for j, observable in enumerate(obs):
                gradient = _reverse(
                    circuit,
                    row,
                    state,
                    _costate(state, (observable,), (1.0,)),
                    derivatives,
                    len(names),
                )
                result[i, j] = real(gradient @ tangents[i])
    return result
