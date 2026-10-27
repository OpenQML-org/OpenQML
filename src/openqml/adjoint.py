"""Prepared, bounded NumPy differentiation of real quantum expectation values.

Inverse gates reconstruct states during reverse products; directional products
propagate a tangent state. See https://arxiv.org/abs/2009.02823 for the adjoint
recurrence. Plans retain symbolic metadata, never bindings or quantum states.
"""

import math
from collections.abc import Mapping, Set
from dataclasses import dataclass, field

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
    # Conservative transient arrays per row, including batched local matrices.
    # Existing circuit, symbolic/input metadata and result arrays are separate.
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
    return required


def _names(values, available):
    if isinstance(values, (str, bytes, Mapping, Set)):
        raise ValidationError("Gradient parameters require ordered circuit parameter names.")
    try:
        names = tuple(available) if values is None else tuple(parameter_name(n) for n in values)
    except TypeError as exc:
        raise ValidationError(
            "Gradient parameters require ordered circuit parameter names."
        ) from exc
    if len(set(names)) != len(names) or set(names) - set(available):
        raise ValidationError("Gradient parameters must be unique circuit parameter names.")
    return names


def _observables(values, width):
    if isinstance(values, (str, bytes, Mapping, Set)):
        raise ValidationError("Adjoint observables require an ordered iterable.")
    try:
        obs = tuple(values)
    except TypeError as exc:
        raise ValidationError("Adjoint observables require an ordered iterable.") from exc
    if not obs:
        raise ValidationError("Adjoint gradients require observables.")
    for observable in obs:
        terms = observable.terms if isinstance(observable, Hamiltonian) else (observable,)
        if any(not isinstance(t, PauliString) or len(t.paulis) != width for t in terms):
            raise ValidationError("Gradient observable width must match the circuit.")
    return obs


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
        return _TensorOps().matrix(
            (
                [[-s / 2, -b * c / 2], [a * c / 2, -ab * s / 2]],
                [[0, 0], [1j * a * s, 1j * ab * c]],
                [[0, -1j * b * s], [0, 1j * ab * c]],
            )[index]
        )
    if name in {"p", "cp"}:
        generator = np.zeros_like(matrix)
        generator[..., -1, -1] = 1j
    elif name in {"rx", "ry", "rz"}:
        generator = -0.5j * {"rx": _X, "ry": _Y, "rz": _Z}[name]
    elif name in {"crx", "cry", "crz"}:
        generator = np.zeros_like(matrix)
        generator[..., 2:, 2:] = -0.5j * {"crx": _X, "cry": _Y, "crz": _Z}[name]
    elif name in {"rxx", "ryy", "rzz"}:
        pauli = {"rxx": _X, "ryy": _Y, "rzz": _Z}[name]
        generator = -0.5j * np.kron(pauli, pauli)
    else:
        raise ValidationError(f"No adjoint derivative for gate {name!r}.")
    return generator @ matrix


def _forward(circuit, bindings, batch_size):
    ops = _TensorOps()
    state = np.zeros((batch_size, 1 << circuit.num_qubits), dtype=complex)
    state[:, 0] = 1
    for op in circuit.operations:
        if op.name != "barrier":
            matrix = _gate_matrix(
                op.name, [p.evaluate(bindings) for p in op.params], ops, op.matrix
            )
            state = _apply_matrix(state, matrix, op.qubits, circuit.num_qubits, ops)
    # Common global phase cancels from real expectations and their derivatives.
    return state


def _pauli_action(state, paulis):
    index = np.arange(state.shape[-1], dtype=np.int64)
    index ^= sum(1 << (len(paulis) - q - 1) for q, p in enumerate(paulis) if p in "XY")
    transformed = state[..., index] * (1j ** paulis.count("Y"))
    for q, p in enumerate(paulis):
        if p in "YZ":
            transformed *= 1 - 2 * ((index >> (len(paulis) - q - 1)) & 1)
    return transformed


def _observable_action(state, observable):
    result = np.zeros_like(state)
    for term in observable.terms if isinstance(observable, Hamiltonian) else (observable,):
        if term.coefficient:
            result += term.coefficient * _pauli_action(state, term.paulis)
    return result


def _costate(state, observables, cotangents):
    terms = {}
    for j, observable in enumerate(observables):
        for term in observable.terms if isinstance(observable, Hamiltonian) else (observable,):
            terms[term.paulis] = terms.get(term.paulis, 0.0) + cotangents[:, j] * term.coefficient
    result = np.zeros_like(state)
    for paulis, coefficient in terms.items():
        if not np.isfinite(coefficient).all():
            raise ValidationError("Adjoint readout coefficients exceed finite real precision.")
        if np.any(coefficient):
            result += coefficient[:, None] * _pauli_action(state, paulis)
    return result


def _reverse(circuit, bindings, state, costate, derivatives, size):
    ops, result = _TensorOps(), np.zeros((state.shape[0], size))
    for op, angles in zip(reversed(circuit.operations), reversed(derivatives), strict=True):
        if op.name == "barrier":
            continue
        params = [p.evaluate(bindings) for p in op.params]
        matrix = _gate_matrix(op.name, params, ops, op.matrix)
        inverse = matrix.conj().swapaxes(-1, -2)
        before = _apply_matrix(state, inverse, op.qubits, circuit.num_qubits, ops)
        for index, coefficients in angles:
            tangent = _apply_matrix(
                before,
                _derivative_matrix(op.name, params, index, matrix),
                op.qubits,
                circuit.num_qubits,
                ops,
            )
            local = 2 * np.real(np.sum(np.conj(costate) * tangent, axis=-1))
            for position, coefficient in coefficients:
                result[:, position] += local * coefficient.evaluate(bindings)
        costate = _apply_matrix(costate, inverse, op.qubits, circuit.num_qubits, ops)
        state = before
    if not np.isfinite(result).all():
        raise ValidationError("Adjoint derivatives exceed finite real precision.")
    return result


def _angle_direction(coefficients, bindings, tangents):
    result = np.zeros(tangents.shape[0])
    for position, coefficient in coefficients:
        active = tangents[:, position] != 0
        if np.any(active):
            local = (
                bindings
                if np.all(active)
                else {name: bindings[name][active] for name in coefficient.parameters}
            )
            result[active] += coefficient.evaluate(local) * tangents[active, position]
    if not np.isfinite(result).all():
        raise ValidationError("Directional derivatives exceed finite real precision.")
    return result


def _tangent_forward(circuit, bindings, derivatives, tangents):
    ops = _TensorOps()
    state = np.zeros((len(tangents), 1 << circuit.num_qubits), dtype=complex)
    state[:, 0] = 1
    tangent, active = np.zeros_like(state), False
    for op, angles in zip(circuit.operations, derivatives, strict=True):
        if op.name == "barrier":
            continue
        params = [p.evaluate(bindings) for p in op.params]
        matrix = _gate_matrix(op.name, params, ops, op.matrix)
        evolved = (
            _apply_matrix(tangent, matrix, op.qubits, circuit.num_qubits, ops)
            if active
            else tangent
        )
        for index, coefficients in angles:
            direction = _angle_direction(coefficients, bindings, tangents)
            if np.any(direction):
                evolved += direction[:, None] * _apply_matrix(
                    state,
                    _derivative_matrix(op.name, params, index, matrix),
                    op.qubits,
                    circuit.num_qubits,
                    ops,
                )
                active = True
        state = _apply_matrix(state, matrix, op.qubits, circuit.num_qubits, ops)
        tangent = evolved
    return state, tangent


@dataclass(frozen=True, eq=False)
class AdjointPlan:
    """Reusable unitary expectation/gradient plan, without model values or states.

    Preparation differentiates symbolic angles once. All rows are validated
    before state allocation; memory budgets can shrink requested batch chunks.
    Parameter overrides select/reorder compiled columns without differentiating
    expressions again. Numerical products remain first order and CPU-only.
    Bindings accept rows of real name mappings or named scalar/vector columns.
    batch_size sets the row count for scalar columns or parameter-free circuits.
    """

    circuit: Circuit
    observables: tuple
    parameters: tuple | None = None
    max_qubits: int = 20
    max_memory_mb: float = 256
    _derivatives: tuple = field(init=False, repr=False)
    _symbols: tuple = field(init=False, repr=False)
    _row_mib: float = field(init=False, repr=False)

    def __post_init__(self):
        required = _check_adjoint(self.circuit, self.max_qubits, self.max_memory_mb)
        symbols = self.circuit.parameters
        names = _names(self.parameters, symbols)
        obs = _observables(self.observables, self.circuit.num_qubits)
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
            for op in self.circuit.operations
        )
        for name, value in (
            ("parameters", names),
            ("observables", obs),
            ("max_qubits", integer(self.max_qubits, "max_qubits", 1)),
            ("max_memory_mb", real(self.max_memory_mb)),
            ("_derivatives", derivatives),
            ("_symbols", symbols),
            ("_row_mib", required),
        ):
            object.__setattr__(self, name, value)

    def _bindings(self, bindings_batch, batch_size):
        if batch_size is not None:
            batch_size = integer(batch_size, "batch_size")
        if isinstance(bindings_batch, Mapping):
            if set(bindings_batch) != set(self._symbols):
                raise ValidationError(
                    "Adjoint bindings must match the circuit parameter names exactly."
                )
            arrays = {}
            try:
                for name, value in bindings_batch.items():
                    array = np.asarray(value)
                    if array.ndim > 1 or array.dtype.kind not in "iuf":
                        raise ValueError("Binding columns require real scalars or vectors.")
                    arrays[name] = array
                shape = np.broadcast_shapes(*(value.shape for value in arrays.values()))
                count = (shape[0] if shape else 1) if batch_size is None else batch_size
                columns = {
                    name: np.array(np.broadcast_to(value, (count,)), dtype=float, copy=True)
                    for name, value in arrays.items()
                }
                if any(not np.isfinite(value).all() for value in columns.values()):
                    raise ValueError("Binding columns must be finite.")
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValidationError(
                    "Adjoint binding columns require broadcastable finite real vectors."
                ) from exc
        else:
            columns, count = self._binding_rows(bindings_batch)
            if batch_size is not None and batch_size != count:
                raise ValidationError("batch_size differs from the binding row count.")
        # Preflight every angle, including phase, before allocating any quantum state.
        self.circuit.global_phase.evaluate(columns)
        for op in self.circuit.operations:
            for angle in op.params:
                angle.evaluate(columns)
        return columns, count

    def _binding_rows(self, bindings_batch):
        try:
            rows = list(bindings_batch)
        except TypeError as exc:
            raise ValidationError("Adjoint bindings require an iterable of name mappings.") from exc
        columns = {name: np.empty(len(rows)) for name in self._symbols}
        expected = set(self._symbols)
        for i, row in enumerate(rows):
            if not isinstance(row, Mapping) or set(row) != expected:
                raise ValidationError(
                    "Adjoint bindings must match the circuit parameter names exactly."
                )
            for name in self._symbols:
                columns[name][i] = real(row[name])
        return columns, len(rows)

    def _selection(self, parameters):
        names = _names(parameters, self.parameters)
        if names == self.parameters:
            return names, self._derivatives
        positions = {self.parameters.index(name): i for i, name in enumerate(names)}
        derivatives = tuple(
            tuple(
                (index, tuple((positions[p], expr) for p, expr in coefficients if p in positions))
                for index, coefficients in angles
                if any(p in positions for p, _ in coefficients)
            )
            for angles in self._derivatives
        )
        return names, derivatives

    def _chunk_size(self, count, requested):
        size = 128 if requested is None else integer(requested, "batch_chunk_size", 1)
        size = min(size, max(1, count))
        if self._row_mib * size > self.max_memory_mb:
            size = max(1, math.floor(self.max_memory_mb / self._row_mib))
        return size

    def expectations(self, bindings_batch, *, batch_chunk_size=None, batch_size=None):
        """Evaluate ordered real readouts directly, without constructing bound IR copies."""
        bindings, count = self._bindings(bindings_batch, batch_size)
        size = self._chunk_size(count, batch_chunk_size)
        result = np.empty((count, len(self.observables)))
        for start in range(0, count, size):
            end = min(start + size, count)
            state = _forward(
                self.circuit, {n: v[start:end] for n, v in bindings.items()}, end - start
            )
            for j, observable in enumerate(self.observables):
                result[start:end, j] = np.real(
                    np.sum(np.conj(state) * _observable_action(state, observable), axis=-1)
                )
        if not np.isfinite(result).all():
            raise ValidationError("Expectation values exceed finite real precision.")
        return result

    def vjp(
        self, bindings_batch, cotangents, *, parameters=None, batch_chunk_size=None, batch_size=None
    ):
        """Return (batch, selected parameters) products with one batched reverse sweep."""
        names, derivatives = self._selection(parameters)
        bindings, count = self._bindings(bindings_batch, batch_size)
        values = _real_array(cotangents, (count, len(self.observables)), "Cotangents")
        size = self._chunk_size(count, batch_chunk_size)
        result = np.zeros((count, len(names)))
        if not any(derivatives):
            return result
        for start in range(0, count, size):
            indices = np.flatnonzero(np.any(values[start : start + size] != 0, axis=-1)) + start
            if indices.size:
                local = {name: value[indices] for name, value in bindings.items()}
                state = _forward(self.circuit, local, len(indices))
                result[indices] = _reverse(
                    self.circuit,
                    local,
                    state,
                    _costate(state, self.observables, values[indices]),
                    derivatives,
                    len(names),
                )
        return result

    def jvp(
        self, bindings_batch, tangents, *, parameters=None, batch_chunk_size=None, batch_size=None
    ):
        """Return (batch, readouts) products with one forward tangent sweep per chunk."""
        names, derivatives = self._selection(parameters)
        bindings, count = self._bindings(bindings_batch, batch_size)
        values = _real_array(tangents, (count, len(names)), "Tangents")
        size = self._chunk_size(count, batch_chunk_size)
        result = np.zeros((count, len(self.observables)))
        if not any(derivatives):
            return result
        for start in range(0, count, size):
            indices = np.flatnonzero(np.any(values[start : start + size] != 0, axis=-1)) + start
            if not indices.size:
                continue
            local = {name: value[indices] for name, value in bindings.items()}
            direction = values[indices]
            if not any(
                np.any(_angle_direction(coefficients, local, direction))
                for angles in derivatives
                for _, coefficients in angles
            ):
                continue
            state, tangent = _tangent_forward(self.circuit, local, derivatives, direction)
            for j, observable in enumerate(self.observables):
                result[indices, j] = 2 * np.real(
                    np.sum(np.conj(tangent) * _observable_action(state, observable), axis=-1)
                )
        if not np.isfinite(result).all():
            raise ValidationError("Directional derivatives exceed finite real precision.")
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
    batch_chunk_size=None,
):
    """Return bounded, first-order (batch, parameters) NumPy reverse products.

    Each chunk uses a constant number of state vectors independent of gate count.
    The working budget excludes existing circuit/input metadata and result arrays.
    Use AdjointPlan to reuse symbolic preparation across repeated calls.
    """
    return AdjointPlan(circuit, observables, parameters, max_qubits, max_memory_mb).vjp(
        bindings_batch, cotangents, batch_chunk_size=batch_chunk_size
    )


def adjoint_jvp(
    circuit,
    observables,
    bindings_batch,
    tangents,
    *,
    parameters=None,
    max_qubits=20,
    max_memory_mb=256,
    batch_chunk_size=None,
):
    """Return (batch, observables) products using one forward tangent sweep per chunk.

    No full Jacobian or per-output reverse sweeps are constructed. Scope and
    working budgets match adjoint_vjp; only unitary simulation is supported.
    """
    return AdjointPlan(circuit, observables, parameters, max_qubits, max_memory_mb).jvp(
        bindings_batch, tangents, batch_chunk_size=batch_chunk_size
    )
