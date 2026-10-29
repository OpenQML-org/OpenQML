"""Prepared, bounded parameter-shift products over portable circuit IR.

Plans compile derivative expressions and backend lowering once. Deterministic
backends can share identical primal points inside a bounded reuse window.
No bound inputs, shifted results or quantum states survive an evaluation.
"""

from dataclasses import dataclass, field, replace
from itertools import islice

import numpy as np

from ._gradient_common import _binding_columns, _names, _observables, _real_array
from .backends import ExecutionOptions, get_backend
from .batch import expectations_batch
from .errors import ValidationError
from .ir import Circuit, integer
from .parameters import Expression, expression, real

_TWO_SHIFTS = ((np.pi / 2, 0.5), (-np.pi / 2, -0.5))
_CONTROL_A = (np.sqrt(2) + 1) / (4 * np.sqrt(2))
_CONTROL_B = (1 - np.sqrt(2)) / (4 * np.sqrt(2))
_CONTROL_SHIFTS = (
    (np.pi / 2, _CONTROL_A),
    (-np.pi / 2, -_CONTROL_A),
    (3 * np.pi / 2, _CONTROL_B),
    (-3 * np.pi / 2, -_CONTROL_B),
)
_PROBABILITY_SHIFTS = ((1.0, 1.0), (0.0, -1.0))


def _coefficient(value, bindings):
    return real(value.evaluate(bindings)) if isinstance(value, Expression) else value


def _recipe(gate, base):
    if gate in {"bit_flip", "phase_flip", "depolarizing"}:
        return _PROBABILITY_SHIFTS, True
    if gate in {"amplitude_damping", "phase_damping"}:
        if base >= 1:
            raise ValidationError("Damping derivatives require an interior probability below 1.")
        scale = 1 / (2 * np.sqrt(1 - base))
        return (
            ((0.0, -scale), (1.0, scale))
            if gate == "phase_damping"
            else ((0.0, -2 + scale), (1.0, -2 + 3 * scale), (0.75, 4 - 4 * scale))
        ), True
    return (_CONTROL_SHIFTS if gate in {"crx", "cry", "crz"} else _TWO_SHIFTS), False


@dataclass(frozen=True)
class _Occurrence:
    path: tuple
    parameter_index: int
    gate: str
    derivatives: tuple


@dataclass(frozen=True)
class ShiftPlan:
    """Compile occurrence-wise Jacobians, JVPs and VJPs for a selected backend.

    parameters fixes source column ownership; method overrides select/reorder it.
    Inputs accept binding rows or named scalar/vector columns. max_reuse_rows
    bounds point grouping; chunk_size bounds submitted shifted circuits. Reuse
    requires the backend's deterministic_expectations contract. Shot estimators
    keep independent requests. Rebuild after changing backend configuration,
    lowering, or the registered backend instance.
    """

    circuit: Circuit
    observables: tuple
    parameters: tuple | None = None
    backend: object = "numpy"
    max_qubits: int | None = None
    max_memory_mb: float = 256
    max_branches: int = 256
    seed: int | None = None
    coalesce: bool | None = None
    max_reuse_rows: int = 256
    lowered_circuit: Circuit = field(init=False)
    _occurrences: tuple = field(init=False, repr=False)
    _symbols: tuple = field(init=False, repr=False)
    _lowered_symbols: tuple = field(init=False, repr=False)
    _execution_backend: object = field(init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.circuit, Circuit):
            raise ValidationError("Shift differentiation requires an OpenQML Circuit.")
        symbols = self.circuit.parameters
        names = _names(self.parameters, symbols)
        obs = _observables(self.observables, self.circuit.num_qubits, label="Shift")
        size = integer(self.max_reuse_rows, "max_reuse_rows", 1)
        ExecutionOptions(self.max_qubits, self.max_memory_mb, self.seed, self.max_branches)
        tensor = isinstance(self.backend, str) and self.backend in {"torch", "jax"}
        backend = self.backend if tensor else get_backend(self.backend)
        exact = tensor or backend.capabilities.deterministic_expectations
        if self.coalesce is not None and type(self.coalesce) is not bool:
            raise ValidationError("coalesce must be a boolean or None.")
        if self.coalesce and not exact:
            raise ValidationError(
                "Job coalescing requires deterministic_expectations=True on the backend."
            )
        prepare = None if tensor else getattr(backend, "prepare_gradient_circuit", None)
        if prepare is not None and not callable(prepare):
            raise ValidationError("prepare_gradient_circuit must be callable.")
        lowered = self.circuit if prepare is None else prepare(self.circuit)
        if (
            not isinstance(lowered, Circuit)
            or lowered.num_qubits != self.circuit.num_qubits
            or set(lowered.parameters) - set(symbols)
        ):
            raise ValidationError(
                "Backend gradient preparation must preserve wires and parameter names."
            )
        positions = {name: index for index, name in enumerate(names)}
        derivatives, occurrences = {}, []
        for path, op in lowered.walk_operations():
            for index, angle in enumerate(op.params):
                partials = []
                for name in sorted(angle.parameters & positions.keys(), key=positions.__getitem__):
                    key = (id(angle), name)
                    if key not in derivatives:
                        derivative = angle.derivative(name)
                        derivatives[key] = (
                            derivative if derivative.parameters else real(derivative.evaluate({}))
                        )
                    partials.append((positions[name], derivatives[key]))
                if partials:
                    occurrences.append(_Occurrence(path, index, op.name, tuple(partials)))
        for name, value in (
            ("parameters", names),
            ("observables", obs),
            ("max_reuse_rows", size),
            ("coalesce", exact if self.coalesce is None else self.coalesce),
            ("lowered_circuit", lowered),
            ("_occurrences", tuple(occurrences)),
            ("_symbols", symbols),
            ("_lowered_symbols", lowered.parameters),
            ("_execution_backend", backend),
        ):
            object.__setattr__(self, name, value)

    @property
    def occurrence_count(self):
        return len(self._occurrences)

    def _key(self, columns, index):
        # Full float bytes, rather than a digest, distinguish every bound point,
        # including signed zero. These keys exist only within this invocation.
        return np.asarray([columns[name][index] for name in self._symbols], dtype="<f8").tobytes()

    def _bindings(self, bindings, batch_size):
        columns, count = _binding_columns(bindings, self._symbols, batch_size, label="Shift")
        checked = set()
        for index in range(count):
            key = self._key(columns, index)
            if key in checked:
                continue
            values = {name: columns[name][index] for name in self._symbols}
            # Scalar IR binding is the authoritative domain check. Preflight
            # all source and lowered rows before submitting any external work.
            self.circuit.bind(values)
            if self.lowered_circuit is not self.circuit:
                self.lowered_circuit.bind({name: values[name] for name in self._lowered_symbols})
            if len(checked) >= self.max_reuse_rows:
                checked.clear()
            checked.add(key)
        return columns, count

    def _selection(self, parameters):
        names = _names(parameters, self.parameters)
        source_positions = {name: index for index, name in enumerate(self.parameters)}
        positions = {source_positions[name]: index for index, name in enumerate(names)}
        occurrences = tuple(
            (
                occurrence,
                tuple((positions[p], expr) for p, expr in occurrence.derivatives if p in positions),
            )
            for occurrence in self._occurrences
            if any(p in positions for p, _ in occurrence.derivatives)
        )
        return names, occurrences

    def _options(self, overrides):
        options = dict(
            max_qubits=self.max_qubits,
            max_memory_mb=self.max_memory_mb,
            max_branches=self.max_branches,
            seed=self.seed,
        )
        if set(overrides) - set(options):
            raise ValidationError("Unknown shift execution option.")
        options.update(overrides)
        ExecutionOptions(**options)
        return options

    def _backend(self):
        if isinstance(self.backend, str) and self.backend not in {"torch", "jax"}:
            if get_backend(self.backend) is not self._execution_backend:
                raise ValidationError(
                    "Registered backend changed; recreate the shift plan or layer."
                )
        if (
            self.coalesce
            and not isinstance(self._execution_backend, str)
            and not self._execution_backend.capabilities.deterministic_expectations
        ):
            raise ValidationError("Backend determinism changed; recreate the shift plan or layer.")
        return self._execution_backend

    def _groups(self, columns, count):
        if not self.coalesce:
            yield from ((index,) for index in range(count))
            return
        for start in range(0, count, self.max_reuse_rows):
            groups = {}
            for index in range(start, min(count, start + self.max_reuse_rows)):
                groups.setdefault(self._key(columns, index), []).append(index)
            yield from (tuple(indices) for indices in groups.values())

    def _tasks(self, columns, count, occurrences, mode, vectors):
        for members in self._groups(columns, count):
            indices = np.asarray(members, dtype=int)
            if mode == "vjp" and self.coalesce:
                indices = indices[np.any(vectors[indices] != 0, axis=-1)]
            if not indices.size:
                continue
            values = {name: columns[name][indices[0]] for name in self._lowered_symbols}
            bound = None
            for occurrence, partials in occurrences:
                if mode == "jvp":
                    if len(indices) == 1:
                        row = vectors[indices[0]]
                        direction = sum(
                            _coefficient(expr, values) * row[p]
                            for p, expr in partials
                            if row[p] != 0
                        )
                        if direction == 0:
                            continue
                        recipients, coefficients = indices, (direction,)
                    else:
                        direction = np.zeros(len(indices))
                        for position, expr in partials:
                            selected = vectors[indices, position]
                            if np.any(selected != 0):
                                direction += _coefficient(expr, values) * selected
                        keep = direction != 0
                        recipients, coefficients = indices[keep], direction[keep]
                        if not recipients.size:
                            continue
                else:
                    coefficients = tuple(
                        (position, value)
                        for position, expr in partials
                        if (value := _coefficient(expr, values)) != 0
                    )
                    if not coefficients:
                        continue
                    recipients = indices
                if bound is None:
                    bound = self.lowered_circuit.bind(values)
                operation = bound.operation_at(occurrence.path)
                base = operation.params[occurrence.parameter_index].constant
                shifts, absolute = _recipe(occurrence.gate, base)
                for shift, weight in shifts:
                    params = list(operation.params)
                    params[occurrence.parameter_index] = expression(
                        shift if absolute else base + shift
                    )
                    job = bound.replace_operation(
                        occurrence.path, replace(operation, params=tuple(params))
                    )
                    yield recipients, coefficients, weight, job

    def _evaluate(self, columns, count, occurrences, mode, vectors, result, chunk_size, options):
        size = integer(chunk_size, "chunk_size", 2)
        options = self._options(options)
        backend = self._backend()
        tasks = iter(self._tasks(columns, count, occurrences, mode, vectors))
        while group := list(islice(tasks, size)):
            values = expectations_batch(
                [job for _, _, _, job in group],
                self.observables,
                backend=backend,
                **options,
            )
            for value, (rows, coefficients, weight, _) in zip(values, group, strict=True):
                delta = weight * value
                if len(rows) == 1:
                    row = rows[0]
                    if mode == "jvp":
                        result[row] += coefficients[0] * delta
                    else:
                        product = float(vectors[row] @ delta) if mode == "vjp" else delta
                        for position, coefficient in coefficients:
                            if mode == "vjp":
                                result[row, position] += coefficient * product
                            else:
                                result[row, :, position] += coefficient * product
                elif mode == "jvp":
                    result[rows] += coefficients[:, None] * delta
                else:
                    product = vectors[rows] @ delta if mode == "vjp" else delta
                    for position, coefficient in coefficients:
                        if mode == "vjp":
                            result[rows, position] += coefficient * product
                        else:
                            result[rows, :, position] += coefficient * product
        if not np.isfinite(result).all():
            raise ValidationError("Shift derivatives exceed finite real precision.")
        return result

    def jacobian(self, bindings, *, parameters=None, chunk_size=64, batch_size=None, **options):
        """Return the explicitly requested (batch, readouts, parameters) Jacobian."""
        names, occurrences = self._selection(parameters)
        columns, count = self._bindings(bindings, batch_size)
        result = np.zeros((count, len(self.observables), len(names)))
        return self._evaluate(
            columns, count, occurrences, "jacobian", None, result, chunk_size, options
        )

    def vjp(
        self, bindings, cotangents, *, parameters=None, chunk_size=64, batch_size=None, **options
    ):
        """Stream (batch, parameters) contractions without a dense Jacobian."""
        names, occurrences = self._selection(parameters)
        columns, count = self._bindings(bindings, batch_size)
        vectors = _real_array(cotangents, (count, len(self.observables)), "Cotangents")
        result = np.zeros((count, len(names)))
        return self._evaluate(
            columns, count, occurrences, "vjp", vectors, result, chunk_size, options
        )

    def jvp(
        self, bindings, tangents, *, parameters=None, chunk_size=64, batch_size=None, **options
    ):
        """Stream (batch, readouts) directions; cancelled occurrences submit no jobs."""
        names, occurrences = self._selection(parameters)
        columns, count = self._bindings(bindings, batch_size)
        vectors = _real_array(tangents, (count, len(names)), "Tangents")
        result = np.zeros((count, len(self.observables)))
        return self._evaluate(
            columns, count, occurrences, "jvp", vectors, result, chunk_size, options
        )


def parameter_shift_jacobian(
    circuit,
    observables,
    bindings_batch,
    *,
    parameters=None,
    chunk_size=64,
    backend="numpy",
    **options,
):
    """Return a (batch, readouts, parameters) Jacobian in source parameter order."""
    return ShiftPlan(circuit, observables, parameters, backend).jacobian(
        bindings_batch, chunk_size=chunk_size, **options
    )


def parameter_shift_vjp(
    circuit,
    observables,
    bindings_batch,
    cotangents,
    *,
    parameters=None,
    chunk_size=64,
    backend="numpy",
    **options,
):
    """Return streamed (batch, parameters) cotangent contractions."""
    return ShiftPlan(circuit, observables, parameters, backend).vjp(
        bindings_batch, cotangents, chunk_size=chunk_size, **options
    )


def parameter_shift_jvp(
    circuit,
    observables,
    bindings_batch,
    tangents,
    *,
    parameters=None,
    chunk_size=64,
    backend="numpy",
    **options,
):
    """Return streamed (batch, readouts) directional derivatives."""
    return ShiftPlan(circuit, observables, parameters, backend).jvp(
        bindings_batch, tangents, chunk_size=chunk_size, **options
    )


def parameter_shift(circuit, observable, bindings, **options):
    names = options.get("parameters")
    names = circuit.parameters if names is None else tuple(names)
    options["parameters"] = names
    jacobian = parameter_shift_jacobian(circuit, (observable,), (bindings,), **options)[0, 0]
    return dict(zip(names, jacobian.tolist(), strict=True))
