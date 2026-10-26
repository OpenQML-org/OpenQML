"""Bounded, occurrence-wise parameter-shift Jacobians and directional products."""

from dataclasses import replace
from itertools import islice

import numpy as np

from .backends import get_backend
from .batch import expectations_batch
from .errors import ValidationError
from .ir import integer
from .observables import Hamiltonian, PauliString
from .parameters import expression


def _prepare(circuit, observables, bindings_batch, parameters, chunk_size, backend):
    integer(chunk_size, "chunk_size", 2)
    names = circuit.parameters if parameters is None else tuple(parameters)
    if len(set(names)) != len(names) or set(names) - set(circuit.parameters):
        raise ValidationError("Gradient parameters must be unique circuit parameter names.")
    rows, observables = list(bindings_batch), tuple(observables)
    if not observables:
        raise ValidationError("Shift gradients require observables.")
    for observable in observables:
        terms = observable.terms if isinstance(observable, Hamiltonian) else (observable,)
        if any(
            not isinstance(t, PauliString) or len(t.paulis) != circuit.num_qubits for t in terms
        ):
            raise ValidationError("Gradient observable width must match the circuit.")
    # Validate every row before submitting any work to a possibly external engine.
    for bindings in rows:
        circuit.bind(bindings)
    device = (
        None if isinstance(backend, str) and backend in {"torch", "jax"} else get_backend(backend)
    )
    prepare = getattr(device, "prepare_gradient_circuit", None)
    prepared = circuit if prepare is None else prepare(circuit)
    from .ir import Circuit

    if (
        not isinstance(prepared, Circuit)
        or prepared.num_qubits != circuit.num_qubits
        or set(prepared.parameters) - set(circuit.parameters)
    ):
        raise ValidationError(
            "Backend gradient preparation must preserve wires and parameter names."
        )
    return names, rows, observables, prepared


def _tasks(lowered, rows, names, directions=None):
    selected = set(names)
    occurrences = [
        (index, pindex, {name: angle.derivative(name) for name in angle.parameters & selected})
        for index, op in lowered.walk_operations()
        for pindex, angle in enumerate(op.params)
        if angle.parameters & selected
    ]
    for row_index, bindings in enumerate(rows):
        active = {name: bindings[name] for name in lowered.parameters}
        bound = lowered.bind(active)
        for index, pindex, derivatives in occurrences:
            coefficients = {
                name: float(value.evaluate(active)) for name, value in derivatives.items()
            }
            coefficients = {name: value for name, value in coefficients.items() if value != 0}
            if directions is not None:
                direction = sum(
                    coefficient * directions[row_index][name]
                    for name, coefficient in coefficients.items()
                )
                coefficients = {"direction": direction} if direction else {}
            if not coefficients:
                continue
            name = lowered.operation_at(index).name
            base = bound.operation_at(index).params[pindex].constant
            absolute = False
            if name in {"bit_flip", "phase_flip", "depolarizing"}:
                shifts = ((1.0, 1.0), (0.0, -1.0))
                absolute = True
            elif name in {"amplitude_damping", "phase_damping"}:
                if base >= 1:
                    raise ValidationError(
                        "Damping derivatives require an interior probability below 1."
                    )
                scale = 1 / (2 * np.sqrt(1 - base))
                shifts = (
                    ((0.0, -scale), (1.0, scale))
                    if name == "phase_damping"
                    else ((0.0, -2 + scale), (1.0, -2 + 3 * scale), (0.75, 4 - 4 * scale))
                )
                absolute = True
            elif name in {"crx", "cry", "crz"}:
                a = (np.sqrt(2) + 1) / (4 * np.sqrt(2))
                b = (1 - np.sqrt(2)) / (4 * np.sqrt(2))
                shifts = (
                    (np.pi / 2, a),
                    (-np.pi / 2, -a),
                    (3 * np.pi / 2, b),
                    (-3 * np.pi / 2, -b),
                )
            else:
                shifts = ((np.pi / 2, 0.5), (-np.pi / 2, -0.5))
            for shift, weight in shifts:
                params = list(bound.operation_at(index).params)
                params[pindex] = expression(shift if absolute else params[pindex].constant + shift)
                job = bound.replace_operation(
                    index, replace(bound.operation_at(index), params=tuple(params))
                )
                yield row_index, coefficients, weight, job


def _deltas(lowered, rows, names, observables, chunk_size, backend, options, directions=None):
    pending = iter(_tasks(lowered, rows, names, directions))
    while group := list(islice(pending, chunk_size)):
        jobs = [job for _, _, _, job in group]
        values = expectations_batch(jobs, observables, backend=backend, **options)
        for index, (row_index, coefficients, weight, _) in enumerate(group):
            yield row_index, coefficients, weight * values[index]


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
    """Return a (batch, observables, parameters) Jacobian using chunked shifts.

    Columns follow parameters or circuit.parameters. Shared and nonlinear
    angles use the chain rule.
    """
    names, rows, observables, lowered = _prepare(
        circuit, observables, bindings_batch, parameters, chunk_size, backend
    )
    result = np.zeros((len(rows), len(observables), len(names)))
    positions = {name: i for i, name in enumerate(names)}
    for row, coefficients, delta in _deltas(
        lowered, rows, names, observables, chunk_size, backend, options
    ):
        for name, coefficient in coefficients.items():
            result[row, :, positions[name]] += coefficient * delta
    return result


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
    """Return streamed (batch, parameters) cotangent contractions.

    Sum shared weight columns across rows; input gradients keep their row axis.
    chunk_size bounds queued shifted circuits.
    """
    names, rows, observables, lowered = _prepare(
        circuit, observables, bindings_batch, parameters, chunk_size, backend
    )
    cotangents = np.asarray(cotangents, dtype=float)
    if cotangents.shape != (len(rows), len(observables)) or not np.isfinite(cotangents).all():
        raise ValidationError("Cotangents must be a finite (batch, observables) array.")
    result = np.zeros((len(rows), len(names)))
    positions = {name: i for i, name in enumerate(names)}
    for row, coefficients, delta in _deltas(
        lowered, rows, names, observables, chunk_size, backend, options
    ):
        contracted = float(cotangents[row] @ delta)
        for name, coefficient in coefficients.items():
            result[row, positions[name]] += coefficient * contracted
    return result


def parameter_shift(circuit, observable, bindings, **options):
    names = options.get("parameters")
    names = circuit.parameters if names is None else tuple(names)
    options["parameters"] = names
    jacobian = parameter_shift_jacobian(circuit, (observable,), (bindings,), **options)[0, 0]
    return dict(zip(names, jacobian.tolist(), strict=True))


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
    """Return streamed (batch, observables) directional derivatives.

    Finite real tangents have shape (batch, parameters), ordered by parameters
    or circuit.parameters. Zero occurrence directions submit no shifted jobs.
    This numerical API does not build an autodiff graph.
    """
    names, rows, observables, lowered = _prepare(
        circuit, observables, bindings_batch, parameters, chunk_size, backend
    )
    try:
        tangents = np.asarray(tangents)
        if tangents.dtype.kind not in "iuf":
            raise ValueError("Directions must contain real numbers.")
        tangents = tangents.astype(float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("Tangents must be a finite real (batch, parameters) array.") from exc
    if tangents.shape != (len(rows), len(names)) or not np.isfinite(tangents).all():
        raise ValidationError("Tangents must be a finite real (batch, parameters) array.")
    directions = [dict(zip(names, row, strict=True)) for row in tangents]
    result = np.zeros((len(rows), len(observables)))
    for row, coefficient, delta in _deltas(
        lowered, rows, names, observables, chunk_size, backend, options, directions
    ):
        result[row] += coefficient["direction"] * delta
    return result
