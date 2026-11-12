"""Prepare one topology and vectorize scalar bindings without bound IR copies."""

from dataclasses import dataclass

import numpy as np

from ..compute.kernel import get_compute_kernel
from ..errors import ValidationError, _as_tuple, _require_mapping
from ..ir.parameters import Parameter, parameter_name, real


@dataclass(frozen=True)
class _Program:
    source: object
    template: object
    expressions: tuple


def _program(circuit):
    def build():
        expressions, names = [], {}

        def rewrite(angle):
            key = id(angle)
            if key in names:
                return names[key]
            if not angle.parameters:
                return angle
            value = Parameter(f"_openqml_batch_{len(expressions)}")
            names[key] = value
            expressions.append((value.name, angle))
            return value

        template = circuit._map_parameters(rewrite)
        return _Program(circuit, template, tuple(expressions))

    return get_compute_kernel().plan("parameter-batch", id(circuit), build)


def numpy_expectations(circuit, observables, bindings, options, device):
    """Return None before consuming inputs when the optional lowering cannot fit."""
    from ..runtime._validation import validate_expectations, validate_states
    from .batch import _limits, _observables
    from .dense import _expectations_from_state, _resource_check, _TensorOps, statevector

    kernel = get_compute_kernel()
    budget = kernel._prepare_budget(options.max_memory_mb)
    if budget.max_bytes >> (circuit.num_qubits + 6) < 1:
        return None
    row_bytes = 64 * (1 << circuit.num_qubits)
    row_bytes += 16 * (
        len(circuit.parameters) + 1 + sum(len(op.params) for op in circuit.operations)
    )
    capacity = budget.max_bytes // row_bytes
    if capacity < 1:
        return None
    readouts = _observables(observables, (circuit,))
    rows = ({},) if bindings is None else _as_tuple(bindings, "batch bindings")
    expected, clean = set(circuit.parameters), []
    for index, row in enumerate(rows):
        _require_mapping(row, f"bindings[{index}]")
        row = {parameter_name(name): real(value) for name, value in row.items()}
        missing, unknown = expected - row.keys(), row.keys() - expected
        if missing or unknown:
            raise ValidationError(
                f"Parameter mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}."
            )
        clean.append(row)
    output = np.empty((len(rows), len(readouts)))
    if not rows:
        return output
    _limits((circuit,), device, options.max_qubits)
    _resource_check(circuit, options.max_qubits, options.max_memory_mb)
    program, ops, chunk = _program(circuit), _TensorOps(), min(128, capacity)
    for start in range(0, len(rows), chunk):
        selected = clean[start : start + chunk]
        columns = {name: np.array([row[name] for row in selected]) for name in circuit.parameters}
        values = {}
        for name, angle in program.expressions:
            value = np.asarray(angle.evaluate(columns))
            if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
                raise ValidationError("Bound circuit angles must be finite real values.")
            values[name] = value
        states = statevector(
            program.template,
            values,
            backend="numpy",
            batch_shape=(len(selected),),
            max_qubits=options.max_qubits,
            max_memory_mb=options.max_memory_mb,
        )
        validate_states(states, (circuit,) * len(selected))
        output[start : start + len(selected)] = _expectations_from_state(
            states, readouts, circuit.num_qubits, ops
        )
    return validate_expectations(output, len(rows), readouts)
