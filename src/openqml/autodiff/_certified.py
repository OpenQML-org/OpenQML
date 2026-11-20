"""Per-parameter truncation certificates for occurrence-wise shift products."""

from dataclasses import dataclass

import numpy as np

from ..compute.kernel import get_compute_kernel
from ..errors import ResourceError, ValidationError
from ..ir.observables import Hamiltonian, _linear_observable
from ..ir.parameters import real
from ..runtime.backends import ExecutionOptions
from ..sim._mps import MPSBackend
from ..sim.readout import plan_readout
from ._common import _names, _real_array
from .gradients import ShiftPlan, _coefficient, _recipe


@dataclass(frozen=True)
class CertifiedGradient:
    parameters: tuple[str, ...]
    values: tuple[float, ...]
    absolute_error_bounds: tuple[float, ...]
    evaluations: int

    @property
    def intervals(self):
        return tuple(
            (v - e, v + e) for v, e in zip(self.values, self.absolute_error_bounds, strict=True)
        )


def certified_vjp(
    plan,
    bindings,
    cotangents,
    *,
    absolute_error,
    parameters=None,
    backend=None,
    options=None,
    strict=True,
):
    engine = MPSBackend() if backend is None else backend
    options = ExecutionOptions() if options is None else options
    if not isinstance(engine, MPSBackend) or not isinstance(options, ExecutionOptions):
        raise ValidationError("Specify an MPSBackend and ExecutionOptions.")
    tolerance = real(absolute_error)
    if tolerance < 0 or type(strict) is not bool:
        raise ValidationError("Supply nonnegative absolute_error and boolean strict.")
    source = plan.source_circuit
    if options.max_qubits is not None and source.num_qubits > options.max_qubits:
        raise ResourceError("Certified source exceeds max_qubits.")
    if len(source.operations) > engine.max_operations:
        raise ResourceError("Certified gradient exceeds max_operations.")
    names = _names(parameters, source.parameters)
    local = plan._binding_values(bindings)
    terms = sum(len(o.terms) if isinstance(o, Hamiltonian) else 1 for o in plan.observables)
    metadata = terms * (128 + plan.circuit.num_qubits) + 128 * len(names)
    if metadata > get_compute_kernel()._prepare_budget(options.max_memory_mb).max_bytes:
        raise ResourceError("Gradient observable descriptions exceed max_memory_mb.")
    vector = _real_array(cotangents, (len(plan.observables),), "Cotangents")
    objective = _linear_observable(plan.observables, enumerate(vector))
    cone = plan_readout(plan.circuit, (objective,), max_operations=engine.max_operations)
    active = tuple(name for name in names if name in cone.circuit.parameters)
    values = np.zeros(len(names))
    errors = np.zeros_like(values)
    if not active:
        return CertifiedGradient(names, tuple(values), tuple(errors), 0)
    positions = [i for i, name in enumerate(names) if name in active]
    restricted = {name: local[name] for name in cone.circuit.parameters}
    shift = ShiftPlan(cone.circuit, cone.observables, parameters=active, backend="numpy")
    if shift.shift_evaluations * max(1, len(cone.circuit.operations)) > engine.max_work:
        raise ResourceError("Certified shift evaluations exceed max_work.")
    _, occurrences = shift._selection(active)
    masses = np.zeros(len(active))
    for occurrence, derivatives in occurrences:
        # Unit-only readout plans have base-independent shift weights.
        recipe, _ = _recipe(occurrence.gate, 0.0)
        l1 = sum(abs(weight) for _, weight in recipe)
        for col, derivative in derivatives:
            masses[col] += l1 * abs(_coefficient(derivative, restricted))
    if not np.isfinite(masses).all():
        raise ValidationError("Shift coefficient mass exceeds finite precision.")
    columns = {name: np.asarray([value]) for name, value in restricted.items()}
    jobs = shift._tasks(columns, 1, occurrences, "jacobian", None)
    evaluations = 0
    for _, coefficients, weight, circuit in jobs:
        limit = min(tolerance / float(masses[i]) for i, _ in coefficients)
        r = engine.readout(
            circuit,
            cone.observables,
            options=options,
            absolute_error=min(limit, float(np.finfo(float).max)),
            strict=False,
        )
        for i, d in coefficients:
            factor = weight * d
            values[positions[i]] += factor * r.values[0]
            errors[positions[i]] += abs(factor) * r.absolute_error_bounds[0]
        evaluations += 1
    if not (np.isfinite(values).all() and np.isfinite(errors).all()):
        raise ValidationError("Certified gradient exceeds finite precision.")
    if strict and any(e > tolerance for e in errors):
        raise ResourceError("Schmidt truncation exceeds the requested gradient tolerance.")
    return CertifiedGradient(
        names, tuple(map(float, values)), tuple(map(float, errors)), evaluations
    )
