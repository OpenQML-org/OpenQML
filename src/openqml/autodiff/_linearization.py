"""Explicit, per-evaluation linearizations for submission-aware quantum training."""

from dataclasses import dataclass
from itertools import chain, islice

import numpy as np

from ..compute.kernel import compute_operation, get_compute_kernel
from ..errors import ValidationError
from ..ir.core import integer
from ._common import _real_array


@dataclass(frozen=True)
class ShiftLinearization:
    """Owned read-only primal/Jacobian arrays; products submit no further circuits.

    A linearization is valid at its original bindings only. Submission batches
    count backend calls, not a provider's internal jobs or billed shot usage.
    """

    values: np.ndarray
    jacobian: np.ndarray
    parameters: tuple[str, ...]
    circuit_evaluations: int
    submission_batches: int

    @compute_operation
    def vjp(self, cotangents):
        value = _real_array(cotangents, self.values.shape, "Cotangents")
        return _finite_product("bm,bmp->bp", value, self.jacobian)

    @compute_operation
    def jvp(self, tangents):
        shape = (self.values.shape[0], len(self.parameters))
        value = _real_array(tangents, shape, "Tangents")
        return _finite_product("bmp,bp->bm", self.jacobian, value)


def _finite_product(equation, *values):
    result = get_compute_kernel().einsum(equation, *values)
    if not np.isfinite(result).all():
        raise ValidationError("Shift products exceed finite real precision.")
    return result


@compute_operation
def linearize(plan, bindings, *, parameters=None, chunk_size=64, batch_size=None, **options):
    """Pack primal rows and occurrence-wise shift stencils before reading results."""
    from ..sim.batch import expectations_batch
    from .gradients import _coefficient, _recipe

    size = integer(chunk_size, "chunk_size", 2)
    names, occurrences = plan._selection(parameters)
    options = plan._options(options)
    backend = plan._backend()
    columns, count = plan._bindings(bindings, batch_size)
    maximum = count * (1 + plan.shift_evaluations)
    # Include the explicit Jacobian, numeric results and bounded cloned IR rows.
    width = len(plan.observables)
    objects = min(size, maximum) * (1 + sum(1 for _ in plan.lowered_circuit.walk_operations()))
    get_compute_kernel().admit(
        count * width * (1 + len(names)) + min(size, maximum) * width + 32 * objects,
        options["max_memory_mb"],
    )
    # A valid primal can still have an undefined derivative (e.g. damping=1 or
    # sqrt at zero). Check every selected occurrence before charging any job.
    for members in plan._groups(columns, count):
        row = {name: columns[name][members[0]] for name in plan._lowered_symbols}
        bound = None
        for occurrence, partials in occurrences:
            coefficients = tuple(_coefficient(expr, row) for _, expr in partials)
            if not any(coefficients):
                continue
            if bound is None:
                bound = plan.lowered_circuit.bind(row)
            operation = bound.operation_at(occurrence.path)
            _recipe(occurrence.gate, operation.params[occurrence.parameter_index].constant)
    values = np.empty((count, width))
    jacobian = np.zeros((count, width, len(names)))

    def primals():
        for members in plan._groups(columns, count):
            row = {name: columns[name][members[0]] for name in plan._lowered_symbols}
            yield np.asarray(members), None, 1, plan.lowered_circuit.bind(row)

    tasks = iter(chain(primals(), plan._tasks(columns, count, occurrences, "jacobian", None)))
    evaluations = submissions = 0
    while group := list(islice(tasks, size)):
        output = expectations_batch(
            [job for _, _, _, job in group], plan.observables, backend=backend, **options
        )
        submissions += 1
        evaluations += len(group)
        for value, (rows, coefficients, weight, _) in zip(output, group, strict=True):
            if coefficients is None:
                values[rows] = value
            else:
                for position, coefficient in coefficients:
                    jacobian[rows, :, position] += coefficient * weight * value
    if not np.isfinite(values).all() or not np.isfinite(jacobian).all():
        raise ValidationError("Shift linearization exceeds finite real precision.")
    values.setflags(write=False)
    jacobian.setflags(write=False)
    return ShiftLinearization(values, jacobian, names, evaluations, submissions)
