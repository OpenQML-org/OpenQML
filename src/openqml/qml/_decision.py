"""Certified inference for affine heads on a quantum layer's declared observables."""

from dataclasses import dataclass

import numpy as np

from ..compute.kernel import get_compute_kernel
from ..errors import ResourceError, ValidationError
from ..ir.observables import Hamiltonian, _linear_observable
from ..ir.parameters import real
from ..runtime.backends import ExecutionOptions
from ..sim._mps import MPSBackend
from ..sim.readout import certified_readout


@dataclass(frozen=True)
class ClassDecision:
    """Margin intervals are candidate score minus each class score, in class order."""

    candidate: int
    label: int | None
    margin_intervals: tuple[tuple[float, float], ...]
    passes: int


def _concrete(value, name):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    try:
        array = np.asarray(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValidationError(f"{name} requires concrete real values.") from exc
    if array.dtype.kind not in "fiu":
        raise ValidationError(f"{name} must be real.")
    return array


def certify_class(
    spec, bindings, *, head=None, bias=None, absolute_error=0.01, backend=None, options=None
):
    """Cancel shared Pauli terms before certifying pairwise logit differences."""
    backend = MPSBackend() if backend is None else backend
    options = ExecutionOptions() if options is None else options
    if not isinstance(backend, MPSBackend) or not isinstance(options, ExecutionOptions):
        raise ValidationError("Specify an MPSBackend and ExecutionOptions.")
    tolerance = real(absolute_error)
    if tolerance < 0:
        raise ValidationError("absolute_error must be nonnegative.")
    obs = spec.observables
    if head is not None:
        head = _concrete(head, "head")
        if head.ndim != 2 or head.shape[1] != len(obs):
            raise ValidationError("head must have shape (classes, readouts).")
    classes = len(obs) if head is None else head.shape[0]
    if classes < 2:
        raise ValidationError("Classification requires at least two class scores.")
    terms = [o.terms if isinstance(o, Hamiltonian) else (o,) for o in obs]
    size = sum(map(len, terms))
    width = spec.circuit.num_qubits
    if (
        classes * size > backend.max_work
        or classes * size * (width + 128)
        > get_compute_kernel()._prepare_budget(options.max_memory_mb).max_bytes
    ):
        raise ResourceError("Class margin descriptions exceed the configured budget.")
    if head is not None and not np.isfinite(head).all():
        raise ValidationError("head must be finite.")
    bias = np.zeros(classes) if bias is None else _concrete(bias, "bias")
    if bias.shape != (classes,) or not np.isfinite(bias).all():
        raise ValidationError("Supply one finite bias per class.")
    source = spec.circuit.bind(bindings)

    def difference(i, j):
        features = (
            ((i, 1.0), (j, -1.0))
            if head is None
            else ((k, float(head[i, k]) - float(head[j, k])) for k in range(len(obs)))
        )
        return _linear_observable(obs, features, float(bias[i]) - float(bias[j]))

    def evaluate(pairs):
        return certified_readout(
            source,
            [difference(i, j) for i, j in pairs],
            backend=backend,
            options=options,
            absolute_error=tolerance,
            strict=False,
        )

    anchor = evaluate((i, 0) for i in range(1, classes))
    values, bounds = (0.0, *anchor.values), (0.0, *anchor.absolute_error_bounds)
    candidate = max(range(classes), key=values.__getitem__)
    rivals = tuple(i for i in range(classes) if i != candidate)
    intervals = [
        (
            values[candidate] - v - bounds[candidate] - e,
            values[candidate] - v + bounds[candidate] + e,
        )
        for v, e in zip(values, bounds, strict=True)
    ]
    intervals[candidate] = (0.0, 0.0)
    passes = 1
    if candidate and any(intervals[i][0] <= 0 for i in rivals if i):
        # The first pass correlates every margin with the anchor. A direct
        # comparison also cancels coefficients shared only with this candidate.
        uncertain = tuple(i for i in rivals if i and intervals[i][0] <= 0)
        refined = evaluate((candidate, i) for i in uncertain)
        for i, v, e in zip(uncertain, refined.values, refined.absolute_error_bounds, strict=True):
            intervals[i] = (v - e, v + e)
        passes += 1
    label = candidate if all(intervals[i][0] > 0 for i in rivals) else None
    return ClassDecision(candidate, label, tuple(intervals), passes)
