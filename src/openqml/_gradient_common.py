"""Shared validation for prepared numerical gradient engines."""

from collections.abc import Mapping, Set

import numpy as np

from .errors import ValidationError
from .ir import integer
from .observables import Hamiltonian, PauliString
from .parameters import parameter_name, real


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


def _observables(values, width, *, label="Adjoint"):
    if isinstance(values, (str, bytes, Mapping, Set)):
        raise ValidationError(f"{label} observables require an ordered iterable.")
    try:
        obs = tuple(values)
    except TypeError as exc:
        raise ValidationError(f"{label} observables require an ordered iterable.") from exc
    if not obs:
        raise ValidationError(f"{label} gradients require observables.")
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


def _binding_columns(bindings, symbols, batch_size, *, label):
    """Snapshot ordered rows or named scalar/vector columns into owned real arrays."""
    if batch_size is not None:
        batch_size = integer(batch_size, "batch_size")
    expected = set(symbols)
    if isinstance(bindings, Mapping):
        if set(bindings) != expected:
            raise ValidationError(f"{label} bindings must match circuit parameter names exactly.")
        try:
            arrays = {name: np.asarray(value) for name, value in bindings.items()}
            if any(value.ndim > 1 or value.dtype.kind not in "iuf" for value in arrays.values()):
                raise ValueError("Binding columns require real scalars or vectors.")
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
                f"{label} binding columns require broadcastable finite real vectors."
            ) from exc
        return columns, count
    try:
        rows = list(bindings)
    except TypeError as exc:
        raise ValidationError(f"{label} bindings require an iterable of name mappings.") from exc
    count = len(rows)
    if batch_size is not None and batch_size != count:
        raise ValidationError("batch_size differs from the binding row count.")
    columns = {name: np.empty(count) for name in symbols}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != expected:
            raise ValidationError(f"{label} bindings must match circuit parameter names exactly.")
        for name in symbols:
            columns[name][index] = real(row[name])
    return columns, count
