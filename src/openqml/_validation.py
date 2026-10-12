"""Small, consistent validation helpers; no optional dependency imports."""

from numbers import Integral, Real

import numpy as np


def integer(value, name, minimum=0, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum or (maximum is not None and value > maximum):
        limit = f" through {maximum}" if maximum is not None else " or greater"
        raise ValueError(f"{name} must be {minimum}{limit}")
    return value


def real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return value


def probability(value, name="probability"):
    value = real(value, name)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


def wires(values, n_qubits, *, allow_empty=False):
    if isinstance(values, Integral):
        values = (values,)
    try:
        result = tuple(integer(q, "wire", 0, n_qubits - 1) for q in values)
    except TypeError as error:
        raise ValueError("wires must be an integer or an iterable of integers") from error
    if (not result and not allow_empty) or len(set(result)) != len(result):
        raise ValueError(
            "wires must be nonempty and distinct" if not allow_empty else "wires must be distinct"
        )
    return result


def pauli_string(value, n_qubits=None):
    if not isinstance(value, str) or not value or any(p not in "IXYZ" for p in value):
        raise ValueError("Pauli operators must be nonempty strings containing only I, X, Y, Z")
    if n_qubits is not None and len(value) != n_qubits:
        raise ValueError(f"Pauli string must have {n_qubits} characters (q0 is rightmost)")
    return value


def vector(value, name="parameters"):
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite one-dimensional array")
    return result.copy()
