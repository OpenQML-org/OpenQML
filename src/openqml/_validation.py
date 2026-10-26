"""Shared numeric provider contracts, independent of optional SDKs."""

from collections.abc import Set
from itertools import islice

import numpy as np

from .errors import BackendError, OpenQMLError, ValidationError
from .ir import integer
from .observables import Hamiltonian


def validate_classical_bits(circuit, bits):
    if bits is None:
        return None
    if isinstance(bits, Set):
        raise ValidationError("classical_bits must have a defined order.")
    try:
        bits = tuple(bits)
    except TypeError as exc:
        raise ValidationError("classical_bits must be an ordered iterable of indices.") from exc
    bits = tuple(integer(bit, "classical bit") for bit in bits)
    if len(set(bits)) != len(bits):
        raise ValidationError("classical_bits cannot contain duplicate indices.")
    if circuit is not None and (
        not circuit.has_measurements or max(bits, default=-1) >= circuit.num_clbits
    ):
        raise ValidationError("classical_bits require measurements and declared storage indices.")
    return bits


def project_counts(counts, bits):
    if bits is None:
        return counts
    output = {}
    for key, value in counts.items():
        target = "".join(key[bit] for bit in bits)
        output[target] = output.get(target, 0) + value
    return output


def invoke(device, method, *args, **kwargs):
    implementation = getattr(device, method, None)
    if not callable(implementation):
        raise BackendError(f"Backend {device.name!r} does not implement {method}().")
    try:
        return implementation(*args, **kwargs)
    except OpenQMLError:
        raise
    except Exception as exc:
        raise BackendError(f"Backend {device.name!r} failed in {method}(): {exc}") from exc


def validate_state(value, circuit):
    try:
        state = np.asarray(value, dtype=np.complex128)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BackendError("Backend returned nonnumeric state data.") from exc
    if state.shape != (2**circuit.num_qubits,) or not np.isfinite(state).all():
        raise BackendError("Backend returned an invalid state shape or values.")
    with np.errstate(over="ignore", invalid="ignore"):
        norm = np.vdot(state, state).real
    if not np.isfinite(norm) or not np.isclose(norm, 1, atol=1e-6, rtol=0):
        raise BackendError("Backend returned an unnormalized state.")
    return state


def validate_states(values, circuits):
    try:
        rows = list(islice(iter(values), len(circuits) + 1))
    except (TypeError, ValueError, RuntimeError) as exc:
        raise BackendError("Backend returned an invalid state batch.") from exc
    if len(rows) != len(circuits):
        raise BackendError("Backend returned an invalid state batch length.")
    return [validate_state(value, circuit) for value, circuit in zip(rows, circuits, strict=True)]


def validate_expectations(value, batch_size, observables):
    try:
        output = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BackendError("Backend returned invalid batched expectation data.") from exc
    if (
        output.shape != (batch_size, len(observables))
        or output.dtype.kind not in "fiu"
        or not np.isfinite(output).all()
    ):
        raise BackendError("Backend returned invalid batched expectation data.")
    bounds = np.asarray(
        [
            sum(abs(t.coefficient) for t in ob.terms)
            if isinstance(ob, Hamiltonian)
            else abs(ob.coefficient)
            for ob in observables
        ]
    )
    if np.any(np.abs(output) > bounds + 1e-6):
        raise BackendError("Backend expectation exceeds the observable's spectral bound.")
    return output.astype(float, copy=False)


def validate_shots(shots):
    integer(shots, "shots", 1)
    if shots > np.iinfo(np.int64).max:
        raise ValidationError("shots must fit in a signed 64-bit integer.")
    return int(shots)


def validate_density(value, circuit):
    try:
        rho = np.asarray(value, dtype=complex)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BackendError("Backend returned invalid density data.") from exc
    size = 2**circuit.num_qubits
    if (
        rho.shape != (size, size)
        or not np.isfinite(rho).all()
        or not np.allclose(rho, rho.conj().T, atol=1e-6, rtol=0)
        or not np.isclose(np.trace(rho), 1, atol=1e-6, rtol=0)
    ):
        raise BackendError("Backend returned an invalid Hermitian, normalized density matrix.")
    if np.linalg.eigvalsh(rho).min() < -1e-6:
        raise BackendError("Backend returned a density matrix that is not positive semidefinite.")
    return rho
