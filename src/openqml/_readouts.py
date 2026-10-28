"""Shared exact Pauli readouts without retained states or full-state constants."""

from .errors import ValidationError
from .observables import Hamiltonian, PauliString


def _diagonal_value(diagonal, mask, num_qubits):
    """Marginalize unused wires before contracting the requested Z parity."""
    shape = diagonal.shape[:-1]
    axes = tuple(
        len(shape) + q for q in range(num_qubits) if not mask & (1 << (num_qubits - q - 1))
    )
    value = diagonal.reshape(shape + (2,) * num_qubits)
    if axes:
        value = value.sum(axis=axes)
    for _ in range(mask.bit_count()):
        value = value[..., 0] - value[..., 1]
    return value


def _walsh(diagonal, num_qubits, ops):
    """Unnormalized Walsh coefficients in computational-basis bit order."""
    shape, value = diagonal.shape, diagonal
    for bit in range(num_qubits):
        stride = 1 << bit
        block = value.reshape(shape[:-1] + (shape[-1] // (2 * stride), 2, stride))
        left, right = block[..., 0, :], block[..., 1, :]
        pair = (left + right, left - right)
        value = (
            ops.xp.stack(pair, dim=-2) if ops.backend == "torch" else ops.xp.stack(pair, axis=-2)
        ).reshape(shape)
    return value


def _evaluate_readouts(observables, num_qubits, ops, anchor, diagonal, pauli):
    """Share unit-coefficient terms, retaining sparse output routes and order.

    A dense Walsh transform is selected only for many distinct diagonal terms.
    Sparse queries first marginalize unused wires. Validation includes zero
    terms; zero coefficients skip execution without changing their gradient.
    All runtime tensors belong to this invocation and remain in native autodiff.
    """
    routes, words = [], {}
    for observable in observables:
        terms = observable.terms if isinstance(observable, Hamiltonian) else (observable,)
        route = []
        for term in terms:
            if not isinstance(term, PauliString) or len(term.paulis) != num_qubits:
                raise ValidationError("Observable width must equal the circuit's qubit count.")
            if term.coefficient:
                words.setdefault(term.paulis, None)
                route.append((term.paulis, term.coefficient))
        routes.append(route)
    masks = {
        word: sum(1 << (num_qubits - q - 1) for q, p in enumerate(word) if p == "Z")
        for word in words
        if not set(word) - {"I", "Z"}
    }
    if masks:
        probabilities = diagonal()
        if len(masks) >= max(8, 2 * num_qubits):
            spectrum = _walsh(probabilities, num_qubits, ops)
            for word, mask in masks.items():
                words[word] = spectrum[..., mask]
        else:
            for word, mask in masks.items():
                words[word] = _diagonal_value(probabilities, mask, num_qubits)
    for word in words:
        if word not in masks:
            words[word] = pauli(word)
    zero, values = ops.real(anchor) * 0, []
    for route in routes:
        total = zero
        for word, coefficient in route:
            total = total + coefficient * words[word]
        values.append(total)
    return (
        ops.xp.stack(tuple(values), dim=-1)
        if ops.backend == "torch"
        else ops.xp.stack(tuple(values), axis=-1)
    )
