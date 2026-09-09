"""Pauli-sum Hamiltonians: parsing, matrices and exact reference energies."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .backends.statevector import PAULI

__all__ = ["parse_terms", "to_matrix", "exact_ground_state", "n_qubits_of", "terms_to_json",
           "terms_from_json"]

PauliWord = Dict[int, str]
Term = Tuple[float, PauliWord]


def parse_terms(terms) -> List[Term]:
    """Accept ``[(coeff, {wire: 'X'}), ...]`` or ``[(coeff, 'X0 Z1'), ...]``.

    Both spellings produce the same word: identities are dropped, and a wire
    named twice is an error rather than a silent overwrite. ``'X0 Y0'`` is
    ``i*Z0``, not ``Y0``, and quietly returning the second factor would hand
    back a different Hermitian operator with a different spectrum.
    """
    parsed: List[Term] = []
    for coefficient, word in terms:
        mapping: PauliWord = {}
        if isinstance(word, str):
            for token in word.split():
                letter, wire = token[0].upper(), token[1:]
                if letter == "I" or not token:
                    continue
                if letter not in PAULI:
                    raise ValueError(f"unknown Pauli letter {letter!r} in {word!r}")
                wire = int(wire)
                if wire in mapping:
                    raise ValueError(
                        f"wire {wire} appears twice in {word!r}; write the product out "
                        f"as a single Pauli letter instead"
                    )
                mapping[wire] = letter
        else:
            for k, v in dict(word).items():
                letter = str(v).upper()
                if letter == "I":
                    continue
                if letter not in PAULI:
                    raise ValueError(f"unknown Pauli letter {letter!r}")
                mapping[int(k)] = letter
        parsed.append((float(coefficient), mapping))
    return parsed


def n_qubits_of(terms: Sequence[Term]) -> int:
    wires = [wire for _, word in terms for wire in word]
    return (max(wires) + 1) if wires else 1


def to_matrix(terms, n_qubits: int | None = None) -> np.ndarray:
    """Dense matrix of a Pauli sum (only sensible for a handful of qubits)."""
    terms = parse_terms(terms)
    needed = n_qubits_of(terms)
    n_qubits = n_qubits or needed
    if needed > n_qubits:
        # kron only walks range(n_qubits), so a factor on a higher wire would be
        # dropped -- silently replacing the operator with its restriction, which
        # has different eigenvalues.
        raise ValueError(
            f"a term acts on wire {needed - 1}, which does not fit in {n_qubits} qubits"
        )
    if n_qubits > 14:
        raise ValueError("dense construction is capped at 14 qubits")
    dimension = 2 ** n_qubits
    matrix = np.zeros((dimension, dimension), dtype=complex)
    for coefficient, word in terms:
        operator = np.array([[1.0 + 0j]])
        for wire in range(n_qubits):
            operator = np.kron(operator, PAULI[word.get(wire, "I")])
        matrix += coefficient * operator
    return matrix


def exact_ground_state(terms, n_qubits: int | None = None):
    """Return ``(energy, statevector)`` by exact diagonalisation."""
    matrix = to_matrix(terms, n_qubits)
    values, vectors = np.linalg.eigh(matrix)
    return float(values[0]), vectors[:, 0]


def terms_to_json(terms) -> List[dict]:
    return [{"coefficient": float(c), "word": {str(k): v for k, v in word.items()}}
            for c, word in parse_terms(terms)]


def terms_from_json(payload) -> List[Term]:
    return [(float(item["coefficient"]), {int(k): v for k, v in item["word"].items()})
            for item in payload]
