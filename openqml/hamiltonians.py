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
    """Accept ``[(coeff, {wire: 'X'}), ...]`` or ``[(coeff, 'X0 Z1'), ...]``."""
    parsed: List[Term] = []
    for coefficient, word in terms:
        if isinstance(word, str):
            mapping: PauliWord = {}
            for token in word.split():
                if token.upper() in ("I", ""):
                    continue
                mapping[int(token[1:])] = token[0].upper()
        else:
            mapping = {int(k): str(v).upper() for k, v in dict(word).items() if str(v).upper() != "I"}
        parsed.append((float(coefficient), mapping))
    return parsed


def n_qubits_of(terms: Sequence[Term]) -> int:
    wires = [wire for _, word in terms for wire in word]
    return (max(wires) + 1) if wires else 1


def to_matrix(terms, n_qubits: int | None = None) -> np.ndarray:
    """Dense matrix of a Pauli sum (only sensible for a handful of qubits)."""
    terms = parse_terms(terms)
    n_qubits = n_qubits or n_qubits_of(terms)
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
