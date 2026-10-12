"""Hermitian sums of Pauli strings with explicit little-endian wire semantics."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ._validation import integer, pauli_string, real
from .exceptions import ResourceLimitError
from .simulation import X, Y, Z


@dataclass(frozen=True, init=False)
class Hamiltonian:
    terms: tuple[tuple[str, float], ...]
    n_qubits: int

    def __init__(self, terms):
        """Accept {'ZZ': -1, 'XI': -0.5} or [(pauli_string, coefficient), ...]."""
        pairs = list(terms.items() if isinstance(terms, Mapping) else terms)
        if not pairs:
            raise ValueError("Hamiltonian needs at least one Pauli term")
        combined = {}
        width = None
        for pauli, coefficient in pairs:
            pauli = pauli_string(pauli, width)
            width = len(pauli)
            combined[pauli] = combined.get(pauli, 0) + real(coefficient, "coefficient")
        object.__setattr__(self, "terms", tuple(combined.items()))
        object.__setattr__(self, "n_qubits", width)

    def matrix(self, *, max_qubits=10):
        max_qubits = integer(max_qubits, "max_qubits", 1, 10)
        if self.n_qubits > max_qubits:
            raise ResourceLimitError("Exact Hamiltonian matrices are limited to max_qubits <= 10")
        result = np.zeros((1 << self.n_qubits, 1 << self.n_qubits), complex)
        for pauli, coefficient in self.terms:
            term = np.array([[1]], complex)
            for p in pauli:
                term = np.kron(term, {"I": np.eye(2), "X": X, "Y": Y, "Z": Z}[p])
            result += coefficient * term
        return result

    def ground_energy(self, *, max_qubits=10):
        return float(np.linalg.eigvalsh(self.matrix(max_qubits=max_qubits))[0])


def ising_hamiltonian(n_qubits, *, coupling=1.0, field=1.0, periodic=False):
    """H = -coupling sum Z_i Z_j - field sum X_i on an open or periodic chain."""
    n_qubits = integer(n_qubits, "n_qubits", 1, 64)
    coupling, field = real(coupling, "coupling"), real(field, "field")
    terms = []
    edges = [(q, q + 1) for q in range(n_qubits - 1)]
    if periodic and n_qubits > 2:
        edges.append((n_qubits - 1, 0))
    for first, second in edges:
        label = ["I"] * n_qubits
        label[-1 - first] = label[-1 - second] = "Z"
        terms.append(("".join(label), -coupling))
    for q in range(n_qubits):
        label = ["I"] * n_qubits
        label[-1 - q] = "X"
        terms.append(("".join(label), -field))
    return Hamiltonian(terms)
