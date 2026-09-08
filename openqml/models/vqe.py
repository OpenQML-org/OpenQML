"""Variational quantum eigensolver, scored against exact diagonalisation."""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from ..backends import DEFAULT_BACKEND, batched_energies, get_backend
from ..circuits.templates import get_ansatz
from ..hamiltonians import exact_ground_state, n_qubits_of, parse_terms
from .base import QuantumModel, adam
from .variational import shift_sets

__all__ = ["VQE", "ExactDiagonalisation"]


class VQE(QuantumModel):
    """Minimise <psi(theta)|H|psi(theta)> with parameter-shift gradients."""

    _estimator_type = "ground_state"

    def __init__(self, ansatz: str = "real_amplitudes", layers: int = 2,
                 n_qubits: Optional[int] = None, maxiter: int = 80,
                 learning_rate: float = 0.1, shots: Optional[int] = None,
                 backend: str = DEFAULT_BACKEND, seed: int = 0):
        self.ansatz = ansatz
        self.layers = layers
        self.n_qubits = n_qubits
        self.maxiter = maxiter
        self.learning_rate = learning_rate
        self.shots = shots
        self.backend = backend
        self.seed = seed

    def estimate(self, terms) -> float:
        """Run the optimiser on one Hamiltonian and return the best energy."""
        terms = parse_terms(terms)
        n_qubits = int(self.n_qubits or n_qubits_of(terms))
        circuit = get_ansatz(self.ansatz, n_qubits, self.layers)
        device = get_backend(self.backend, n_qubits, seed=self.seed, shots=self.shots)
        rng = np.random.default_rng(self.seed)

        def energies(parameter_sets: np.ndarray) -> np.ndarray:
            return batched_energies(circuit, terms, parameter_sets, device=device)

        def energy(weights: np.ndarray) -> float:
            return float(energies(np.atleast_2d(weights))[0])

        def loss_and_gradient(weights):
            # the value and every parameter-shift evaluation in one batched pass
            sets, scale = shift_sets(weights, unique=True)
            values = energies(sets)
            return float(values[0]), scale * (values[1::2] - values[2::2])

        initial = rng.normal(0, 0.3, max(circuit.n_parameters, 1))
        weights, history = adam(loss_and_gradient, initial, self.maxiter, self.learning_rate)
        self.circuit_ = circuit
        self.weights_ = weights
        self.loss_curve_ = history
        self.energy_ = float(min(history + [energy(weights)]))
        device.run(circuit, weights, None)
        self.state_ = device.state
        return self.energy_

    def fit(self, terms, y=None):
        self.estimate(terms)
        return self

    def predict(self, terms=None):
        return np.array([self.energy_])


class ExactDiagonalisation(QuantumModel):
    """Reference solver. Not a quantum algorithm -- the yardstick for one."""

    _estimator_type = "ground_state"

    def __init__(self, n_qubits: Optional[int] = None):
        self.n_qubits = n_qubits

    def estimate(self, terms) -> float:
        energy, state = exact_ground_state(terms, self.n_qubits)
        self.energy_, self.state_ = energy, state
        return energy

    def fit(self, terms, y=None):
        self.estimate(terms)
        return self

    def predict(self, terms=None):
        return np.array([self.energy_])
