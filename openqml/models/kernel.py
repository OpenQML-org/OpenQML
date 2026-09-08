"""Kernel methods: one solver, two kernels, so comparisons are honest.

Both the quantum and the classical estimator below use the identical kernel
ridge solve. The only thing that differs is the Gram matrix, which is exactly
the thing being tested.
"""

from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from typing import Optional

import numpy as np

from ..backends import DEFAULT_BACKEND, batched_states, get_backend
from ..circuits.circuit import Circuit
from ..circuits.templates import get_feature_map
from .base import QuantumModel, pad_to_power_of_two

__all__ = [
    "quantum_kernel", "encode_states", "clear_state_cache",
    "QuantumKernelClassifier", "QuantumKernelRegressor",
    "ClassicalKernelClassifier", "ClassicalKernelRegressor",
]

#: Cross-validation encodes the same rows once per fold. Keeping the last few
#: encodings turns that into one pass, which is where most of the time went.
_STATE_CACHE: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_STATE_CACHE_SIZE = 8


def clear_state_cache() -> None:
    """Drop the encoded-state cache (only worth calling to free memory)."""
    _STATE_CACHE.clear()


def encode_states(X, feature_map="zz", n_qubits: Optional[int] = None,
                  backend: str = DEFAULT_BACKEND, seed=None, use_cache: bool = True) -> np.ndarray:
    """Map every row to the statevector its feature map prepares.

    The whole matrix goes through the simulator in one batched pass; identical
    calls are served from a small cache, which is what makes k-fold
    cross-validation cost one encoding instead of k.
    """
    X = np.asarray(X)
    if feature_map == "amplitude" or np.iscomplexobj(X):
        states = pad_to_power_of_two(np.asarray(X, dtype=complex))
        norms = np.linalg.norm(states, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return states / norms
    n_qubits = int(n_qubits or X.shape[1])
    key = None
    if use_cache and isinstance(feature_map, str):
        digest = hashlib.blake2b(np.ascontiguousarray(X).tobytes(), digest_size=16).hexdigest()
        key = (digest, X.shape, feature_map, n_qubits, backend)
        cached = _STATE_CACHE.get(key)
        if cached is not None:
            _STATE_CACHE.move_to_end(key)
            return cached
    circuit = get_feature_map(feature_map, n_qubits, X.shape[1])
    states = batched_states(circuit, X, backend=backend, seed=seed)
    if key is not None:
        _STATE_CACHE[key] = states
        while len(_STATE_CACHE) > _STATE_CACHE_SIZE:
            _STATE_CACHE.popitem(last=False)
    return states


def quantum_kernel(X, Z=None, feature_map="zz", n_qubits: Optional[int] = None,
                   backend: str = DEFAULT_BACKEND, seed=None) -> np.ndarray:
    """Fidelity kernel |<phi(x)|phi(z)>|^2 as a dense Gram matrix."""
    states_x = encode_states(X, feature_map, n_qubits, backend, seed)
    states_z = states_x if Z is None else encode_states(Z, feature_map, n_qubits, backend, seed)
    overlaps = states_x @ states_z.conj().T
    return np.abs(overlaps) ** 2


def _classical_kernel(X, Z, kernel: str, gamma: Optional[float], degree: int, coef0: float):
    X = np.asarray(X, dtype=float)
    Z = np.asarray(Z, dtype=float)
    if kernel == "linear":
        return X @ Z.T
    if kernel == "poly":
        gamma = gamma or 1.0 / X.shape[1]
        return (gamma * (X @ Z.T) + coef0) ** degree
    if kernel == "rbf":
        gamma = gamma or 1.0 / X.shape[1]
        distances = (np.sum(X ** 2, axis=1)[:, None] + np.sum(Z ** 2, axis=1)[None, :]
                     - 2 * X @ Z.T)
        return np.exp(-gamma * np.maximum(distances, 0.0))
    raise ValueError(f"unknown classical kernel {kernel!r}")


class _KernelRidgeBase(QuantumModel):
    """Shared solve: alpha = (K + lambda I)^-1 Y."""

    def _solve(self, gram: np.ndarray, targets: np.ndarray) -> np.ndarray:
        n = gram.shape[0]
        return np.linalg.solve(gram + self.alpha * np.eye(n), targets)


class QuantumKernelClassifier(_KernelRidgeBase):
    """Fidelity-kernel classifier trained by kernel ridge regression.

    No variational parameters and no optimiser, so a result here says something
    about the *feature map* rather than about the training loop.
    """

    _estimator_type = "classifier"

    def __init__(self, feature_map: str = "zz", n_qubits: Optional[int] = None,
                 alpha: float = 1e-3, rescale="auto", backend: str = DEFAULT_BACKEND,
                 seed: Optional[int] = None):
        self.feature_map = feature_map
        self.n_qubits = n_qubits
        self.alpha = alpha
        self.rescale = rescale
        self.backend = backend
        self.seed = seed

    def fit(self, X, y):
        X = self._fit_features(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self._states = encode_states(X, self.feature_map, self.n_qubits, self.backend, self.seed)
        self.n_qubits_ = int(math.log2(self._states.shape[1]))
        gram = np.abs(self._states @ self._states.conj().T) ** 2
        targets = -np.ones((len(y), len(self.classes_)))
        for column, label in enumerate(self.classes_):
            targets[y == label, column] = 1.0
        self.dual_coef_ = self._solve(gram, targets)
        self.gram_ = gram
        return self

    def decision_function(self, X) -> np.ndarray:
        states = encode_states(self._transform_features(X), self.feature_map, self.n_qubits_,
                               self.backend, self.seed)
        gram = np.abs(states @ self._states.conj().T) ** 2
        return gram @ self.dual_coef_

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

    def predict_proba(self, X) -> np.ndarray:
        scores = self.decision_function(X)
        scores = scores - scores.max(axis=1, keepdims=True)
        exponentiated = np.exp(2.0 * scores)
        return exponentiated / exponentiated.sum(axis=1, keepdims=True)


class QuantumKernelRegressor(_KernelRidgeBase):
    """Kernel ridge regression with the fidelity kernel."""

    _estimator_type = "regressor"

    def __init__(self, feature_map: str = "zz", n_qubits: Optional[int] = None,
                 alpha: float = 1e-3, rescale="auto", backend: str = DEFAULT_BACKEND,
                 seed: Optional[int] = None):
        self.feature_map = feature_map
        self.n_qubits = n_qubits
        self.alpha = alpha
        self.rescale = rescale
        self.backend = backend
        self.seed = seed

    def fit(self, X, y):
        self._states = encode_states(self._fit_features(X), self.feature_map, self.n_qubits,
                                     self.backend, self.seed)
        self.n_qubits_ = int(math.log2(self._states.shape[1]))
        gram = np.abs(self._states @ self._states.conj().T) ** 2
        self._y_mean = float(np.mean(y))
        self.dual_coef_ = self._solve(gram, np.asarray(y, dtype=float) - self._y_mean)
        return self

    def predict(self, X):
        states = encode_states(self._transform_features(X), self.feature_map, self.n_qubits_,
                               self.backend, self.seed)
        gram = np.abs(states @ self._states.conj().T) ** 2
        return gram @ self.dual_coef_ + self._y_mean


class ClassicalKernelClassifier(_KernelRidgeBase):
    """The same solver with an RBF/linear/polynomial kernel: the control arm."""

    _estimator_type = "classifier"

    def __init__(self, kernel: str = "rbf", gamma: Optional[float] = None, degree: int = 3,
                 coef0: float = 1.0, alpha: float = 1e-3, rescale="auto"):
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.alpha = alpha
        self.rescale = rescale

    def _prepare(self, X):
        X = np.asarray(X)
        return np.column_stack([X.real, X.imag]) if np.iscomplexobj(X) else X.astype(float)

    def fit(self, X, y):
        self._X = self._prepare(self._fit_features(X))
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        gram = _classical_kernel(self._X, self._X, self.kernel, self.gamma, self.degree, self.coef0)
        targets = -np.ones((len(y), len(self.classes_)))
        for column, label in enumerate(self.classes_):
            targets[y == label, column] = 1.0
        self.dual_coef_ = self._solve(gram, targets)
        return self

    def decision_function(self, X):
        gram = _classical_kernel(self._prepare(self._transform_features(X)), self._X, self.kernel,
                                 self.gamma, self.degree, self.coef0)
        return gram @ self.dual_coef_

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

    def predict_proba(self, X):
        scores = self.decision_function(X)
        scores = scores - scores.max(axis=1, keepdims=True)
        exponentiated = np.exp(2.0 * scores)
        return exponentiated / exponentiated.sum(axis=1, keepdims=True)


class ClassicalKernelRegressor(_KernelRidgeBase):
    """Classical kernel ridge regression."""

    _estimator_type = "regressor"

    def __init__(self, kernel: str = "rbf", gamma: Optional[float] = None, degree: int = 3,
                 coef0: float = 1.0, alpha: float = 1e-3, rescale="auto"):
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.alpha = alpha
        self.rescale = rescale

    def fit(self, X, y):
        self._X = np.asarray(self._fit_features(X), dtype=float)
        gram = _classical_kernel(self._X, self._X, self.kernel, self.gamma, self.degree, self.coef0)
        self._y_mean = float(np.mean(y))
        self.dual_coef_ = self._solve(gram, np.asarray(y, dtype=float) - self._y_mean)
        return self

    def predict(self, X):
        gram = _classical_kernel(np.asarray(self._transform_features(X), dtype=float), self._X,
                                 self.kernel, self.gamma, self.degree, self.coef0)
        return gram @ self.dual_coef_ + self._y_mean
