"""Classical baselines.

Every benchmark in this package ships with a classical arm on purpose. A
quantum result with no baseline next to it is not a result.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import QuantumModel

__all__ = ["MajorityClassifier", "LinearRegressor"]


class MajorityClassifier(QuantumModel):
    """Always predicts the most frequent training label -- the floor."""

    _estimator_type = "classifier"

    def __init__(self):
        pass

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_, counts = np.unique(y, return_counts=True)
        self._majority = self.classes_[int(np.argmax(counts))]
        self._prior = counts / counts.sum()
        return self

    def predict(self, X):
        return np.full(len(np.asarray(X)), self._majority)

    def predict_proba(self, X):
        return np.tile(self._prior, (len(np.asarray(X)), 1))


class LinearRegressor(QuantumModel):
    """Ridge regression on the raw features, optionally with a Fourier basis.

    With ``n_frequencies=0`` this is the naive baseline that fails on periodic
    targets; raise it and the classical model catches up -- which is the useful
    thing to report.
    """

    _estimator_type = "regressor"

    def __init__(self, alpha: float = 1e-6, n_frequencies: int = 0):
        self.alpha = alpha
        self.n_frequencies = n_frequencies

    def _design(self, X):
        X = np.asarray(X, dtype=float)
        columns = [np.ones((len(X), 1)), X]
        for k in range(1, int(self.n_frequencies) + 1):
            columns.append(np.cos(k * np.pi * X))
            columns.append(np.sin(k * np.pi * X))
        return np.hstack(columns)

    def fit(self, X, y):
        design = self._design(X)
        gram = design.T @ design + self.alpha * np.eye(design.shape[1])
        self.coef_ = np.linalg.solve(gram, design.T @ np.asarray(y, dtype=float))
        return self

    def predict(self, X):
        return self._design(X) @ self.coef_
