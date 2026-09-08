"""Estimator base class -- the scikit-learn conventions, nothing more."""

from __future__ import annotations

import inspect
import math
from typing import Any, Dict, Optional

import numpy as np

from ..circuits.templates import get_ansatz, get_feature_map
from ..metrics import get_measure

__all__ = ["QuantumModel", "adam", "pad_to_power_of_two"]


def pad_to_power_of_two(X: np.ndarray) -> np.ndarray:
    """Right-pad the columns of ``X`` so a row fills a whole qubit register."""
    width = X.shape[1]
    target = 2 ** max(1, int(math.ceil(math.log2(max(width, 2)))))
    if width == target:
        return X
    padded = np.zeros((X.shape[0], target), dtype=X.dtype)
    padded[:, :width] = X
    return padded


class QuantumModel:
    """``get_params``/``set_params``/``fit``/``predict``, so runs stay portable.

    Anything with this surface can be handed to ``openqml.runs.run_model_on_task``
    -- including a scikit-learn estimator, which is the point: the comparison
    between a quantum model and a classical one has to be trivial to make.
    """

    _estimator_type = "classifier"

    @classmethod
    def _param_names(cls):
        signature = inspect.signature(cls.__init__)
        return sorted(p for p in signature.parameters if p not in ("self", "args", "kwargs"))

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self._param_names()}

    def set_params(self, **params) -> "QuantumModel":
        for key, value in params.items():
            if key not in self._param_names():
                raise ValueError(f"{type(self).__name__} has no parameter {key!r}")
            setattr(self, key, value)
        return self

    # -- helpers shared by the circuit-based models -----------------------
    def _resolve_qubits(self, n_features: int, encoding: str) -> int:
        if getattr(self, "n_qubits", None):
            return int(self.n_qubits)
        if encoding == "amplitude":
            return max(1, int(math.ceil(math.log2(max(n_features, 2)))))
        return int(n_features)

    def _build_feature_map(self, n_qubits: int, n_features: int):
        return get_feature_map(self.feature_map, n_qubits, n_features)

    def _build_ansatz(self, n_qubits: int, weight_offset: int = 0):
        return get_ansatz(self.ansatz, n_qubits, self.layers, weight_offset)

    # -- feature range ----------------------------------------------------
    def _fit_features(self, X) -> np.ndarray:
        """Fit the input scaler and return the transformed training features.

        Angle-style encodings map a feature onto a rotation, so a value outside
        [-1, 1] wraps past 2*pi and two distant points become indistinguishable.
        ``rescale="auto"`` (the default) min-max scales to [-1, 1] only when the
        training data actually leaves that range, which fixes the wrap-around
        without disturbing data that is already in range -- bit strings, say,
        where rescaling would destroy the structure the encoding relies on.
        """
        X = np.asarray(X)
        self._scaler_ = None
        mode = getattr(self, "rescale", "auto")
        if mode is False or np.iscomplexobj(X):
            return X
        low, high = X.min(axis=0), X.max(axis=0)
        if mode == "auto" and low.min() >= -1.0 and high.max() <= 1.0:
            return X
        span = np.where(high - low == 0, 1.0, high - low)
        self._scaler_ = (low, span)
        return self._transform_features(X)

    def _transform_features(self, X) -> np.ndarray:
        """Apply the fitted scaler; unseen values are clipped, not wrapped."""
        X = np.asarray(X)
        if getattr(self, "_scaler_", None) is None:
            return X
        low, span = self._scaler_
        return np.clip(2.0 * (X - low) / span - 1.0, -1.0, 1.0)

    def score(self, X, y, measure: Optional[str] = None) -> float:
        """Score with the model's natural measure unless one is given."""
        default = "accuracy" if self._estimator_type == "classifier" else "mean_squared_error"
        function = get_measure(measure or default)
        proba = self.predict_proba(X) if hasattr(self, "predict_proba") else None
        return function(y, self.predict(X), proba)

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in sorted(self.get_params().items())
                           if v is not None)
        return f"{type(self).__name__}({params})"


def adam(gradient_fn, initial, maxiter: int = 50, learning_rate: float = 0.1,
         beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8, callback=None,
         return_best: bool = True):
    """Plain Adam; the loop is here so no optimiser dependency is needed.

    ``return_best`` keeps the best iterate rather than the last one. With a
    stochastic mini-batch loss the final step is often not the best one, and
    silently reporting a worse model would flatter nobody.
    """
    theta = np.array(initial, dtype=float)
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    history = []
    best_loss, best_theta = float("inf"), theta.copy()
    for step in range(1, int(maxiter) + 1):
        loss, gradient = gradient_fn(theta)
        history.append(float(loss))
        if loss < best_loss:
            best_loss, best_theta = float(loss), theta.copy()
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * gradient ** 2
        m_hat = m / (1 - beta1 ** step)
        v_hat = v / (1 - beta2 ** step)
        theta = theta - learning_rate * m_hat / (np.sqrt(v_hat) + eps)
        if callback is not None:
            callback(step, float(loss), theta)
    return (best_theta if return_best else theta), history
