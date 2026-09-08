"""Variational models trained with parameter-shift gradients.

The gradient of a circuit costs two evaluations per gate parameter. Rather than
paying that as ``2P`` separate Python-level runs per sample, every shifted
parameter set and every sample in the mini-batch is stacked into one batched
simulation, so a training step is a handful of NumPy calls regardless of ``P``.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from ..backends import (DEFAULT_BACKEND, batched_z, batched_z_jacobian, get_backend,
                        prefers_adjoint, supports_adjoint)
from ..circuits.circuit import Circuit, Ref
from ..circuits.templates import data_reuploading, get_ansatz, get_feature_map
from .base import QuantumModel, adam, pad_to_power_of_two

__all__ = [
    "VariationalQuantumClassifier", "VariationalQuantumRegressor",
    "parameter_shift_gradient", "shift_sets", "GRADIENT_METHODS",
]

#: ``"auto"`` picks ``"adjoint"`` when it applies and ``"parameter_shift"`` otherwise.
GRADIENT_METHODS = ("auto", "adjoint", "parameter_shift")

SHIFT = math.pi / 2
EPSILON = 1e-4


def _weights_are_unique(circuit: Circuit) -> bool:
    """True when each weight drives exactly one gate (the two-term rule applies)."""
    indices = [index for _, index in circuit.parameter_indices_per_gate()]
    return len(indices) == len(set(indices))


def shift_sets(weights: np.ndarray, unique: bool = True) -> Tuple[np.ndarray, float]:
    """Stack ``[w, w+d*e_0, w-d*e_0, w+d*e_1, ...]`` into a ``(2P+1, P)`` array.

    Returns the array and the factor that turns a plus/minus difference into a
    derivative: exactly 1/2 for the parameter-shift rule, 1/(2*eps) when weight
    sharing forces central differences instead.
    """
    weights = np.asarray(weights, dtype=float)
    n = len(weights)
    delta = SHIFT if unique else EPSILON
    sets = np.repeat(weights[None, :], 2 * n + 1, axis=0)
    rows = np.arange(n)
    sets[1 + 2 * rows, rows] += delta
    sets[2 + 2 * rows, rows] -= delta
    return sets, (0.5 if unique else 1.0 / (2 * EPSILON))


def parameter_shift_gradient(evaluate: Callable[[np.ndarray], float], weights,
                             unique: bool = True, epsilon: float = EPSILON) -> np.ndarray:
    """Gradient of an **expectation value** with respect to the weights.

    ``evaluate`` must be linear in the circuit output -- an expectation value or
    an energy. The two-term rule is exact there because such a function is a
    sinusoid in each gate angle. It is *not* exact for a squared loss, which
    carries a second harmonic; differentiate the expectation values and apply
    the chain rule instead, as :meth:`_VariationalBase._loss_and_gradient` does.

    Kept for callers with their own evaluation loop; the models below use
    :func:`shift_sets` and evaluate every shift in one batch.
    """
    weights = np.asarray(weights, dtype=float)
    gradient = np.zeros_like(weights)
    delta = SHIFT if unique else epsilon
    scale = 0.5 if unique else 1.0 / (2 * epsilon)
    for index in range(len(weights)):
        plus = weights.copy(); plus[index] += delta
        minus = weights.copy(); minus[index] -= delta
        gradient[index] = scale * (evaluate(plus) - evaluate(minus))
    return gradient


class _VariationalBase(QuantumModel):
    """Shared plumbing: build the circuit, read out <Z>, train with Adam."""

    def _encoding(self) -> str:
        """Complex inputs are already states, so they load into the amplitudes."""
        return "amplitude" if getattr(self, "_amplitude_input", False) else self.feature_map

    def _prepare_X(self, X) -> np.ndarray:
        X = np.asarray(X)
        if np.iscomplexobj(X):
            states = pad_to_power_of_two(X.astype(complex))
            norms = np.linalg.norm(states, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return states / norms
        return X.astype(float)

    def _make_circuit(self, n_features: int) -> Circuit:
        encoding = self._encoding()
        n_qubits = self._resolve_qubits(n_features, encoding)
        if encoding == "reuploading":
            circuit = data_reuploading(n_qubits, n_features, self.layers)
        else:
            encoder = get_feature_map(encoding, n_qubits, n_features)
            circuit = encoder.compose(get_ansatz(self.ansatz, n_qubits, self.layers))
        self.n_qubits_ = circuit.n_qubits
        return circuit

    def _get_device(self):
        """One simulator per model: rebuilding it per row used to dominate."""
        device = getattr(self, "_device", None)
        if device is None or device.n_qubits != self.circuit_.n_qubits:
            device = get_backend(self.backend, self.circuit_.n_qubits, seed=self.seed,
                                 shots=self.shots)
            self._device = device
        return device

    def _outputs_prepared(self, weights, X) -> np.ndarray:
        """``(n_samples, n_readout)`` expectation values in one batched pass."""
        return batched_z(self.circuit_, self._readout_wires, weights, X,
                         device=self._get_device())

    def _outputs(self, weights, X) -> np.ndarray:
        """Same, for raw user input: scale first, then encode."""
        return self._outputs_prepared(weights, self._prepare_X(self._transform_features(X)))

    def _gradient_method(self, batch: int = 1) -> str:
        """Which differentiation rule this fit will actually use, and why.

        The adjoint sweep and the shift rule agree to machine precision on an
        exact simulator, and the sweep is O(P) state passes where the rule is
        O(P**2) -- so it is the default wherever it applies. It does not apply
        under shot noise (it is analytic), on a backend that cannot run it, or
        on a circuit carrying a gate whose generator is not registered.
        """
        requested = getattr(self, "gradient", "auto")
        if requested not in GRADIENT_METHODS:
            raise ValueError(f"unknown gradient method {requested!r}; "
                             f"expected one of {list(GRADIENT_METHODS)}")
        if requested == "parameter_shift":
            return "parameter_shift"
        device = self._get_device()
        usable = (not self.shots and supports_adjoint(device)
                  and all(device.is_differentiable(gate.name)
                          for gate in self.circuit_.gates
                          if any(isinstance(p, Ref) and p.kind == "weights"
                                 for p in gate.params)))
        if requested == "adjoint":
            if not usable:
                raise ValueError(
                    "gradient='adjoint' needs an analytic run (shots=None) on a backend "
                    "that implements z_jacobian, with a differentiable circuit"
                )
            return "adjoint"
        if usable and prefers_adjoint(batch, self.circuit_.n_qubits,
                                      self.circuit_.n_parameters, len(self._readout_wires)):
            return "adjoint"
        return "parameter_shift"

    def _loss_and_adjoint_gradient(self, weights, X, targets):
        """Square loss and its gradient from one forward pass and one sweep."""
        outputs, jacobian = batched_z_jacobian(self.circuit_, self._readout_wires, weights, X,
                                               device=self._get_device())
        residual = outputs - targets
        loss = float(np.mean(residual ** 2))
        gradient = 2.0 * np.einsum("bw,bwp->p", residual, jacobian) / residual.size
        return loss, gradient

    def _loss_and_gradient(self, weights, X, targets, unique):
        """Square loss and its exact gradient, evaluated in a single batch.

        All ``2P+1`` parameter sets are tiled against all ``n`` samples, so the
        simulator sees one array of ``(2P+1) * n`` circuits instead of that many
        separate calls.
        """
        sets, scale = shift_sets(weights, unique)
        n_sets, n_samples = len(sets), len(X)
        tiled_weights = np.repeat(sets, n_samples, axis=0)
        tiled_features = np.tile(X, (n_sets, 1))
        outputs = self._outputs_prepared(tiled_weights, tiled_features).reshape(n_sets, n_samples, -1)
        residual = outputs[0] - targets
        loss = float(np.mean(residual ** 2))
        differences = outputs[1::2] - outputs[2::2]           # (P, n_samples, n_readout)
        gradient = 2.0 * scale * np.mean(residual[None, ...] * differences, axis=(1, 2))
        return loss, gradient

    def _fit_loop(self, X, targets):
        rng = np.random.default_rng(self.seed)
        self._device = None
        self.circuit_ = self._make_circuit(X.shape[1])
        n_weights = max(self.circuit_.n_parameters, 1)
        unique = _weights_are_unique(self.circuit_)
        X = self._prepare_X(X)
        step_size = min(int(self.batch_size), len(X)) if self.batch_size else len(X)
        self.gradient_method_ = self._gradient_method(step_size)

        def step(weights):
            if self.batch_size and self.batch_size < len(X):
                batch = rng.choice(len(X), int(self.batch_size), replace=False)
            else:
                batch = slice(None)
            if self.gradient_method_ == "adjoint":
                return self._loss_and_adjoint_gradient(weights, X[batch], targets[batch])
            return self._loss_and_gradient(weights, X[batch], targets[batch], unique)

        weights, history = adam(step, rng.normal(0, 0.1, n_weights), self.maxiter,
                                self.learning_rate)
        self.weights_ = weights
        self.loss_curve_ = history
        return self


class VariationalQuantumClassifier(_VariationalBase):
    """Feature map + trainable ansatz, read out as <Z> on one wire per class.

    Cost is honest about itself: one Adam step evaluates ``(2P+1) * batch_size``
    circuits. Batching makes that one NumPy pass rather than that many, but it
    still grows with the parameter count.
    """

    _estimator_type = "classifier"

    def __init__(self, feature_map: str = "angle", ansatz: str = "hardware_efficient",
                 layers: int = 2, n_qubits: Optional[int] = None, maxiter: int = 60,
                 learning_rate: float = 0.2, batch_size: Optional[int] = None,
                 rescale="auto", shots: Optional[int] = None, gradient: str = "auto",
                 backend: str = DEFAULT_BACKEND, seed: int = 0):
        self.feature_map = feature_map
        self.ansatz = ansatz
        self.layers = layers
        self.n_qubits = n_qubits
        self.maxiter = maxiter
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.rescale = rescale
        self.shots = shots
        self.gradient = gradient
        self.backend = backend
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self._amplitude_input = np.iscomplexobj(X)
        X = self._fit_features(X)
        self.classes_ = np.unique(y)
        n_qubits = self._resolve_qubits(X.shape[1], self._encoding())
        if len(self.classes_) > n_qubits and len(self.classes_) > 2:
            raise ValueError(
                f"{len(self.classes_)} classes need at least that many read-out wires; "
                f"increase n_qubits (currently {n_qubits})"
            )
        self._readout_wires = [0] if len(self.classes_) == 2 else list(range(len(self.classes_)))
        if len(self.classes_) == 2:
            targets = np.where(y == self.classes_[0], -1.0, 1.0).reshape(-1, 1)
        else:
            targets = -np.ones((len(y), len(self.classes_)))
            for column, label in enumerate(self.classes_):
                targets[y == label, column] = 1.0
        return self._fit_loop(X, targets)

    def decision_function(self, X) -> np.ndarray:
        return self._outputs(self.weights_, np.asarray(X))

    def predict(self, X):
        scores = self.decision_function(X)
        if len(self.classes_) == 2:
            return np.where(scores[:, 0] < 0, self.classes_[0], self.classes_[1])
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X) -> np.ndarray:
        scores = self.decision_function(X)
        if len(self.classes_) == 2:
            positive = np.clip((scores[:, 0] + 1.0) / 2.0, 1e-6, 1 - 1e-6)
            return np.column_stack([1 - positive, positive])
        exponentiated = np.exp(2.0 * (scores - scores.max(axis=1, keepdims=True)))
        return exponentiated / exponentiated.sum(axis=1, keepdims=True)


class VariationalQuantumRegressor(_VariationalBase):
    """Data re-uploading regressor with a linear read-out.

    A re-uploading circuit is a truncated Fourier series in its input, which is
    why it fits periodic targets well and says nothing about anything else.
    """

    _estimator_type = "regressor"

    def __init__(self, feature_map: str = "reuploading", ansatz: str = "hardware_efficient",
                 layers: int = 3, n_qubits: Optional[int] = None, maxiter: int = 120,
                 learning_rate: float = 0.25, batch_size: Optional[int] = 24,
                 rescale="auto", shots: Optional[int] = None, gradient: str = "auto",
                 backend: str = DEFAULT_BACKEND, seed: int = 0):
        self.feature_map = feature_map
        self.ansatz = ansatz
        self.layers = layers
        self.n_qubits = n_qubits
        self.maxiter = maxiter
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.rescale = rescale
        self.shots = shots
        self.gradient = gradient
        self.backend = backend
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y, dtype=float)
        self._amplitude_input = np.iscomplexobj(X)
        X = self._fit_features(X)
        self._readout_wires = [0]
        self._scale, self._offset = 1.0, 0.0
        # scale into the range <Z> can actually reach, then undo it below
        low, high = float(y.min()), float(y.max())
        span = (high - low) or 1.0
        self._fit_loop(X, (1.8 * (y - low) / span - 0.9).reshape(-1, 1))
        # close the loop analytically: least squares from <Z> to the raw target
        raw = self._outputs_prepared(self.weights_, self._prepare_X(X))[:, 0]
        design = np.column_stack([raw, np.ones_like(raw)])
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        self._scale, self._offset = float(coefficients[0]), float(coefficients[1])
        return self

    def predict(self, X):
        raw = self._outputs(self.weights_, X)[:, 0]
        return self._scale * raw + self._offset
