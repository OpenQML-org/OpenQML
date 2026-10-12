"""NumPy-native learning models with fit/predict APIs and exact quantum features.

Training uses batched statevectors. Differentiation shifts individual quantum
expectations, then applies the loss chain rule; it never shifts the loss itself.
"""

import numpy as np

from ._validation import integer, real
from .circuit import rotation
from .exceptions import NotFittedError, ResourceLimitError
from .optimization import minimize
from .simulation import H, _apply_dense, _indices
from .simulation import X as PAULI_X


def _features(values, expected=None):
    result = np.asarray(values, dtype=float)
    if result.ndim == 1:
        result = result.reshape(-1, 1)
    if (
        result.ndim != 2
        or not result.shape[0]
        or not result.shape[1]
        or not np.isfinite(result).all()
    ):
        raise ValueError("X must be a nonempty finite (samples, features) array")
    if expected is not None and result.shape[1] != expected:
        raise ValueError(f"X must contain {expected} features")
    return result.copy()


def _labels(values, samples):
    result = np.asarray(values)
    if result.ndim != 1 or result.size != samples:
        raise ValueError("y must have one label per sample")
    if result.dtype.kind not in "biufUS" or (
        result.dtype.kind in "iuf" and not np.isfinite(result).all()
    ):
        raise ValueError("labels must be finite numbers or strings")
    return result.copy()


def _targets(values, samples):
    result = np.asarray(values, dtype=float)
    if result.shape != (samples,) or not np.isfinite(result).all():
        raise ValueError("y must be a finite one-dimensional array matching X")
    return result.copy()


def _fitted(model):
    if not getattr(model, "is_fitted_", False):
        raise NotFittedError("call fit(X, y) before prediction")


def _sigmoid(values):
    values = np.clip(values, -700, 700)
    return np.exp(-np.logaddexp(0, -values))


def _scale_fit(model, features, *, angle_scale=np.pi / 2):
    model.n_features_in_ = features.shape[1]
    model.feature_scale_ = np.maximum(np.max(np.abs(features), axis=0), 1e-12)
    model.angle_scale_ = angle_scale
    return features / model.feature_scale_ * model.angle_scale_


def _scale_predict(model, features):
    return _features(features, model.n_features_in_) / model.feature_scale_ * model.angle_scale_


def _initial_batch(samples, n_qubits):
    if (1 << n_qubits) * samples > 4_000_000:
        raise ResourceLimitError(
            "batch statevectors exceed 4 million amplitudes; use fewer samples"
        )
    states = np.zeros((1 << n_qubits, samples), complex)
    states[0] = 1
    return states


def _encode(states, angles):
    n_qubits = states.shape[0].bit_length() - 1
    for q in range(angles.shape[1]):
        indices = _indices(n_qubits, (q,))
        first, second = states[indices[0]].copy(), states[indices[1]].copy()
        c, s = np.cos(angles[:, q] / 2), np.sin(angles[:, q] / 2)
        states[indices[0]] = c * first - s * second
        states[indices[1]] = s * first + c * second
    return states


def _encode_phase(states, angles):
    indices = np.arange(states.shape[0])
    for q in range(angles.shape[1]):
        signs = 1 - 2 * ((indices >> q) & 1)
        states *= np.exp(-0.5j * signs[:, None] * angles[:, q])
    return states


def _entangle(states, n_qubits, *, ring=False):
    for q in range(n_qubits - 1):
        states = _apply_dense(states, PAULI_X, (q + 1,), (q,))
    if ring and n_qubits > 1:
        states = _apply_dense(states, PAULI_X, (0,), (n_qubits - 1,))
    return states


class QuantumKernel:
    """Fidelity kernel |<phi(x)|phi(y)>|² of a data re-uploading feature map.

    Inputs are angles in radians. Model wrappers learn feature scaling from the
    training set only. Exact Gram matrices are PSD up to floating-point error.
    """

    def __init__(self, *, layers=2, n_qubits=None):
        self.layers = integer(layers, "layers", 1, 32)
        self.n_qubits = None if n_qubits is None else integer(n_qubits, "n_qubits", 1, 8)

    def statevectors(self, features):
        features = _features(features)
        n_qubits = self.n_qubits or features.shape[1]
        if not 1 <= features.shape[1] <= n_qubits <= 8:
            raise ValueError(
                "feature map needs features <= n_qubits <= 8; reduce feature dimensions"
            )
        states = _initial_batch(features.shape[0], n_qubits)
        for layer in range(self.layers):
            states = _encode(states, features)
            for q in range(n_qubits):
                states = _apply_dense(states, rotation("z", (layer + 1) * (q + 1) * 0.37), (q,))
            states = _entangle(states, n_qubits)
        return states.T.copy()

    def pairwise(self, features, other=None):
        features = _features(features)
        other = features if other is None else _features(other, features.shape[1])
        if features.shape[0] * other.shape[0] > 4_000_000:
            raise ResourceLimitError("kernel matrix exceeds 4 million entries; use fewer samples")
        first = self.statevectors(features)
        second = first if other is features else self.statevectors(other)
        return np.abs(first.conj() @ second.T) ** 2

    __call__ = pairwise


class QuantumKernelClassifier:
    """Multiclass kernel ridge classifier using an exact quantum fidelity kernel.

    This is kernel ridge classification, not an SVM. predict_proba returns a
    softmax of ridge scores; these scores are not calibrated probabilities.
    """

    def __init__(self, *, layers=2, regularization=0.01, n_qubits=None):
        self.layers, self.n_qubits = layers, n_qubits
        self.regularization = real(regularization, "regularization")
        if self.regularization <= 0:
            raise ValueError("regularization must be positive")
        self.kernel = QuantumKernel(layers=layers, n_qubits=n_qubits)

    def fit(self, features, labels):
        self.is_fitted_ = False
        features = _features(features)
        labels = _labels(labels, features.shape[0])
        self.classes_, indices = np.unique(labels, return_inverse=True)
        if self.classes_.size < 2:
            raise ValueError("classification requires at least two classes")
        self.X_train_ = _scale_fit(self, features)
        gram = self.kernel.pairwise(self.X_train_)
        self.coefficients_ = np.linalg.solve(
            gram + self.regularization * np.eye(len(labels)), np.eye(self.classes_.size)[indices]
        )
        self.n_circuit_evaluations_ = len(labels)
        self.n_parameters_ = int(self.coefficients_.size)
        self.is_fitted_ = True
        return self

    def decision_function(self, features):
        _fitted(self)
        angles = _scale_predict(self, features)
        self.n_circuit_evaluations_ += len(angles) + len(self.X_train_)
        return self.kernel.pairwise(angles, self.X_train_) @ self.coefficients_

    def predict(self, features):
        _fitted(self)
        return self.classes_[np.argmax(self.decision_function(features), axis=1)]

    def predict_proba(self, features):
        scores = self.decision_function(features)
        scores -= scores.max(axis=1, keepdims=True)
        values = np.exp(scores)
        return values / values.sum(axis=1, keepdims=True)

    def score(self, features, labels):
        prediction = self.predict(features)
        return float(np.mean(prediction == _labels(labels, len(prediction))))

    def get_params(self):
        return {
            "layers": self.layers,
            "regularization": self.regularization,
            "n_qubits": self.n_qubits,
        }


class _VariationalBase:
    encoding_scale = np.pi / 2

    def __init__(self, *, n_qubits=None, layers=2, epochs=60, learning_rate=0.08, seed=1):
        self.n_qubits = None if n_qubits is None else integer(n_qubits, "n_qubits", 1, 8)
        self.layers = integer(layers, "layers", 1, 16)
        self.epochs = integer(epochs, "epochs", 1)
        self.learning_rate = real(learning_rate, "learning_rate")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self.seed = seed

    def get_params(self):
        return {
            "n_qubits": self.n_qubits,
            "layers": self.layers,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
        }

    def _prepare(self, features):
        self.n_qubits_ = self.n_qubits or features.shape[1]
        if not 1 <= features.shape[1] <= self.n_qubits_ <= 8:
            raise ValueError("variational models require features <= n_qubits <= 8")
        self.n_angles_ = 2 * self.layers * self.n_qubits_
        self.n_circuit_evaluations_ = 0
        return _scale_fit(self, features, angle_scale=self.encoding_scale)

    def _expectations(self, angles, parameters):
        self.n_circuit_evaluations_ += len(angles)
        states = _initial_batch(len(angles), self.n_qubits_)
        for q in range(self.n_qubits_):
            states = _apply_dense(states, H, (q,))
        for layer in range(self.layers):
            # Learn a phase offset before each data phase. Without this offset,
            # a one-qubit first harmonic can be constrained to cos(x), not sin(x).
            for q in range(self.n_qubits_):
                index = 2 * (layer * self.n_qubits_ + q)
                states = _apply_dense(states, rotation("z", parameters[index]), (q,))
            states = _encode_phase(states, angles)
            for q in range(self.n_qubits_):
                index = 2 * (layer * self.n_qubits_ + q)
                states = _apply_dense(states, rotation("y", parameters[index + 1]), (q,))
            states = _entangle(states, self.n_qubits_, ring=True)
        signs = 1 - 2 * (np.arange(1 << self.n_qubits_) & 1)
        return np.sum(signs[:, None] * np.abs(states) ** 2, axis=0)

    def _jacobian(self, angles, parameters):
        result = np.empty((len(angles), self.n_angles_))
        for i in range(self.n_angles_):
            offset = np.zeros(self.n_angles_)
            offset[i] = np.pi / 2
            result[:, i] = (
                self._expectations(angles, parameters + offset)
                - self._expectations(angles, parameters - offset)
            ) / 2
        return result

    def _fit_objective(self, objective, derivative, initial):
        fit = minimize(
            objective,
            initial,
            gradient_fn=derivative,
            steps=self.epochs,
            learning_rate=self.learning_rate,
        )
        self.parameters_ = fit.parameters.copy()
        self.n_parameters_ = int(fit.parameters.size)
        self.history_ = fit.history
        self.optimization_result_ = fit
        self.is_fitted_ = True


class VariationalQuantumClassifier(_VariationalBase):
    """Binary data re-uploading classifier trained with exact gate-wise gradients."""

    def fit(self, features, labels):
        self.is_fitted_ = False
        features = _features(features)
        labels = _labels(labels, len(features))
        self.classes_, targets = np.unique(labels, return_inverse=True)
        if self.classes_.size != 2:
            raise ValueError("VariationalQuantumClassifier supports exactly two classes")
        angles = self._prepare(features)
        initial = np.random.default_rng(self.seed).normal(0, 0.3, self.n_angles_)

        def probabilities(parameters):
            return np.clip((1 - self._expectations(angles, parameters)) / 2, 1e-7, 1 - 1e-7)

        def objective(parameters):
            p = probabilities(parameters)
            return float(-np.mean(targets * np.log(p) + (1 - targets) * np.log1p(-p)))

        def derivative(parameters):
            p = probabilities(parameters)
            sensitivity = (p - targets) / (p * (1 - p))
            return np.mean(
                sensitivity[:, None] * (-0.5) * self._jacobian(angles, parameters), axis=0
            )

        self._fit_objective(objective, derivative, initial)
        return self

    def predict_proba(self, features):
        _fitted(self)
        p = np.clip(
            (1 - self._expectations(_scale_predict(self, features), self.parameters_)) / 2, 0, 1
        )
        return np.column_stack([1 - p, p])

    def predict(self, features):
        _fitted(self)
        return self.classes_[np.argmax(self.predict_proba(features), axis=1)]

    def score(self, features, labels):
        prediction = self.predict(features)
        return float(np.mean(prediction == _labels(labels, len(prediction))))


class VariationalQuantumRegressor(_VariationalBase):
    """Data re-uploading expectation model with a trainable affine classical head."""

    encoding_scale = np.pi

    def fit(self, features, targets):
        self.is_fitted_ = False
        features = _features(features)
        targets = _targets(targets, len(features))
        angles = self._prepare(features)
        initial = np.r_[
            np.random.default_rng(self.seed).uniform(-np.pi, np.pi, self.n_angles_),
            max(float(np.std(targets)), 0.1),
            float(np.mean(targets)),
        ]

        def objective(parameters):
            prediction = (
                parameters[-2] * self._expectations(angles, parameters[:-2]) + parameters[-1]
            )
            return float(np.mean((prediction - targets) ** 2))

        def derivative(parameters):
            expectations = self._expectations(angles, parameters[:-2])
            residual = parameters[-2] * expectations + parameters[-1] - targets
            quantum = np.mean(
                2 * residual[:, None] * parameters[-2] * self._jacobian(angles, parameters[:-2]),
                axis=0,
            )
            return np.r_[quantum, np.mean(2 * residual * expectations), np.mean(2 * residual)]

        self._fit_objective(objective, derivative, initial)
        return self

    def predict(self, features):
        _fitted(self)
        expectations = self._expectations(_scale_predict(self, features), self.parameters_[:-2])
        return self.parameters_[-2] * expectations + self.parameters_[-1]

    def score(self, features, targets):
        prediction = self.predict(features)
        targets = _targets(targets, len(prediction))
        denominator = np.sum((targets - targets.mean()) ** 2)
        if denominator == 0:
            return 1.0 if np.allclose(prediction, targets) else 0.0
        return float(1 - np.sum((prediction - targets) ** 2) / denominator)


class LogisticClassifier:
    """Binary linear classical comparator; feature normalization is fit on training data."""

    def __init__(self, *, epochs=150, learning_rate=0.08, regularization=0.001, seed=1):
        self.epochs = integer(epochs, "epochs", 1)
        self.learning_rate = real(learning_rate, "learning_rate")
        self.regularization = real(regularization, "regularization")
        if self.learning_rate <= 0 or self.regularization < 0:
            raise ValueError("positive learning_rate and nonnegative regularization required")
        self.seed = seed

    def fit(self, features, labels):
        self.is_fitted_ = False
        features = _features(features)
        self.classes_, targets = np.unique(_labels(labels, len(features)), return_inverse=True)
        if self.classes_.size != 2:
            raise ValueError("LogisticClassifier supports exactly two classes")
        self.n_features_in_ = features.shape[1]
        self.mean_, self.scale_ = features.mean(axis=0), np.maximum(features.std(axis=0), 1e-12)
        design = np.c_[(features - self.mean_) / self.scale_, np.ones(len(features))]

        def objective(parameters):
            logits = design @ parameters
            return float(
                np.mean(np.logaddexp(0, logits) - targets * logits)
                + self.regularization * np.sum(parameters[:-1] ** 2) / 2
            )

        def derivative(parameters):
            return (
                design.T @ (_sigmoid(design @ parameters) - targets) / len(features)
                + np.r_[self.regularization * parameters[:-1], 0]
            )

        fit = minimize(
            objective,
            np.zeros(design.shape[1]),
            gradient_fn=derivative,
            steps=self.epochs,
            learning_rate=self.learning_rate,
        )
        self.parameters_, self.history_ = fit.parameters, fit.history
        self.n_parameters_ = int(fit.parameters.size)
        self.is_fitted_ = True
        return self

    def predict_proba(self, features):
        _fitted(self)
        features = _features(features, self.n_features_in_)
        p = _sigmoid(
            ((features - self.mean_) / self.scale_) @ self.parameters_[:-1] + self.parameters_[-1]
        )
        return np.c_[1 - p, p]

    def predict(self, features):
        _fitted(self)
        return self.classes_[np.argmax(self.predict_proba(features), axis=1)]

    def score(self, features, labels):
        prediction = self.predict(features)
        return float(np.mean(prediction == _labels(labels, len(prediction))))

    def get_params(self):
        return {
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "regularization": self.regularization,
            "seed": self.seed,
        }


class RidgeRegressor:
    """Linear ridge baseline (with an unpenalized intercept)."""

    def __init__(self, *, regularization=0.01):
        self.regularization = real(regularization, "regularization")
        if self.regularization <= 0:
            raise ValueError("regularization must be positive")

    def _design(self, features):
        return np.c_[(features - self.mean_) / self.scale_, np.ones(len(features))]

    def fit(self, features, targets):
        self.is_fitted_ = False
        features = _features(features)
        targets = _targets(targets, len(features))
        self.n_features_in_ = features.shape[1]
        self.mean_, self.scale_ = features.mean(axis=0), np.maximum(features.std(axis=0), 1e-12)
        design = self._design(features)
        penalty = np.diag([self.regularization] * (design.shape[1] - 1) + [0])
        self.parameters_ = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
        self.n_parameters_ = int(self.parameters_.size)
        self.is_fitted_ = True
        return self

    def predict(self, features):
        _fitted(self)
        return self._design(_features(features, self.n_features_in_)) @ self.parameters_

    def score(self, features, targets):
        prediction = self.predict(features)
        targets = _targets(targets, len(prediction))
        denominator = np.sum((targets - targets.mean()) ** 2)
        if denominator == 0:
            return 1.0 if np.allclose(prediction, targets) else 0.0
        return float(1 - np.sum((prediction - targets) ** 2) / denominator)

    def get_params(self):
        return {"regularization": self.regularization}


class FourierRegressor(RidgeRegressor):
    """Strong classical baseline for periodic targets; frequencies are fixed in radians."""

    def __init__(self, *, harmonics=6, regularization=0.01):
        super().__init__(regularization=regularization)
        self.harmonics = integer(harmonics, "harmonics", 1, 100)

    def _design(self, features):
        columns = [np.ones(len(features))]
        for q in range(features.shape[1]):
            for frequency in range(1, self.harmonics + 1):
                columns.extend(
                    [np.sin(frequency * features[:, q]), np.cos(frequency * features[:, q])]
                )
        # Intercept last, matching RidgeRegressor's unpenalized final coefficient.
        return np.column_stack(columns[1:] + columns[:1])

    def get_params(self):
        return {"harmonics": self.harmonics, "regularization": self.regularization}
