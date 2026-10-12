"""Hybrid optimization without framework or service dependencies."""

from dataclasses import dataclass

import numpy as np

from ._validation import integer, real, vector


def _value(function, parameters):
    return real(function(parameters.copy()), "objective value")


def gradient(function, parameters, *, epsilon=1e-6):
    """Central finite differences; valid for nonlinear losses and shared angles."""
    parameters = vector(parameters)
    epsilon = real(epsilon, "epsilon")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    result = np.zeros_like(parameters)
    for index in range(parameters.size):
        offset = np.zeros_like(parameters)
        offset[index] = epsilon
        result[index] = (
            _value(function, parameters + offset) - _value(function, parameters - offset)
        ) / (2 * epsilon)
    return result


def parameter_shift(function, parameters):
    """±pi/2 shift of an expectation, assuming one unscaled Pauli rotation per angle.

    Do NOT directly shift nonlinear losses, repeated/shared angles, or controlled
    rotations. Differentiate their gate expectations and apply the chain rule,
    or call gradient() on the full objective instead.
    """
    parameters = vector(parameters)
    result = np.zeros_like(parameters)
    for index in range(parameters.size):
        offset = np.zeros_like(parameters)
        offset[index] = np.pi / 2
        result[index] = (
            _value(function, parameters + offset) - _value(function, parameters - offset)
        ) / 2
    return result


@dataclass
class OptimizeResult:
    parameters: np.ndarray
    value: float
    history: list[float]
    n_steps: int
    n_evaluations: int
    converged: bool


def minimize(
    function,
    initial,
    *,
    gradient_fn=None,
    steps=100,
    learning_rate=0.05,
    method="adam",
    tolerance=1e-8,
):
    """Minimize a scalar objective with Adam or gradient descent with backtracking.

    The returned parameters/value are the best evaluated iterate. History holds
    the actual initial value and each accepted iterate. n_evaluations includes
    finite differences; user-supplied gradient function calls are not counted.
    """
    parameters = vector(initial)
    steps = integer(steps, "steps", 0)
    learning_rate, tolerance = real(learning_rate, "learning_rate"), real(tolerance, "tolerance")
    if learning_rate <= 0 or tolerance < 0 or method not in ("adam", "gd"):
        raise ValueError(
            "positive learning_rate, nonnegative tolerance, method 'adam'/'gd' required"
        )
    evaluations = 0

    def objective(p):
        nonlocal evaluations
        evaluations += 1
        return _value(function, p)

    value = objective(parameters)
    best_parameters, best_value = parameters.copy(), value
    history, first, second = [value], np.zeros_like(parameters), np.zeros_like(parameters)
    converged = False
    for step in range(1, steps + 1):
        grad = vector(
            gradient_fn(parameters.copy())
            if gradient_fn is not None
            else gradient(objective, parameters),
            "gradient",
        )
        if grad.shape != parameters.shape:
            raise ValueError("gradient must have the same shape as parameters")
        if np.linalg.norm(grad) <= tolerance:
            converged = True
            break
        if method == "adam":
            first = 0.9 * first + 0.1 * grad
            second = 0.999 * second + 0.001 * grad**2
            candidate = parameters - learning_rate * (first / (1 - 0.9**step)) / (
                np.sqrt(second / (1 - 0.999**step)) + 1e-8
            )
            next_value = objective(candidate)
        else:
            rate = learning_rate
            for _ in range(30):
                candidate = parameters - rate * grad
                next_value = objective(candidate)
                if next_value <= value - 1e-4 * rate * np.dot(grad, grad):
                    break
                rate *= 0.5
            else:
                break
        parameters, value = candidate, next_value
        history.append(value)
        if value < best_value:
            best_parameters, best_value = parameters.copy(), value
    return OptimizeResult(
        best_parameters, best_value, history, len(history) - 1, evaluations, converged
    )
