"""Built-in models. Any scikit-learn style estimator works here too."""

from .base import QuantumModel, adam, pad_to_power_of_two
from .baselines import LinearRegressor, MajorityClassifier
from .kernel import (
    ClassicalKernelClassifier,
    ClassicalKernelRegressor,
    QuantumKernelClassifier,
    QuantumKernelRegressor,
    clear_state_cache,
    encode_states,
    quantum_kernel,
)
from .variational import (
    VariationalQuantumClassifier,
    VariationalQuantumRegressor,
    parameter_shift_gradient,
)
from .vqe import VQE, ExactDiagonalisation

__all__ = [
    "QuantumModel", "adam",
    "QuantumKernelClassifier", "QuantumKernelRegressor",
    "ClassicalKernelClassifier", "ClassicalKernelRegressor",
    "VariationalQuantumClassifier", "VariationalQuantumRegressor",
    "VQE", "ExactDiagonalisation",
    "MajorityClassifier", "LinearRegressor",
    "quantum_kernel", "encode_states", "parameter_shift_gradient",
    "clear_state_cache", "pad_to_power_of_two",
    "list_models",
]


def list_models():
    """Name -> one-line summary of every built-in model."""
    return {
        "QuantumKernelClassifier": "fidelity kernel + kernel ridge (no training loop)",
        "QuantumKernelRegressor": "fidelity kernel regression",
        "ClassicalKernelClassifier": "same solver, RBF/linear/poly kernel (control arm)",
        "ClassicalKernelRegressor": "classical kernel ridge regression",
        "VariationalQuantumClassifier": "feature map + ansatz, parameter-shift training",
        "VariationalQuantumRegressor": "data re-uploading regressor with linear read-out",
        "VQE": "variational ground-state energy",
        "ExactDiagonalisation": "exact reference solver",
        "MajorityClassifier": "majority-class floor",
        "LinearRegressor": "ridge regression, optional Fourier basis",
    }
