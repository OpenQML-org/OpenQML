"""OpenQML: importable quantum programming, models, and local benchmark tasks."""

__version__ = "0.1.0"

from . import algorithms, benchmarks, datasets, models
from .algorithms import bell, ghz, grover, hardware_efficient_ansatz, qaoa_maxcut, qft, vqe
from .benchmarks import (
    BenchmarkResult,
    ComparisonResult,
    Suite,
    Task,
    compare_models,
    get_model,
    get_suite,
    get_task,
    list_models,
    make_task,
    run_benchmark,
)
from .circuit import Circuit
from .datasets import Dataset, list_datasets, load_dataset, train_test_split
from .exceptions import (
    NotFittedError,
    OpenQMLError,
    ResourceLimitError,
    UnsupportedOperationError,
)
from .hamiltonian import Hamiltonian, ising_hamiltonian
from .interop import from_qasm, to_qasm, to_qiskit
from .models import (
    FourierRegressor,
    LogisticClassifier,
    QuantumKernel,
    QuantumKernelClassifier,
    RidgeRegressor,
    VariationalQuantumClassifier,
    VariationalQuantumRegressor,
)
from .optimization import OptimizeResult, gradient, minimize, parameter_shift
from .simulation import SimulationBudget, State

__all__ = [
    "__version__",
    "Circuit",
    "State",
    "SimulationBudget",
    "Hamiltonian",
    "ising_hamiltonian",
    "gradient",
    "parameter_shift",
    "minimize",
    "OptimizeResult",
    "QuantumKernel",
    "QuantumKernelClassifier",
    "VariationalQuantumClassifier",
    "VariationalQuantumRegressor",
    "LogisticClassifier",
    "RidgeRegressor",
    "FourierRegressor",
    "Dataset",
    "list_datasets",
    "load_dataset",
    "train_test_split",
    "Task",
    "Suite",
    "get_task",
    "make_task",
    "get_suite",
    "list_models",
    "get_model",
    "run_benchmark",
    "compare_models",
    "BenchmarkResult",
    "ComparisonResult",
    "bell",
    "ghz",
    "grover",
    "qft",
    "qaoa_maxcut",
    "vqe",
    "hardware_efficient_ansatz",
    "to_qasm",
    "from_qasm",
    "to_qiskit",
    "OpenQMLError",
    "NotFittedError",
    "ResourceLimitError",
    "UnsupportedOperationError",
    "algorithms",
    "benchmarks",
    "datasets",
    "models",
]
