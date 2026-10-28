"""Hybrid quantum learning and explicit circuit interoperability.

Install once with ``python -m pip install openqml`` and import as
``import openqml as oq``. All integrations load only when accessed.
"""

from importlib import import_module as _import_module

from ._scheduling import ExecutionBatch, ReadoutRoute
from .adjoint import AdjointPlan, adjoint_jvp, adjoint_vjp
from .aer import AerBackend
from .backends import (
    BackendCapabilities,
    DensityMatrixBackend,
    ExecutionOptions,
    ExpectationBackend,
    SamplingBackend,
    StatevectorBackend,
    get_backend,
    list_backends,
    register_backend,
    unregister_backend,
)
from .batch import execute_batch, expectations_batch
from .compiler import CircuitStats, circuit_stats, decompose, inverse, simplify
from .conversion import ConversionPlan, ConversionResult, convert, diagnose, plan_conversion
from .density import density_matrix
from .environment import environment_report
from .errors import (
    BackendError,
    ConversionError,
    Diagnostic,
    MissingDependencyError,
    OpenQMLError,
    ResourceError,
    ValidationError,
)
from .execution import ExecutionGroup, ExpectationCheck, ExpectationPlan
from .frontends import FrameworkAdapter, list_formats, register_adapter, unregister_adapter
from .gradients import (
    ShiftPlan,
    parameter_shift,
    parameter_shift_jacobian,
    parameter_shift_jvp,
    parameter_shift_vjp,
)
from .inspection import PythonSourceReport, inspect_python
from .ir import Circuit, CircuitBuilder, ClassicalCondition, Operation
from .layers import LayerCheckpoint, LayerSpec
from .measurement import EstimateResult, MeasurementPlan, ShotBackend, estimate, measurement_plan
from .observables import Hamiltonian, PauliString, z_observables
from .optimization import OptimizationResult, optimize
from .parameters import Expression, Parameter
from .pauli import PauliPlan
from .planning import DensityPlan, plan_density
from .program import QuantumProgram, as_circuit
from .providers import CirqDensityBackend, QiskitEstimatorBackend, QiskitSamplerBackend
from .readout import ReadoutPlan, plan_readout
from .resources import ResourceEstimate, estimate_resources
from .simulator import ExecutionResult, execute, expectation, expectations, sample, statevector
from .templates import data_reuploading, hardware_efficient
from .verification import ConversionCheck

_LAZY_MODULES = {
    "torch": ".interfaces.torch",
    "jax": ".interfaces.jax",
    "flax": ".interfaces.flax",
    "numpy": "numpy",
    "optax": "optax",
    "qiskit": "qiskit",
    "pennylane": "pennylane",
    "cirq": "cirq",
    "qiskit_aer": "qiskit_aer",
    "openqasm3": "openqasm3",
    "sympy": "sympy",
}


def __getattr__(name):
    if name in _LAZY_MODULES:
        try:
            module = _import_module(_LAZY_MODULES[name], __name__)
        except ImportError as exc:
            raise MissingDependencyError(
                f"Cannot load {name}. Run: python -m pip install openqml."
            ) from exc
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _LAZY_MODULES.keys())


__version__ = "0.3.0a13"

__all__ = [
    "environment_report",
    "PythonSourceReport",
    "inspect_python",
    "ResourceEstimate",
    "estimate_resources",
    "ReadoutRoute",
    "ConversionCheck",
    "ExecutionBatch",
    "ExpectationCheck",
    "ExpectationPlan",
    "ExecutionGroup",
    "PauliPlan",
    "ReadoutPlan",
    "plan_readout",
    "AdjointPlan",
    "ShiftPlan",
    "adjoint_vjp",
    "adjoint_jvp",
    "ConversionPlan",
    "plan_conversion",
    "data_reuploading",
    "LayerCheckpoint",
    "OptimizationResult",
    "optimize",
    "DensityPlan",
    "plan_density",
    "Circuit",
    "CircuitBuilder",
    "ClassicalCondition",
    "density_matrix",
    "DensityMatrixBackend",
    "Operation",
    "Parameter",
    "Expression",
    "PauliString",
    "Hamiltonian",
    "statevector",
    "expectation",
    "expectations",
    "execute",
    "sample",
    "parameter_shift",
    "convert",
    "diagnose",
    "hardware_efficient",
    "z_observables",
    "LayerSpec",
    "ConversionResult",
    "ExecutionResult",
    "Diagnostic",
    "OpenQMLError",
    "ValidationError",
    "ConversionError",
    "ResourceError",
    "MissingDependencyError",
    "BackendError",
    "BackendCapabilities",
    "ExecutionOptions",
    "StatevectorBackend",
    "ExpectationBackend",
    "SamplingBackend",
    "register_backend",
    "unregister_backend",
    "list_backends",
    "get_backend",
    "CircuitStats",
    "circuit_stats",
    "decompose",
    "inverse",
    "simplify",
    "QuantumProgram",
    "as_circuit",
    "execute_batch",
    "expectations_batch",
    "parameter_shift_jacobian",
    "parameter_shift_vjp",
    "parameter_shift_jvp",
    "estimate",
    "measurement_plan",
    "MeasurementPlan",
    "EstimateResult",
    "ShotBackend",
    "AerBackend",
    "CirqDensityBackend",
    "QiskitEstimatorBackend",
    "QiskitSamplerBackend",
    "FrameworkAdapter",
    "list_formats",
    "register_adapter",
    "unregister_adapter",
]
