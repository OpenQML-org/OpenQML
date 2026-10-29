"""OpenQML: hybrid quantum learning and explicit circuit interoperability."""

from .adjoint import adjoint_jvp, adjoint_vjp
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
from .errors import (
    BackendError,
    ConversionError,
    Diagnostic,
    MissingDependencyError,
    OpenQMLError,
    ResourceError,
    ValidationError,
)
from .frontends import FrameworkAdapter, list_formats, register_adapter, unregister_adapter
from .gradients import (
    parameter_shift,
    parameter_shift_jacobian,
    parameter_shift_jvp,
    parameter_shift_vjp,
)
from .ir import Circuit, ClassicalCondition, Operation
from .layers import LayerCheckpoint, LayerSpec
from .measurement import EstimateResult, MeasurementPlan, ShotBackend, estimate, measurement_plan
from .observables import Hamiltonian, PauliString, z_observables
from .optimization import OptimizationResult, optimize
from .parameters import Expression, Parameter
from .planning import DensityPlan, plan_density
from .program import QuantumProgram, as_circuit
from .providers import CirqDensityBackend, QiskitEstimatorBackend, QiskitSamplerBackend
from .simulator import ExecutionResult, execute, expectation, expectations, sample, statevector
from .templates import data_reuploading, hardware_efficient

__version__ = "0.3.0a8"

__all__ = [
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
