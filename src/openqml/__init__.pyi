import cirq as cirq
import numpy as numpy
import openqasm3 as openqasm3
import optax as optax
import pennylane as pennylane
import qiskit as qiskit
import qiskit_aer as qiskit_aer
import sympy as sympy

from . import adaptive as adaptive
from . import adaptive_mps as adaptive_mps
from . import adjoint as adjoint
from . import aer as aer
from . import amplitude as amplitude
from . import architecture as architecture
from . import architecture_runtime as architecture_runtime
from . import autoregressive as autoregressive
from . import backends as backends
from . import batch as batch
from . import calibration as calibration
from . import cli as cli
from . import compiler as compiler
from . import components as components
from . import conversion as conversion
from . import costs as costs
from . import deep as deep
from . import density as density
from . import distributed_tensor as distributed_tensor
from . import environment as environment
from . import errors as errors
from . import execution as execution
from . import finance as finance
from . import frontends as frontends
from . import geometry as geometry
from . import gradients as gradients
from . import inspection as inspection
from . import iqp as iqp
from . import ir as ir
from . import layers as layers
from . import measurement as measurement
from . import models as models
from . import mps as mps
from . import multiscale as multiscale
from . import network_attention as network_attention
from . import network_graph as network_graph
from . import network_operators as network_operators
from . import network_training as network_training
from . import observables as observables
from . import optimization as optimization
from . import parameters as parameters
from . import pauli as pauli
from . import pauli_fusion as pauli_fusion
from . import pauli_transfer as pauli_transfer
from . import planning as planning
from . import program as program
from . import program_compilation as program_compilation
from . import providers as providers
from . import quantum_activations as quantum_activations
from . import quantum_cells as quantum_cells
from . import quantum_channels as quantum_channels
from . import quantum_generative as quantum_generative
from . import quantum_measurement as quantum_measurement
from . import quantum_tree as quantum_tree
from . import readout as readout
from . import resources as resources
from . import rl as rl
from . import rl_training as rl_training
from . import runtime as runtime
from . import semantics as semantics
from . import shadows as shadows
from . import simulator as simulator
from . import sparse_pauli as sparse_pauli
from . import templates as templates
from . import tensor_contraction as tensor_contraction
from . import tensor_train as tensor_train
from . import truncated_mps as truncated_mps
from . import verification as verification
from ._native_training import NativeNaturalGradientResult as NativeNaturalGradientResult
from ._scheduling import ExecutionBatch as ExecutionBatch
from ._scheduling import ReadoutRoute as ReadoutRoute
from .adaptive import (
    AdaptivePlan as AdaptivePlan,
)
from .adaptive import (
    ExecutionCandidate as ExecutionCandidate,
)
from .adaptive import (
    plan_execution as plan_execution,
)
from .adaptive_amplitude import (
    AdaptiveAmplitudeEstimate as AdaptiveAmplitudeEstimate,
)
from .adaptive_amplitude import (
    AdaptiveAmplitudeEstimationPlan as AdaptiveAmplitudeEstimationPlan,
)
from .adaptive_mps import (
    AdaptiveMPSBackend as AdaptiveMPSBackend,
)
from .adaptive_mps import (
    AdaptiveMPSResult as AdaptiveMPSResult,
)
from .adjoint import (
    AdjointPlan as AdjointPlan,
)
from .adjoint import (
    adjoint_jvp as adjoint_jvp,
)
from .adjoint import (
    adjoint_vjp as adjoint_vjp,
)
from .aer import AerBackend as AerBackend
from .amplitude import (
    AmplitudeEstimate as AmplitudeEstimate,
)
from .amplitude import (
    AmplitudeEstimationPlan as AmplitudeEstimationPlan,
)
from .amplitude import (
    AmplitudeOracle as AmplitudeOracle,
)
from .architecture import (
    HybridArchitecture as HybridArchitecture,
)
from .architecture import (
    NetworkParameter as NetworkParameter,
)
from .architecture import (
    QuantumCellSpec as QuantumCellSpec,
)
from .architecture_runtime import NetworkExecutionPlan as NetworkExecutionPlan
from .autoregressive import (
    detach_quantum_state as detach_quantum_state,
)
from .autoregressive import (
    sample_tokens as sample_tokens,
)
from .backends import (
    BackendCapabilities as BackendCapabilities,
)
from .backends import (
    DensityMatrixBackend as DensityMatrixBackend,
)
from .backends import (
    ExecutionOptions as ExecutionOptions,
)
from .backends import (
    ExpectationBackend as ExpectationBackend,
)
from .backends import (
    SamplingBackend as SamplingBackend,
)
from .backends import (
    StatevectorBackend as StatevectorBackend,
)
from .backends import (
    get_backend as get_backend,
)
from .backends import (
    list_backends as list_backends,
)
from .backends import (
    register_backend as register_backend,
)
from .backends import (
    unregister_backend as unregister_backend,
)
from .batch import execute_batch as execute_batch
from .batch import expectations_batch as expectations_batch
from .calibration import (
    CalibrationModel as CalibrationModel,
)
from .calibration import (
    GateCalibration as GateCalibration,
)
from .calibration import (
    NoiseEstimate as NoiseEstimate,
)
from .compiler import (
    CircuitStats as CircuitStats,
)
from .compiler import (
    circuit_stats as circuit_stats,
)
from .compiler import (
    decompose as decompose,
)
from .compiler import (
    inverse as inverse,
)
from .compiler import (
    simplify as simplify,
)
from .components import ComponentPlan as ComponentPlan
from .components import plan_components as plan_components
from .conversion import (
    ConversionPlan as ConversionPlan,
)
from .conversion import (
    ConversionResult as ConversionResult,
)
from .conversion import (
    convert as convert,
)
from .conversion import (
    diagnose as diagnose,
)
from .conversion import (
    plan_conversion as plan_conversion,
)
from .costs import (
    BudgetLedger as BudgetLedger,
)
from .costs import (
    ExecutionCost as ExecutionCost,
)
from .costs import (
    ExecutionFeedback as ExecutionFeedback,
)
from .costs import (
    ExecutionProfile as ExecutionProfile,
)
from .costs import (
    SchedulingPolicy as SchedulingPolicy,
)
from .costs import (
    SchedulingRequest as SchedulingRequest,
)
from .costs import (
    TrainingWorkload as TrainingWorkload,
)
from .deep import DeepQuantumPlan as DeepQuantumPlan
from .density import density_matrix as density_matrix
from .distributed_tensor import (
    DistributedTensorPlan as DistributedTensorPlan,
)
from .distributed_tensor import (
    Interconnect as Interconnect,
)
from .environment import environment_report as environment_report
from .errors import (
    BackendError as BackendError,
)
from .errors import (
    ConversionError as ConversionError,
)
from .errors import (
    Diagnostic as Diagnostic,
)
from .errors import (
    MissingDependencyError as MissingDependencyError,
)
from .errors import (
    OpenQMLError as OpenQMLError,
)
from .errors import (
    ResourceError as ResourceError,
)
from .errors import (
    ValidationError as ValidationError,
)
from .execution import (
    ExecutionGroup as ExecutionGroup,
)
from .execution import (
    ExpectationCheck as ExpectationCheck,
)
from .execution import (
    ExpectationPlan as ExpectationPlan,
)
from .frontends import (
    FrameworkAdapter as FrameworkAdapter,
)
from .frontends import (
    list_formats as list_formats,
)
from .frontends import (
    register_adapter as register_adapter,
)
from .frontends import (
    unregister_adapter as unregister_adapter,
)
from .geometry import (
    NaturalGradientResult as NaturalGradientResult,
)
from .geometry import (
    QuantumMetricPlan as QuantumMetricPlan,
)
from .geometry import (
    quantum_metric as quantum_metric,
)
from .gradients import (
    ShiftPlan as ShiftPlan,
)
from .gradients import (
    parameter_shift as parameter_shift,
)
from .gradients import (
    parameter_shift_jacobian as parameter_shift_jacobian,
)
from .gradients import (
    parameter_shift_jvp as parameter_shift_jvp,
)
from .gradients import (
    parameter_shift_vjp as parameter_shift_vjp,
)
from .inspection import PythonSourceReport as PythonSourceReport
from .inspection import inspect_python as inspect_python
from .interfaces import flax as flax
from .interfaces import jax as jax
from .interfaces import torch as torch
from .iqp import IQPGraph as IQPGraph
from .ir import (
    Circuit as Circuit,
)
from .ir import (
    CircuitBuilder as CircuitBuilder,
)
from .ir import (
    ClassicalCondition as ClassicalCondition,
)
from .ir import (
    Operation as Operation,
)
from .layers import LayerCheckpoint as LayerCheckpoint
from .layers import LayerSpec as LayerSpec
from .measurement import (
    AdaptiveEstimateResult as AdaptiveEstimateResult,
)
from .measurement import (
    EstimateResult as EstimateResult,
)
from .measurement import (
    MeasurementPlan as MeasurementPlan,
)
from .measurement import (
    ShotBackend as ShotBackend,
)
from .measurement import (
    estimate as estimate,
)
from .measurement import (
    estimate_adaptive as estimate_adaptive,
)
from .measurement import (
    measurement_plan as measurement_plan,
)
from .models import Model as Model
from .mps import MPSBackend as MPSBackend
from .multiscale import MultiscalePlan as MultiscalePlan
from .network_operators import (
    NetworkOperator as NetworkOperator,
)
from .network_operators import (
    OperatorSpec as OperatorSpec,
)
from .network_operators import (
    list_network_operators as list_network_operators,
)
from .network_operators import (
    register_network_operator as register_network_operator,
)
from .network_training import (
    DepthSchedule as DepthSchedule,
)
from .network_training import (
    StagewiseOptimizer as StagewiseOptimizer,
)
from .observables import (
    Hamiltonian as Hamiltonian,
)
from .observables import (
    PauliString as PauliString,
)
from .observables import (
    SparseHamiltonian as SparseHamiltonian,
)
from .observables import (
    SparsePauliString as SparsePauliString,
)
from .observables import (
    z_observables as z_observables,
)
from .optimization import (
    CertifiedOptimization as CertifiedOptimization,
)
from .optimization import (
    OptimizationResult as OptimizationResult,
)
from .optimization import (
    optimize as optimize,
)
from .optimization import (
    optimize_verified as optimize_verified,
)
from .parameters import Expression as Expression
from .parameters import Parameter as Parameter
from .pauli import PauliPlan as PauliPlan
from .pauli_fusion import (
    PauliFusionResult as PauliFusionResult,
)
from .pauli_fusion import (
    fuse_pauli_rotations as fuse_pauli_rotations,
)
from .pauli_transfer import PauliTransferPlan as PauliTransferPlan
from .planning import DensityPlan as DensityPlan
from .planning import plan_density as plan_density
from .program import QuantumProgram as QuantumProgram
from .program import as_circuit as as_circuit
from .program_compilation import (
    ProgramConversion as ProgramConversion,
)
from .program_compilation import (
    compile_program as compile_program,
)
from .providers import (
    CirqDensityBackend as CirqDensityBackend,
)
from .providers import (
    QiskitEstimatorBackend as QiskitEstimatorBackend,
)
from .providers import (
    QiskitSamplerBackend as QiskitSamplerBackend,
)
from .quantum_activations import QuantumActivationPlan as QuantumActivationPlan
from .quantum_cells import FeatureCircuitPlan as FeatureCircuitPlan
from .quantum_channels import QuantumChannelPlan as QuantumChannelPlan
from .quantum_generative import (
    BornCircuitPlan as BornCircuitPlan,
)
from .quantum_generative import (
    QuantumDiffusionPlan as QuantumDiffusionPlan,
)
from .quantum_generative import (
    born_nll as born_nll,
)
from .quantum_generative import (
    born_probabilities as born_probabilities,
)
from .quantum_generative import (
    shifted_tokens as shifted_tokens,
)
from .quantum_measurement import QuantumPOVMPlan as QuantumPOVMPlan
from .quantum_tree import QuantumTreePlan as QuantumTreePlan
from .readout import ReadoutPlan as ReadoutPlan
from .readout import plan_readout as plan_readout
from .resources import (
    ResourceEstimate as ResourceEstimate,
)
from .resources import (
    estimate_resources as estimate_resources,
)
from .rl import (
    ActorCriticOutput as ActorCriticOutput,
)
from .rl import (
    AdvantageTargets as AdvantageTargets,
)
from .rl import (
    CategoricalStatistics as CategoricalStatistics,
)
from .rl import (
    PPOLoss as PPOLoss,
)
from .rl import (
    SACLoss as SACLoss,
)
from .rl import (
    VTraceTargets as VTraceTargets,
)
from .rl import (
    categorical_statistics as categorical_statistics,
)
from .rl import (
    generalized_advantages as generalized_advantages,
)
from .rl import (
    ppo_loss as ppo_loss,
)
from .rl import (
    sac_loss as sac_loss,
)
from .rl import (
    squashed_gaussian as squashed_gaussian,
)
from .rl import (
    squashed_gaussian_log_probability as squashed_gaussian_log_probability,
)
from .rl import (
    surrogate_fit_loss as surrogate_fit_loss,
)
from .rl import (
    vtrace_targets as vtrace_targets,
)
from .rl_training import PPOConfig as PPOConfig
from .runtime import (
    CostAwareBackend as CostAwareBackend,
)
from .runtime import (
    DispatchCandidate as DispatchCandidate,
)
from .runtime import (
    DispatchPlan as DispatchPlan,
)
from .runtime import (
    PriorityBackend as PriorityBackend,
)
from .runtime import (
    QPUBackend as QPUBackend,
)
from .runtime import (
    configure_runtime as configure_runtime,
)
from .runtime import (
    register_qpu_backend as register_qpu_backend,
)
from .semantics import (
    CertifiedConversion as CertifiedConversion,
)
from .semantics import (
    EquivalenceProof as EquivalenceProof,
)
from .semantics import (
    compile_verified as compile_verified,
)
from .semantics import (
    prove_equivalence as prove_equivalence,
)
from .semantics import (
    semantic_coverage as semantic_coverage,
)
from .shadows import ShadowEstimate as ShadowEstimate
from .shadows import shadow_estimate as shadow_estimate
from .simulator import (
    ExecutionResult as ExecutionResult,
)
from .simulator import (
    execute as execute,
)
from .simulator import (
    expectation as expectation,
)
from .simulator import (
    expectations as expectations,
)
from .simulator import (
    sample as sample,
)
from .simulator import (
    statevector as statevector,
)
from .sparse_pauli import SparsePauliPlan as SparsePauliPlan
from .sparse_pauli import SparsePauliResult as SparsePauliResult
from .templates import (
    IdentityBlockInitialization as IdentityBlockInitialization,
)
from .templates import (
    data_reuploading as data_reuploading,
)
from .templates import (
    hardware_efficient as hardware_efficient,
)
from .templates import (
    identity_block_ansatz as identity_block_ansatz,
)
from .tensor_contraction import TensorNetworkPlan as TensorNetworkPlan
from .tensor_train import TensorTrainMatrix as TensorTrainMatrix
from .tensor_train import TensorTrainPlan as TensorTrainPlan
from .truncated_mps import (
    MPSApproximation as MPSApproximation,
)
from .truncated_mps import (
    TruncatedMPSBackend as TruncatedMPSBackend,
)
from .verification import ConversionCheck as ConversionCheck

__version__: str
__all__: list[str]

from . import adaptive_amplitude as adaptive_amplitude
