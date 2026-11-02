"""Flax Linen quantum modules sharing standard parameter trees and Optax optimizers."""

from __future__ import annotations

try:
    import flax as _flax
    import flax.linen as nn
    import jax.numpy as jnp
except ImportError as exc:
    from ..errors import MissingDependencyError

    raise MissingDependencyError(
        'Cannot load Flax. Run: python -m pip install "openqml[flax]".'
    ) from exc

from ._flax_network import HybridNetwork as HybridNetwork
from ._jax_ppo import PPOTrainer as PPOTrainer
from ._jax_training import accumulate_gradients as accumulate_gradients
from ._jax_training import make_train_step as make_train_step
from ._spec import LayerSpecMixin
from .jax import QuantumLayer as FunctionalLayer


def QuantumDeepNetwork(
    input_size,
    output_size,
    *,
    backend=None,
    active_depth=None,
    dtype=jnp.float32,
    **architecture_options,
):
    from ..architecture import HybridArchitecture

    return HybridArchitecture.deep(input_size, output_size, **architecture_options).build(
        "flax", backend=backend, active_depth=active_depth, dtype=dtype
    )


class QuantumActorCritic(nn.Module):
    """Linen hybrid actor and independent classical value baseline."""

    observation_size: int
    action_size: int
    num_qubits: int = 4
    depth: int = 2
    hidden_features: int = 64
    continuous: bool = False
    quantum_layer: object = None
    backend: object = "auto"
    dtype: object = jnp.float32

    @nn.compact
    def __call__(self, observations):
        import math

        from ..errors import ValidationError
        from ..ir import integer
        from ..rl import ActorCriticOutput

        for name in ("observation_size", "action_size", "num_qubits", "hidden_features"):
            integer(getattr(self, name), name, 1)
        if type(self.continuous) is not bool or observations.shape[-1] != self.observation_size:
            raise ValidationError("Invalid policy configuration/observation width.")
        controls = math.pi * jnp.tanh(
            nn.Dense(self.num_qubits, dtype=self.dtype, name="encoder")(observations)
        )
        controls *= self.param("input_scale", nn.initializers.ones, (self.num_qubits,), self.dtype)
        quantum = self.quantum_layer
        if quantum is None:
            quantum = QuantumLayer.identity_blocks(
                self.num_qubits, self.depth, backend=self.backend, dtype=self.dtype
            ).clone(parent=self, name="quantum")
        features = quantum(controls)
        width = getattr(quantum, "num_outputs", self.num_qubits)
        if features.shape != observations.shape[:-1] + (width,):
            raise ValidationError(
                "Quantum policy readouts differ from the configured feature width."
            )
        policy = nn.Dense(
            self.action_size,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(0.01 / math.sqrt(width)),
            name="actor",
        )(features)
        hidden = jnp.tanh(
            nn.Dense(self.hidden_features, dtype=self.dtype, name="critic_hidden")(observations)
        )
        value = nn.Dense(1, dtype=self.dtype, name="critic_value")(hidden)[..., 0]
        log_std = None
        if self.continuous:
            parameter = self.param(
                "log_std", nn.initializers.constant(-0.5), (self.action_size,), self.dtype
            )
            log_std = jnp.broadcast_to(jnp.clip(parameter, -20, 2), policy.shape)
        return ActorCriticOutput(policy, value, log_std)


class QuantumLayer(LayerSpecMixin, nn.Module):
    """Linen layer with optional adjoint gradients and unitary readout reduction.

    reduce_readout=True keeps source parameter shapes and checkpoint semantics.
    execution_method='pauli' uses bounded real arithmetic with native autodiff.
    execution_method='planned' selects and shares exact local query engines.
    """

    circuit: object
    input_parameters: tuple
    weight_parameters: tuple
    observables: tuple | None = None
    dtype: object = jnp.float32
    init_scale: float = 0.1
    max_qubits: int | None = None
    max_memory_mb: float = 256
    backend: object = None
    batch_chunk_size: int | None = None
    checkpoint_interval: int | None = None
    gradient_chunk_size: int = 64
    max_branches: int = 256
    optimize: bool = False
    optimization_options: object = None
    gradient_method: str = "auto"
    reduce_readout: bool = False
    execution_method: str = "auto"
    pauli_options: object = None
    planning_options: object = None
    tensor_options: object = None
    initialization_pairs: tuple = ()
    initialization_values: tuple = ()

    @property
    def num_outputs(self):
        return self.circuit.num_qubits if self.observables is None else len(self.observables)

    def _initialize_identity_blocks(self, pairs):
        return self.clone(initialization_pairs=pairs)

    def _initialize_quantum_template(self, pairs, constants):
        return self.clone(initialization_pairs=pairs, initialization_values=constants)

    @classmethod
    def from_checkpoint(cls, checkpoint, **kwargs):
        """Return (module, variables); optimizer state remains caller-owned."""
        from ..errors import ValidationError
        from ..layers import LayerCheckpoint

        if not isinstance(checkpoint, LayerCheckpoint):
            raise ValidationError("Expected a LayerCheckpoint.")
        if kwargs.get("dtype") is None:
            kwargs["dtype"] = jnp.dtype(checkpoint.dtype)
        module = cls.from_spec(checkpoint.spec, **kwargs)
        return module, module.load_checkpoint(checkpoint, allow_cast=True)

    def load_checkpoint(self, checkpoint, *, allow_cast=False):
        """Return a fresh variables tree after validating the full configuration."""
        from ..errors import ValidationError
        from ..layers import LayerCheckpoint
        from .jax import _checkpoint_weights

        if type(allow_cast) is not bool or not isinstance(checkpoint, LayerCheckpoint):
            raise ValidationError("Use a LayerCheckpoint and boolean allow_cast.")
        if not allow_cast and jnp.dtype(self.dtype).name != checkpoint.dtype:
            raise ValidationError("Checkpoint precision differs; set allow_cast=True explicitly.")
        return {"params": {"weights": _checkpoint_weights(checkpoint, self.to_spec(), self.dtype)}}

    @nn.compact
    def __call__(self, inputs, *, weights=None):
        """Use owned weights or differentiable weights supplied by a classical model."""
        if weights is None:

            def initialize(key, shape, dtype):
                import jax

                value = self.init_scale * jax.random.normal(key, shape, dtype)
                if self.initialization_pairs:
                    forward, reverse = zip(*self.initialization_pairs, strict=True)
                    value = value.at[jnp.asarray(reverse)].set(-value[jnp.asarray(forward)])
                for index, constant in self.initialization_values:
                    value = value.at[index].set(constant)
                return value

            weights = self.param(
                "weights",
                initialize,
                (len(self.weight_parameters),),
                self.dtype,
            )
        else:
            weights = jnp.asarray(weights, dtype=self.dtype)
        layer = FunctionalLayer(
            self.circuit,
            self.input_parameters,
            self.weight_parameters,
            self.observables,
            max_qubits=self.max_qubits,
            max_memory_mb=self.max_memory_mb,
            backend=self.backend,
            batch_chunk_size=self.batch_chunk_size,
            checkpoint_interval=self.checkpoint_interval,
            gradient_chunk_size=self.gradient_chunk_size,
            gradient_method=self.gradient_method,
            max_branches=self.max_branches,
            optimize=self.optimize,
            optimization_options=self.optimization_options,
            reduce_readout=self.reduce_readout,
            execution_method=self.execution_method,
            pauli_options=self.pauli_options,
            planning_options=self.planning_options,
            tensor_options=self.tensor_options,
        )
        return layer(weights, jnp.asarray(inputs, dtype=self.dtype))


class NativePennyLaneLayer(nn.Module):
    qnode: object
    weight_shape: tuple
    dtype: object = jnp.float32
    init_scale: float = 0.1

    @nn.compact
    def __call__(self, inputs):
        from .jax import NativePennyLaneLayer as FunctionalQNode

        weights = self.param(
            "weights", nn.initializers.normal(self.init_scale), self.weight_shape, self.dtype
        )
        layer = FunctionalQNode(self.qnode, {"weights": self.weight_shape})
        return layer({"weights": weights}, jnp.asarray(inputs, dtype=self.dtype))


class TreeQuantumLayer(nn.Module):
    num_qubits: int
    num_outputs: int = 4
    dtype: object = jnp.float32
    init_scale: float = 0.02
    batch_chunk_size: int | None = 256
    max_memory_mb: float = 256

    @nn.compact
    def __call__(self, inputs):
        from ._jax_tensor import TreeQuantumLayer as FunctionalTree

        layer = FunctionalTree(
            self.num_qubits,
            num_outputs=self.num_outputs,
            batch_chunk_size=self.batch_chunk_size,
            max_memory_mb=self.max_memory_mb,
        )
        weights = self.param(
            "weights",
            lambda key, shape, dtype: layer.init(key, scale=self.init_scale, dtype=dtype),
            (layer.plan.num_parameters,),
            self.dtype,
        )
        return layer(weights, jnp.asarray(inputs, self.dtype))


class TensorTrainLinear(nn.Module):
    in_features: int
    out_features: int
    bond_dimension: object = 8
    input_modes: object = None
    output_modes: object = None
    bias: bool = True
    dtype: object = jnp.float32
    batch_chunk_size: int | None = 256
    max_memory_mb: float = 256
    max_work: int = 2_000_000
    initial_error_bound: float | None = None
    initial_source_norm: float | None = None

    @classmethod
    def from_dense(cls, weight, bias=None, *, dtype=jnp.float32, **options):
        import numpy as np

        from ..errors import ValidationError
        from ..tensor_train import TensorTrainMatrix
        from ._jax_hybrid import _dtype

        _dtype(dtype)
        compressed = TensorTrainMatrix.from_dense(weight, **options)
        plan = compressed.plan
        model = cls(
            plan.in_features,
            plan.out_features,
            bond_dimension=plan.ranks[1:-1],
            input_modes=plan.input_modes,
            output_modes=plan.output_modes,
            bias=bias is not None,
            dtype=dtype,
            max_memory_mb=plan.max_memory_mb,
            max_work=plan.max_work,
            initial_error_bound=compressed.error_bound,
            initial_source_norm=compressed.source_norm,
        )
        parameters = {f"core_{i}": jnp.asarray(c, dtype) for i, c in enumerate(compressed.cores)}
        if bias is not None:
            value = np.asarray(bias)
            if value.shape != (plan.out_features,) or not np.isfinite(value).all():
                raise ValidationError("Dense bias has the wrong shape or nonfinite entries.")
            parameters["bias"] = jnp.asarray(value, dtype)
        return model, {"params": parameters}

    @nn.compact
    def __call__(self, inputs):
        import math

        from ._jax_tensor import TensorTrainLinear as FunctionalMPO

        layer = FunctionalMPO(
            self.in_features,
            self.out_features,
            bond_dimension=self.bond_dimension,
            input_modes=self.input_modes,
            output_modes=self.output_modes,
            bias=self.bias,
            batch_chunk_size=self.batch_chunk_size,
            max_memory_mb=self.max_memory_mb,
            max_work=self.max_work,
        )
        plan = layer.plan
        count = len(plan.core_shapes)
        factor = math.prod(plan.ranks[1:-1]) ** (-1 / (2 * count))
        parameters = {
            "cores": tuple(
                self.param(
                    f"core_{k}",
                    nn.initializers.normal(factor / math.sqrt(shape[1])),
                    shape,
                    self.dtype,
                )
                for k, shape in enumerate(plan.core_shapes)
            )
        }
        if self.bias:
            parameters["bias"] = self.param(
                "bias", nn.initializers.zeros, (self.out_features,), self.dtype
            )
        return layer(parameters, jnp.asarray(inputs, self.dtype))


class TensorNetworkAdapter(nn.Module):
    features: int
    num_qubits: int = 64
    num_outputs: int = 8
    bond_dimension: int = 8
    residual_scale: float = 0.01
    dtype: object = jnp.float32
    batch_chunk_size: int | None = 256
    max_memory_mb: float = 256

    @nn.compact
    def __call__(self, inputs):
        import math

        from ..errors import ValidationError

        if inputs.shape[-1] != self.features:
            raise ValidationError("Tensor adapter input feature width differs.")
        options = dict(
            bond_dimension=self.bond_dimension,
            dtype=self.dtype,
            batch_chunk_size=self.batch_chunk_size,
            max_memory_mb=self.max_memory_mb,
        )
        controls = math.pi * jnp.tanh(
            TensorTrainLinear(self.features, self.num_qubits, name="encoder", **options)(inputs)
        )
        quantum = TreeQuantumLayer(
            self.num_qubits,
            self.num_outputs,
            self.dtype,
            batch_chunk_size=self.batch_chunk_size,
            max_memory_mb=self.max_memory_mb,
            name="quantum",
        )(controls)
        delta = TensorTrainLinear(self.num_outputs, self.features, name="projection", **options)(
            quantum
        )
        scale = self.param(
            "residual_scale", nn.initializers.constant(self.residual_scale), (), self.dtype
        )
        return (inputs + scale * delta).astype(inputs.dtype)


class QuantumAdapter(nn.Module):
    """Residual adapter retaining every leading dimension, including token sequences."""

    num_qubits: int = 4
    depth: int = 2
    dtype: object = jnp.float32

    @nn.compact
    def __call__(self, inputs):
        from ..templates import hardware_efficient

        spec = hardware_efficient(self.num_qubits, self.depth)
        features = jnp.tanh(
            nn.Dense(self.num_qubits, dtype=self.dtype, param_dtype=self.dtype, name="encoder")(
                inputs
            )
        )
        quantum = QuantumLayer(
            spec.circuit,
            spec.input_parameters,
            spec.weight_parameters,
            dtype=self.dtype,
            name="quantum",
        )(features)
        delta = nn.Dense(
            inputs.shape[-1],
            dtype=self.dtype,
            param_dtype=self.dtype,
            kernel_init=nn.initializers.normal(0.01),
            name="projection",
        )(quantum)
        return inputs + delta


class IQPQuantumLayer(nn.Module):
    num_qubits: int
    edges: tuple | None = None
    dtype: object = jnp.float32
    init_scale: float = 0.1
    max_memory_mb: float = 256

    @nn.compact
    def __call__(self, inputs):
        from ._jax_hybrid import IQPQuantumLayer as FunctionalIQP

        layer = FunctionalIQP(self.num_qubits, edges=self.edges, max_memory_mb=self.max_memory_mb)
        weights = self.param(
            "weights", lambda key: layer.init(key, scale=self.init_scale, dtype=self.dtype)
        )
        return layer(weights, jnp.asarray(inputs, dtype=self.dtype))


class HyperQuantumAdapter(nn.Module):
    features: int
    num_qubits: int = 64
    rank: int = 16
    edges: tuple | None = None
    residual_scale: float = 0.01
    batch_chunk_size: int | None = 256
    max_memory_mb: float = 256
    dtype: object = jnp.float32

    @nn.compact
    def __call__(self, inputs):
        from ._jax_hybrid import HyperQuantumAdapter as FunctionalAdapter

        layer = FunctionalAdapter(
            self.features,
            num_qubits=self.num_qubits,
            rank=self.rank,
            edges=self.edges,
            residual_scale=self.residual_scale,
            batch_chunk_size=self.batch_chunk_size,
            max_memory_mb=self.max_memory_mb,
        )
        parameters = self.param("hybrid", lambda key: layer.init(key, dtype=self.dtype))
        return layer(parameters, jnp.asarray(inputs, dtype=self.dtype))


def __getattr__(name):
    """Expose native Flax APIs alongside the quantum Linen modules."""
    return getattr(_flax, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_flax)))
