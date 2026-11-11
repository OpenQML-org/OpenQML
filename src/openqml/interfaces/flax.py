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

from ._flax_channels import ChannelLayer as ChannelLayer
from ._flax_network import HybridNetwork as HybridNetwork
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
