"""Flax Linen quantum modules sharing standard parameter trees and Optax optimizers."""

from __future__ import annotations

try:
    import flax.linen as nn
    import jax.numpy as jnp
except ImportError as exc:
    from ..errors import MissingDependencyError

    raise MissingDependencyError("Install openqml[flax] for Linen modules.") from exc

from ._spec import LayerSpecMixin
from .jax import QuantumLayer as FunctionalLayer


class QuantumLayer(LayerSpecMixin, nn.Module):
    circuit: object
    input_parameters: tuple
    weight_parameters: tuple
    observables: tuple | None = None
    dtype: object = jnp.float32
    init_scale: float = 0.1
    max_qubits: int = 20
    max_memory_mb: float = 256
    backend: str = "jax"
    gradient_chunk_size: int = 64
    max_branches: int = 256
    optimize: bool = False
    optimization_options: object = None

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
    def __call__(self, inputs):
        weights = self.param(
            "weights",
            nn.initializers.normal(self.init_scale),
            (len(self.weight_parameters),),
            self.dtype,
        )
        layer = FunctionalLayer(
            self.circuit,
            self.input_parameters,
            self.weight_parameters,
            self.observables,
            max_qubits=self.max_qubits,
            max_memory_mb=self.max_memory_mb,
            backend=self.backend,
            gradient_chunk_size=self.gradient_chunk_size,
            max_branches=self.max_branches,
            optimize=self.optimize,
            optimization_options=self.optimization_options,
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
