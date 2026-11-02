"""Linen parameter ownership for native CPTP channel layers."""

import flax.linen as nn
import jax.numpy as jnp

from ..errors import ValidationError
from ..parameters import real
from ..quantum_channels import QuantumChannelPlan


class ChannelLayer(nn.Module):
    plan: QuantumChannelPlan = QuantumChannelPlan()
    dtype: object = jnp.float32
    init_scale: float = 0.1

    @nn.compact
    def __call__(self, rho, controls):
        if (
            not isinstance(self.plan, QuantumChannelPlan)
            or real(self.init_scale) < 0
            or jnp.dtype(self.dtype) not in {jnp.dtype("float32"), jnp.dtype("float64")}
        ):
            raise ValidationError("Channels require a plan, float32/64 and nonnegative init_scale.")
        weights = self.param(
            "weights",
            nn.initializers.normal(self.init_scale),
            (len(self.plan.spec.weight_parameters),),
            self.dtype,
        )
        return self.plan.apply(weights, controls, rho, backend="jax")
