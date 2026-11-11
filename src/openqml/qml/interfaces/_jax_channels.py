"""Functional native JAX channel layer."""

import jax
import jax.numpy as jnp

from ...errors import ValidationError
from ...ir.parameters import real
from ..channels import QuantumChannelPlan


class ChannelLayer:
    def __init__(self, plan=None, *, init_scale=0.1):
        self.plan = QuantumChannelPlan() if plan is None else plan
        if not isinstance(self.plan, QuantumChannelPlan) or real(init_scale) < 0:
            raise ValidationError("Provide a QuantumChannelPlan and nonnegative init_scale.")
        self.init_scale = init_scale

    def init(self, key, *, dtype=jnp.float32):
        if jnp.dtype(dtype) not in {jnp.dtype("float32"), jnp.dtype("float64")}:
            raise ValidationError("Channels require float32 or float64.")
        return (
            jax.random.normal(key, (len(self.plan.spec.weight_parameters),), dtype=dtype)
            * self.init_scale
        )

    def __call__(self, weights, rho, controls):
        return self.plan.apply(weights, controls, rho, backend="jax")

    def time_controls(self, times, *, like=None):
        return self.plan.time_controls(times, backend="jax", like=like)
