"""Flax parameter ownership for native Lindblad control models."""

import flax.linen as nn
import jax
import jax.numpy as jnp

from ...errors import ValidationError
from ...ir.parameters import real
from ..lindblad import LindbladPlan


class LindbladLayer(nn.Module):
    plan: LindbladPlan
    rate_floor: float = 0
    dtype: object = jnp.float32
    checkpoint: bool = True

    @nn.compact
    def __call__(self, controls, durations=1, *, state=None, mask=None, tolerance=None):
        floor = real(self.rate_floor)
        if not isinstance(self.plan, LindbladPlan) or floor < 0 or controls.ndim < 2:
            raise ValidationError(
                "Lindblad layers require a plan, time/feature axes and nonnegative rate floor."
            )
        if jnp.dtype(self.dtype) not in {jnp.dtype("float32"), jnp.dtype("float64")}:
            raise ValidationError("Lindblad master weights require float32 or float64.")
        width = len(self.plan.hamiltonians) + self.plan.num_rates
        projected = nn.Dense(width, dtype=self.dtype, param_dtype=self.dtype)(controls)
        size = len(self.plan.hamiltonians)
        return self.plan.evolve(
            projected[..., :size],
            jax.nn.softplus(projected[..., size:]) + floor,
            durations,
            state=state,
            backend="jax",
            mask=mask,
            checkpoint=self.checkpoint,
            tolerance=tolerance,
        )
