"""Trainable real controls for native continuous-time quantum layers."""

import torch
from torch import nn

from ...errors import ValidationError
from ...ir.core import integer
from ...ir.parameters import real
from ..lindblad import LindbladPlan


class LindbladLayer(nn.Module):
    def __init__(
        self, plan, input_features, *, dtype=None, device=None, rate_floor=0, checkpoint=True
    ):
        super().__init__()
        if not isinstance(plan, LindbladPlan):
            raise ValidationError("Provide a LindbladPlan.")
        features = integer(input_features, "input_features", 1)
        dtype = torch.get_default_dtype() if dtype is None else dtype
        if dtype not in {torch.float32, torch.float64}:
            raise ValidationError("Lindblad master weights require float32 or float64.")
        self.rate_floor = real(rate_floor)
        if self.rate_floor < 0 or type(checkpoint) is not bool:
            raise ValidationError("Use a nonnegative rate floor and boolean checkpoint option.")
        self.plan, self.checkpoint = plan, checkpoint
        device = ("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.encoder = nn.Linear(
            features, len(plan.hamiltonians) + plan.num_rates, dtype=dtype, device=device
        )

    def forward(self, controls, durations=1, *, state=None, mask=None, tolerance=None):
        if controls.ndim < 2 or controls.shape[-1] != self.encoder.in_features:
            raise ValidationError("Lindblad inputs require time and feature axes.")
        if controls.device != self.encoder.weight.device:
            raise ValidationError("Move controls and Lindblad parameters to the same device.")
        with torch.autocast(controls.device.type, enabled=False):
            projected = self.encoder(controls.to(self.encoder.weight.dtype))
            size = len(self.plan.hamiltonians)
            rates = torch.nn.functional.softplus(projected[..., size:]) + self.rate_floor
            return self.plan.evolve(
                projected[..., :size],
                rates,
                durations,
                state=state,
                backend="torch",
                mask=mask,
                checkpoint=self.checkpoint,
                tolerance=tolerance,
            )

    def get_extra_state(self):
        return self.plan.to_dict(), self.rate_floor, self.encoder.in_features

    def set_extra_state(self, value):
        if value != self.get_extra_state():
            raise ValidationError("Lindblad checkpoint architecture or rate floor differs.")
