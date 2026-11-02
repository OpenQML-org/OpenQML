"""Trainable native CPTP channel layer with caller-owned density state."""

import torch
from torch import nn

from ..errors import ValidationError
from ..parameters import real
from ..quantum_channels import QuantumChannelPlan


class ChannelLayer(nn.Module):
    def __init__(self, plan=None, *, dtype=None, device=None, init_scale=0.1):
        super().__init__()
        self.plan = QuantumChannelPlan() if plan is None else plan
        if not isinstance(self.plan, QuantumChannelPlan):
            raise ValidationError("Provide a QuantumChannelPlan.")
        dtype = torch.get_default_dtype() if dtype is None else dtype
        if dtype not in {torch.float32, torch.float64} or real(init_scale) < 0:
            raise ValidationError("Channels require float32/64 and a nonnegative init_scale.")
        device = ("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.weights = nn.Parameter(
            torch.randn(len(self.plan.spec.weight_parameters), dtype=dtype, device=device)
            * init_scale
        )

    def forward(self, rho, controls):
        return self.plan.apply(self.weights, controls, rho, backend="torch")

    def time_controls(self, times):
        return self.plan.time_controls(times, backend="torch", like=self.weights)

    def get_extra_state(self):
        return self.plan.to_dict()

    def set_extra_state(self, value):
        if value != self.plan.to_dict():
            raise ValidationError("Channel checkpoint architecture does not match.")
