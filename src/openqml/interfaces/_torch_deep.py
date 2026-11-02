"""End-to-end deep quantum residual models in ordinary PyTorch optimizers."""

import math

import torch
from torch import nn

from ..ir import integer


class QuantumDeepNetwork(nn.Module):
    """Classical encoder, persistent deep quantum state, and trainable readout.

    Outputs are logits/features, with arbitrary leading batch/token dimensions.
    The quantum core uses residual CPTP channels, not classical feature skips.
    """

    def __init__(
        self,
        input_size,
        output_size,
        *,
        num_qubits=4,
        depth=4,
        reupload=True,
        initial_mixing=0.1,
        dtype=None,
        device=None,
        backend=None,
        max_memory_mb=256,
    ):
        super().__init__()
        from .torch import QuantumLayer

        integer(input_size, "input_size", 1)
        integer(output_size, "output_size", 1)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = nn.Linear(input_size, num_qubits, dtype=dtype, device=device)
        self.quantum = QuantumLayer.deep(
            num_qubits,
            depth,
            reupload=reupload,
            initial_mixing=initial_mixing,
            dtype=dtype,
            device=device,
            backend=backend,
            max_memory_mb=max_memory_mb,
        )
        self.readout = nn.Linear(num_qubits, output_size, dtype=dtype, device=device)

    def forward(self, inputs):
        controls = math.pi * torch.tanh(self.encoder(inputs))
        return self.readout(self.quantum(controls))
