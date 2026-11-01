"""Quantum policy modules with ordinary PyTorch parameter ownership."""

import math

import torch
from torch import nn

from ..errors import ValidationError
from ..ir import integer
from ..rl import ActorCriticOutput, surrogate_fit_loss
from ._torch_hybrid import _input


class QuantumActorCritic(nn.Module):
    """Hybrid discrete/Gaussian actor and independent classical value baseline.

    The default universal circuit uses capability-aware QPU/GPU/CPU dispatch
    and batched parameter shifts. Supply a native IQPQuantumLayer to retain
    higher derivatives and structured large-width local readouts instead.
    """

    def __init__(
        self,
        observation_size,
        action_size,
        *,
        num_qubits=4,
        depth=2,
        hidden_features=64,
        continuous=False,
        quantum_layer=None,
        backend="auto",
        dtype=None,
        device=None,
        max_memory_mb=256,
    ):
        super().__init__()
        self.observation_size = integer(observation_size, "observation_size", 1)
        self.action_size = integer(action_size, "action_size", 1)
        self.num_qubits = integer(num_qubits, "num_qubits", 1)
        hidden = integer(hidden_features, "hidden_features", 1)
        if type(continuous) is not bool:
            raise ValidationError("continuous must be a boolean.")
        self.continuous = continuous
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.get_default_dtype() if dtype is None else dtype
        self.encoder = nn.Linear(observation_size, num_qubits, dtype=dtype, device=device)
        self.input_scale = nn.Parameter(torch.ones(num_qubits, dtype=dtype, device=device))
        if quantum_layer is None:
            from .torch import QuantumLayer

            quantum_layer = QuantumLayer.identity_blocks(
                num_qubits,
                depth,
                backend=backend,
                dtype=dtype,
                device=device,
                max_memory_mb=max_memory_mb,
                gradient_chunk_size=256,
            )
        if not isinstance(quantum_layer, nn.Module):
            raise ValidationError("quantum_layer must be a PyTorch module.")
        self.quantum = quantum_layer
        self.actor = nn.Linear(num_qubits, action_size, dtype=dtype, device=device)
        nn.init.normal_(self.actor.weight, std=0.01 / math.sqrt(num_qubits))
        nn.init.zeros_(self.actor.bias)
        self.critic = nn.Sequential(
            nn.Linear(observation_size, hidden, dtype=dtype, device=device),
            nn.Tanh(),
            nn.Linear(hidden, 1, dtype=dtype, device=device),
        )
        self.log_std = (
            nn.Parameter(torch.full((action_size,), -0.5, dtype=dtype, device=device))
            if continuous
            else None
        )

    def forward(self, observations):
        _input(observations, self.observation_size)
        controls = math.pi * torch.tanh(self.encoder(observations)) * self.input_scale
        features = self.quantum(controls)
        if features.shape != controls.shape:
            raise ValidationError("Quantum policies require one readout per declared quantum wire.")
        policy = self.actor(features.to(self.actor.weight.dtype))
        value = self.critic(observations)[..., 0]
        log_std = None if self.log_std is None else self.log_std.clamp(-20, 2).expand_as(policy)
        return ActorCriticOutput(policy, value, log_std)


class QuantumSurrogate(nn.Module):
    """Explicit measured/surrogate paths, with separate response-fit gradients.

    Surrogate updates introduce bias and never silently replace hardware calls.
    During actor updates surrogate parameters are detached while input gradients
    remain; fit_loss trains only the surrogate on detached replay pairs. All
    trainable circuit controls must be inputs to the surrogate for that update.
    """

    def __init__(self, quantum_layer, surrogate, *, binary_logits=False):
        super().__init__()
        if not isinstance(quantum_layer, nn.Module) or not isinstance(surrogate, nn.Module):
            raise ValidationError("Provide quantum and surrogate PyTorch modules.")
        if type(binary_logits) is not bool:
            raise ValidationError("binary_logits must be a boolean.")
        self.quantum, self.surrogate = quantum_layer, surrogate
        self.binary_logits = binary_logits

    def forward(self, controls, *, mode="quantum"):
        if mode == "quantum":
            return self.quantum(controls)
        if mode not in {"surrogate", "straight_through"}:
            raise ValidationError("mode must be quantum, surrogate or straight_through.")
        parameters = {k: v.detach() for k, v in self.surrogate.named_parameters()}
        prediction = torch.func.functional_call(self.surrogate, parameters, (controls,))
        if self.binary_logits:
            prediction = torch.sigmoid(prediction)
        if mode == "surrogate":
            return prediction
        with torch.no_grad():
            measured = self.quantum(controls)
        if measured.shape != prediction.shape:
            raise ValidationError("Measured and surrogate readouts must have identical shapes.")
        return measured.detach() + (prediction - prediction.detach())

    def fit_loss(self, controls, measurements):
        return surrogate_fit_loss(
            self.surrogate(controls.detach()), measurements, binary_logits=self.binary_logits
        )
