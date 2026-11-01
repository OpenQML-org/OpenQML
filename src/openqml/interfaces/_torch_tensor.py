"""Native trainable quantum trees, MPO projections and residual tensor adapters."""

import math

import numpy as np
import torch
from torch import nn

from ..errors import ValidationError
from ..ir import integer
from ..parameters import real
from ..quantum_tree import QuantumTreePlan
from ..tensor_train import TensorTrainMatrix, TensorTrainPlan
from ._spec import LayerSpecMixin
from ._torch_hybrid import _input, _quantum_precision


def _device(device):
    return ("cuda" if torch.cuda.is_available() else "cpu") if device is None else device


class TreeQuantumLayer(nn.Module):
    """Exact unitary/partial-trace quantum forest; (...,n) -> (...,num_outputs)."""

    def __init__(
        self,
        num_qubits,
        *,
        num_outputs=4,
        init_scale=0.02,
        dtype=None,
        device=None,
        batch_chunk_size=256,
        max_memory_mb=256,
        max_work=2_000_000,
    ):
        super().__init__()
        self.plan = QuantumTreePlan(num_qubits, num_outputs, max_memory_mb, max_work)
        self.num_outputs = num_outputs
        dtype = torch.get_default_dtype() if dtype is None else dtype
        if dtype not in {torch.float32, torch.float64}:
            raise ValidationError("Tree weights require float32/float64.")
        scale = real(init_scale)
        if scale < 0:
            raise ValidationError("init_scale must be nonnegative.")
        values = torch.randn(self.plan.num_parameters, dtype=dtype, device=_device(device)) * scale
        count = 12 * num_outputs * (num_qubits - 1)
        if count:
            values[:count].reshape(num_outputs, num_qubits - 1, 12)[..., 6:9] += math.pi / 4
        values[count:] = math.pi / 4
        self.weights = nn.Parameter(values)
        self.batch_chunk_size = (
            None if batch_chunk_size is None else integer(batch_chunk_size, "batch_chunk_size", 1)
        )

    @_quantum_precision
    def functional(self, weights, inputs):
        _input(inputs, self.plan.num_qubits)
        _input(weights, self.plan.num_parameters)
        if weights.device != inputs.device:
            raise ValidationError("Tree inputs and weights must share a device.")
        value = self.plan.evaluate(weights, inputs, backend="torch")
        return value.to(inputs.dtype)

    def forward(self, inputs):
        _input(inputs, self.plan.num_qubits)
        shape = inputs.shape[:-1]
        x = inputs.reshape(-1, self.plan.num_qubits)
        size = self.batch_chunk_size
        if size is not None and x.shape[0] > size:
            return torch.cat(
                tuple(self.functional(self.weights, y) for y in x.split(size)), dim=0
            ).reshape(shape + (self.num_outputs,))
        return self.functional(self.weights, inputs)

    def to_spec(self):
        return self.plan.to_spec()

    def checkpoint(self):
        return LayerSpecMixin.checkpoint(self)

    def get_extra_state(self):
        return {"quantum_tree": self.plan.fingerprint}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValidationError("Quantum tree checkpoint topology differs.")


class TensorTrainLinear(nn.Module):
    """An MPO linear map, without a dense weight matrix in forward/backward."""

    def __init__(
        self,
        in_features,
        out_features,
        *,
        bond_dimension=8,
        input_modes=None,
        output_modes=None,
        bias=True,
        dtype=None,
        device=None,
        batch_chunk_size=256,
        max_memory_mb=256,
        max_work=2_000_000,
    ):
        super().__init__()
        if (input_modes is None) != (output_modes is None) or type(bias) is not bool:
            raise ValidationError("Specify both mode lists and a boolean bias.")
        options = dict(
            bond_dimension=bond_dimension, max_memory_mb=max_memory_mb, max_work=max_work
        )
        self.plan = (
            TensorTrainPlan.from_features(in_features, out_features, **options)
            if input_modes is None
            else TensorTrainPlan(
                input_modes,
                output_modes,
                in_features=in_features,
                out_features=out_features,
                **options,
            )
        )
        self.in_features, self.out_features = self.plan.in_features, self.plan.out_features
        self.batch_chunk_size = (
            min(self.plan.batch_capacity, integer(batch_chunk_size, "batch_chunk_size", 1))
            if batch_chunk_size is not None
            else self.plan.batch_capacity
        )
        dtype = torch.get_default_dtype() if dtype is None else dtype
        if dtype not in {torch.float32, torch.float64}:
            raise ValidationError("MPO weights require float32/float64.")
        device = _device(device)
        # Var(W_ij) ~= 1/prod(input_modes), independently of bond rank.
        count = len(self.plan.core_shapes)
        bond_factor = math.prod(self.plan.ranks[1:-1]) ** (-1 / (2 * count))
        self.cores = nn.ParameterList(
            nn.Parameter(torch.randn(s, dtype=dtype, device=device) * bond_factor / math.sqrt(s[1]))
            for s in self.plan.core_shapes
        )
        self.bias = (
            nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device)) if bias else None
        )
        self.initial_compression = None

    def forward(self, inputs):
        _input(inputs, self.in_features)
        if inputs.device != self.cores[0].device:
            raise ValidationError("MPO inputs and cores must share a device.")
        shape = inputs.shape[:-1]
        flat = inputs.reshape(-1, self.in_features)
        size = self.batch_chunk_size
        if flat.shape[0] > size:
            value = torch.cat(
                tuple(self.plan.apply(self.cores, x, backend="torch") for x in flat.split(size)),
                dim=0,
            ).reshape(shape + (self.out_features,))
        else:
            value = self.plan.apply(self.cores, inputs, backend="torch")
        return value if self.bias is None else value + self.bias

    @classmethod
    def from_dense(cls, weight, bias=None, *, dtype=None, device=None, **options):
        data = weight.detach().cpu().numpy() if isinstance(weight, torch.Tensor) else weight
        compressed = TensorTrainMatrix.from_dense(data, **options)
        plan = compressed.plan
        model = cls(
            plan.in_features,
            plan.out_features,
            bond_dimension=plan.ranks[1:-1],
            input_modes=plan.input_modes,
            output_modes=plan.output_modes,
            bias=bias is not None,
            dtype=dtype,
            device=device,
            max_memory_mb=plan.max_memory_mb,
            max_work=plan.max_work,
        )
        with torch.no_grad():
            for p, c in zip(model.cores, compressed.cores, strict=True):
                p.copy_(torch.as_tensor(np.array(c), dtype=p.dtype, device=p.device))
            if bias is not None:
                b = torch.as_tensor(bias, dtype=model.bias.dtype, device=model.bias.device)
                if b.shape != model.bias.shape or not torch.isfinite(b).all():
                    raise ValidationError("Dense bias has the wrong shape or nonfinite entries.")
                model.bias.copy_(b)
        model.initial_compression = compressed.to_dict()
        return model

    def get_extra_state(self):
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "input_modes": self.plan.input_modes,
            "output_modes": self.plan.output_modes,
            "ranks": self.plan.ranks,
        }

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValidationError("MPO checkpoint modes/ranks differ.")


class TensorNetworkAdapter(nn.Module):
    """MPO -> coherent quantum tree -> MPO residual, preserving (...,features)."""

    def __init__(
        self,
        features,
        *,
        num_qubits=64,
        num_outputs=8,
        bond_dimension=8,
        residual_scale=0.01,
        dtype=None,
        device=None,
        batch_chunk_size=256,
        max_memory_mb=256,
    ):
        super().__init__()
        device = _device(device)
        self.features = integer(features, "features", 1)
        options = dict(
            bond_dimension=bond_dimension,
            dtype=dtype,
            device=device,
            batch_chunk_size=batch_chunk_size,
            max_memory_mb=max_memory_mb,
        )
        self.encoder = TensorTrainLinear(features, num_qubits, **options)
        self.quantum = TreeQuantumLayer(
            num_qubits,
            num_outputs=num_outputs,
            dtype=dtype,
            device=device,
            batch_chunk_size=batch_chunk_size,
            max_memory_mb=max_memory_mb,
        )
        self.projection = TensorTrainLinear(num_outputs, features, **options)
        self.residual_scale = nn.Parameter(
            torch.tensor(real(residual_scale), dtype=self.encoder.cores[0].dtype, device=device)
        )

    def forward(self, inputs):
        _input(inputs, self.features)
        controls = math.pi * torch.tanh(self.encoder(inputs))
        return (inputs + self.residual_scale * self.projection(self.quantum(controls))).to(
            inputs.dtype
        )
