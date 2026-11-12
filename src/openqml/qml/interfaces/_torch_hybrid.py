"""Native sparse IQP layers and input-conditioned residual hypernetworks."""

import math
from functools import wraps

import torch
from torch import nn

from ...errors import ValidationError
from ...ir.core import integer
from ...ir.parameters import real
from ...sim.dense import _TensorOps
from .._native_training import solve_metric
from ..iqp import IQPGraph


def _quantum_precision(function):
    @wraps(function)
    def wrapped(self, weights, inputs):
        if isinstance(inputs, torch.Tensor) and inputs.device.type in {"cpu", "cuda"}:
            # Classical projections can use AMP; quantum rotations/readouts
            # retain their explicitly selected float32/float64 arithmetic.
            with torch.autocast(inputs.device.type, enabled=False):
                return function(self, weights, inputs)
        return function(self, weights, inputs)

    return wrapped


class _IQPReadout(nn.Module):
    def __init__(self, graph):
        super().__init__()
        self.graph = graph
        for i, (_, edges, _) in enumerate(graph._groups):
            self.register_buffer(
                f"edges_{i}", torch.tensor(edges, dtype=torch.long), persistent=False
            )
        self.register_buffer(
            "order", torch.tensor(graph._order, dtype=torch.long), persistent=False
        )

    def forward(self, coordinates):
        value = (
            coordinates
            if coordinates.dtype in {torch.float32, torch.float64}
            else coordinates.float()
        )
        indices = tuple(getattr(self, f"edges_{i}") for i in range(len(self.graph._groups))) + (
            self.order,
        )
        return self.graph.evaluate(value, backend="torch", _indices=indices)

    def get_extra_state(self):
        return {"graph": self.graph.fingerprint, "coordinates": "alpha,omega,polar,azimuth"}

    def set_extra_state(self, value):
        if value != self.get_extra_state():
            raise ValidationError("IQP checkpoint graph/coordinate configuration differs.")


class IQPQuantumLayer(nn.Module):
    """Trainable (..., n) -> (..., n) exact two-body IQP expectation layer.

    Work follows n+edges rather than 2**n. These are local readouts, not samples
    or universal circuit simulation. All coordinates remain in native autodiff.
    """

    def __init__(
        self,
        num_qubits,
        *,
        edges=None,
        init_scale=0.1,
        dtype=None,
        device=None,
        max_memory_mb=256,
        max_work=2_000_000,
    ):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.graph = IQPGraph(num_qubits, edges, max_memory_mb, max_work)
        scale = real(init_scale)
        if scale < 0:
            raise ValidationError("init_scale must be nonnegative.")
        self.readout = _IQPReadout(self.graph).to(device=device)
        n, m = self.graph.num_qubits, len(self.graph.edges)
        values = torch.randn(self.graph.num_coordinates, dtype=dtype, device=device) * scale
        values[n + m : 2 * n + m] = math.pi / 4
        values[2 * n + m :] = math.pi / 4
        self.weights = nn.Parameter(values)

    def forward(self, inputs):
        return self.functional(self.weights, inputs)

    @_quantum_precision
    def functional(self, weights, inputs):
        _input(inputs, self.graph.num_qubits)
        _input(weights, self.graph.num_coordinates)
        if weights.device != inputs.device:
            raise ValidationError("IQP weights and inputs must share a device.")
        shape = torch.broadcast_shapes(weights.shape[:-1], inputs.shape[:-1])
        w = weights.expand(shape + (self.graph.num_coordinates,))
        x = inputs.expand(shape + (self.graph.num_qubits,)).to(w.dtype)
        n = self.graph.num_qubits
        coordinates = torch.cat((w[..., :n] + x, w[..., n:]), dim=-1)
        return self.readout(coordinates).to(inputs.dtype)

    def extra_repr(self):
        return f"qubits={self.graph.num_qubits}, edges={len(self.graph.edges)}, readouts=local"


class _HyperEncoder(nn.Module):
    def __init__(self, features, rank, coordinates, *, dtype, device):
        super().__init__()
        self.down = nn.Linear(features, rank, dtype=dtype, device=device)
        self.angles = nn.Linear(rank, coordinates, dtype=dtype, device=device)
        nn.init.normal_(self.angles.weight, std=0.1 / math.sqrt(rank))
        nn.init.zeros_(self.angles.bias)

    def forward(self, inputs):
        code = torch.tanh(self.down(inputs))
        return code, math.pi * torch.tanh(self.angles(code))


class HyperQuantumAdapter(nn.Module):
    """Low-rank, per-token circuit hypernetwork with an ordinary residual output.

    Inspired by HyperQ (arXiv:2609.24657), independently implemented without
    its pretrained model/data or reported task scores. Native CPU/CUDA tensors,
    higher derivatives and torch.func transforms retain their computation graph.
    """

    def __init__(
        self,
        features,
        *,
        num_qubits=64,
        rank=16,
        edges=None,
        residual_scale=0.01,
        batch_chunk_size=256,
        dtype=None,
        device=None,
        max_memory_mb=256,
    ):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.features = integer(features, "features", 1)
        rank = integer(rank, "rank", 1)
        self.graph = IQPGraph(num_qubits, edges, max_memory_mb)
        self.batch_chunk_size = (
            None if batch_chunk_size is None else integer(batch_chunk_size, "batch_chunk_size", 1)
        )
        self.encoder = _HyperEncoder(
            features, rank, num_qubits + len(self.graph.edges), dtype=dtype, device=device
        )
        self.axes = nn.Linear(rank, 2 * num_qubits, dtype=dtype, device=device)
        nn.init.normal_(self.axes.weight, std=0.1 / math.sqrt(rank))
        nn.init.constant_(self.axes.bias, math.pi / 4)
        self.projection = nn.Linear(num_qubits, features, bias=False, dtype=dtype, device=device)
        self.residual_scale = nn.Parameter(
            torch.tensor(real(residual_scale), dtype=dtype, device=device)
        )
        self.readout = _IQPReadout(self.graph).to(device=device)

    def forward(self, inputs):
        _input(inputs, self.features)
        if inputs.device != self.residual_scale.device:
            raise ValidationError("Adapter inputs and parameters must share a device.")
        shape = inputs.shape
        flat = inputs.reshape(-1, self.features)
        size = self.batch_chunk_size
        if size is not None and flat.shape[0] > size:
            return torch.cat(tuple(self._forward(x) for x in flat.split(size)), dim=0).reshape(
                shape
            )
        return self._forward(inputs)

    def _forward(self, inputs):
        code, angles = self.encoder(inputs.to(self.encoder.down.weight.dtype))
        coordinates = torch.cat((angles, self.axes(code)), dim=-1)
        quantum = self.readout(coordinates)
        delta = self.projection(quantum.to(self.projection.weight.dtype))
        return (inputs + self.residual_scale * delta).to(inputs.dtype)

    def metric_product(self, inputs, tangent):
        """Exact prepared-state pullback metric on encoder parameter coordinates.

        IQP generator covariances are I/4. JVP/VJP applies J.T J/(4*batch),
        without a dense metric or quantum state. Measurement/projection weights
        are intentionally outside this prepared-state geometry.
        """
        _input(inputs, self.features)
        parameters = dict(self.encoder.named_parameters())
        _vector(parameters, tangent)
        rows = math.prod(inputs.shape[:-1])
        if not rows:
            raise ValidationError("Metric products need a nonempty input batch.")
        dtype = self.encoder.down.weight.dtype
        if dtype not in {torch.float32, torch.float64}:
            raise ValidationError("Quantum geometry requires float32/float64 parameters.")
        from contextlib import nullcontext

        context = (
            torch.autocast(inputs.device.type, enabled=False)
            if inputs.device.type in {"cpu", "cuda"}
            else nullcontext()
        )
        with context:

            def coordinates(p):
                return torch.func.functional_call(self.encoder, p, (inputs.to(dtype),))[1]

            _, tangent_output = torch.func.jvp(coordinates, (parameters,), (tangent,))
            _, pullback = torch.func.vjp(coordinates, parameters)
            return {k: v / (4 * rows) for k, v in pullback(tangent_output)[0].items()}

    def natural_gradient(self, inputs, gradient=None, **options):
        parameters = dict(self.encoder.named_parameters())
        gradient = {k: p.grad for k, p in parameters.items()} if gradient is None else gradient
        _vector(parameters, gradient)
        return solve_metric(
            lambda v: self.metric_product(inputs, v),
            gradient,
            _TensorOps("torch", self.encoder.down.weight),
            **options,
        )


def _input(value, features):
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim < 1
        or value.shape[-1] != features
        or value.dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}
    ):
        raise ValidationError(f"Expected a real floating tensor with shape (..., {features}).")


def _vector(parameters, vector):
    if not isinstance(vector, dict) or set(vector) != set(parameters):
        raise ValidationError("Encoder vectors must match encoder parameter names.")
    if any(
        not isinstance(vector[k], torch.Tensor)
        or vector[k].shape != p.shape
        or vector[k].dtype != p.dtype
        or vector[k].device != p.device
        for k, p in parameters.items()
    ):
        raise ValidationError("Encoder vectors must match parameter shapes, dtypes and devices.")
