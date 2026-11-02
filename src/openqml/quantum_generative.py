"""Born-rule distributions, teacher forcing and quantum-state diffusion."""

import math
from dataclasses import dataclass, field

import numpy as np

from .errors import ResourceError, ValidationError
from .ir import integer
from .layers import LayerSpec
from .observables import PauliString
from .parameters import real
from .quantum_cells import FeatureCircuitPlan
from .quantum_channels import QuantumChannelPlan, _checked, _real_tensor, _validate_density
from .simulator import _TensorOps


@dataclass(frozen=True)
class BornCircuitPlan:
    num_qubits: int = 2
    visible_qubits: int = 1
    depth: int = 2
    reupload: bool = True
    max_operations: int = 100_000
    max_memory_mb: float = 256
    spec: LayerSpec = field(init=False, repr=False)

    def __post_init__(self):
        for name in ("num_qubits", "visible_qubits", "depth", "max_operations"):
            object.__setattr__(self, name, integer(getattr(self, name), name, 1))
        if self.visible_qubits > self.num_qubits:
            raise ValidationError("Visible width exceeds the circuit width.")
        object.__setattr__(self, "max_memory_mb", real(self.max_memory_mb))
        if (
            self.max_memory_mb <= 0
            or self.visible_qubits > int(self.max_memory_mb * 2**20).bit_length()
            or 64 * (1 << self.visible_qubits) * self.num_qubits > self.max_memory_mb * 2**20
        ):
            raise ResourceError("Born readout descriptors exceed max_memory_mb.")
        base = FeatureCircuitPlan(
            self.num_qubits, self.depth, 1, self.reupload, self.max_operations
        ).to_spec()
        observables = tuple(
            PauliString(
                "".join(
                    "Z" if i & (1 << (self.visible_qubits - q - 1)) else "I"
                    for q in range(self.visible_qubits)
                )
                + "I" * (self.num_qubits - self.visible_qubits)
            )
            for i in range(1, 1 << self.visible_qubits)
        )
        object.__setattr__(
            self,
            "spec",
            LayerSpec(base.circuit, base.input_parameters, base.weight_parameters, observables),
        )

    def to_spec(self):
        return self.spec

    def to_dict(self):
        return {
            "architecture": "born_circuit",
            "num_qubits": self.num_qubits,
            "visible_qubits": self.visible_qubits,
            "parameters": len(self.spec.weight_parameters),
            "readouts": len(self.spec.observables),
        }


def born_probabilities(moments, *, backend="numpy", like=None, depolarization=0.001):
    ops = _TensorOps(backend, moments if like is None else like)
    moments = _real_tensor(moments, ops)
    classes = moments.shape[-1] + 1
    if classes < 2 or classes & (classes - 1):
        raise ValidationError("Born moments require a complete nonidentity Z basis.")
    noise = real(depolarization)
    if not 0 <= noise < 1:
        raise ValidationError("depolarization must lie in [0,1).")
    xp = ops.xp
    identity = xp.ones_like(moments[..., :1])
    value = (
        xp.cat((identity, moments), dim=-1)
        if backend == "torch"
        else xp.concatenate((identity, moments), axis=-1)
    )
    stride = 1
    while stride < classes:
        original = value.shape
        paired = value.reshape(value.shape[:-1] + (-1, 2, stride))
        a, b = paired[..., 0, :], paired[..., 1, :]
        value = (
            xp.stack((a + b, a - b), dim=-2)
            if backend == "torch"
            else xp.stack((a + b, a - b), axis=-2)
        ).reshape(original)
        stride *= 2
    return (1 - noise) * value / classes + noise / classes


def born_nll(probabilities, targets, *, backend="numpy", mask=None, reduction="mean"):
    ops = _TensorOps(backend, probabilities)
    p = _real_tensor(probabilities, ops)
    xp = ops.xp
    y = xp.as_tensor(targets, device=p.device) if backend == "torch" else xp.asarray(targets)
    if y.shape != p.shape[:-1] or "int" not in str(y.dtype) or "uint" in str(y.dtype):
        raise ValidationError("Targets must be signed integers with the probability batch shape.")
    valid = (
        xp.ones_like(y, dtype=xp.bool if backend == "torch" else bool)
        if mask is None
        else (
            xp.as_tensor(mask, device=p.device, dtype=xp.bool)
            if backend == "torch"
            else xp.asarray(mask, dtype=bool)
        )
    )
    if valid.shape != y.shape or reduction not in {"none", "sum", "mean"}:
        raise ValidationError("Invalid NLL mask or reduction.")
    in_range = ~valid | ((y >= 0) & (y < p.shape[-1]))
    marker = _checked(
        ops.array(0, real_value=True), in_range, ops, "Active target outside vocabulary."
    )
    safe = xp.where(valid & in_range, y, 0)
    safe = safe.to(xp.int64) if backend == "torch" else safe.astype("int32")
    gathered = (
        xp.take_along_dim(p, safe[..., None], dim=-1)[..., 0]
        if backend == "torch"
        else xp.take_along_axis(p, safe[..., None], axis=-1)[..., 0]
    )
    loss = -xp.log(xp.where(valid, gathered, 1))
    loss = loss + marker
    if reduction == "none":
        return loss
    total = loss.sum()
    if reduction == "sum":
        return total
    count = valid.sum()
    return total / (count.clamp_min(1) if backend == "torch" else xp.maximum(count, 1))


def shifted_tokens(
    tokens, classes, *, backend="numpy", like=None, mask=None, reset_mask=None, max_memory_mb=256
):
    classes = integer(classes, "classes", 2)
    ops = _TensorOps(backend, like)
    xp = ops.xp
    ids = xp.as_tensor(tokens, device=ops.device) if backend == "torch" else xp.asarray(tokens)
    if (
        ids.ndim != 2
        or "int" not in str(ids.dtype)
        or "uint" in str(ids.dtype)
        or min(ids.shape) < 1
    ):
        raise ValidationError("Tokens require nonempty signed integer IDs[B,T].")
    if 16 * math.prod(ids.shape) * classes > real(max_memory_mb) * 2**20:
        raise ResourceError("Teacher-forcing features exceed max_memory_mb.")

    def boolean(value, default):
        value = np.full(ids.shape, default) if value is None else value
        result = (
            xp.as_tensor(value, device=ops.device, dtype=xp.bool)
            if backend == "torch"
            else xp.asarray(value, dtype=bool)
        )
        if result.shape != ids.shape:
            raise ValidationError("Token masks must match IDs[B,T].")
        return result

    valid, reset = boolean(mask, True), boolean(reset_mask, False)
    checked = _checked(
        ops.array(ids, real_value=True),
        ~valid | ((ids >= 0) & (ids < classes)),
        ops,
        "Active token outside vocabulary.",
    )
    # Equality avoids unsafe indexing and works with arbitrary caller integer width.
    vocab = xp.arange(classes, device=ops.device) if backend == "torch" else xp.arange(classes)
    features = (
        (ids[..., None] == vocab).to(ops.real_dtype)
        if backend == "torch"
        else (ids[..., None] == vocab).astype(ops.real_dtype)
    )
    features = xp.where(valid[..., None], features, 0) + checked[..., None] * 0
    bos = xp.zeros_like(features[:, :1])
    shifted = (
        xp.cat((bos, features[:, :-1]), dim=1)
        if backend == "torch"
        else xp.concatenate((bos, features[:, :-1]), axis=1)
    )
    return xp.where((valid & ~reset)[..., None], shifted, 0)


@dataclass(frozen=True)
class QuantumDiffusionPlan:
    channel: QuantumChannelPlan = field(default_factory=QuantumChannelPlan)
    steps: int = 16
    schedule: str = "cosine"

    def __post_init__(self):
        if not isinstance(self.channel, QuantumChannelPlan) or self.schedule not in {
            "cosine",
            "linear",
        }:
            raise ValidationError(
                "Diffusion requires a QuantumChannelPlan and linear/cosine schedule."
            )
        object.__setattr__(self, "steps", integer(self.steps, "steps", 1))
        if 8 * (self.steps + 1) > self.channel.max_memory_mb * 2**20:
            raise ResourceError("Diffusion schedule exceeds max_memory_mb.")

    @property
    def retention(self):
        t = np.arange(self.steps + 1) / self.steps
        value = 1 - t if self.schedule == "linear" else np.cos(t * math.pi / 2) ** 2
        value[0], value[-1] = 1, 0
        return value

    def training_pair(self, rho, times, *, backend="numpy", like=None):
        ops = _TensorOps(backend, like)
        d = 1 << self.channel.memory_qubits
        rho = _validate_density(rho, d, ops)
        t = (
            ops.xp.as_tensor(times, device=ops.device)
            if backend == "torch"
            else ops.xp.asarray(times)
        )
        if t.shape != rho.shape[:-2] or "int" not in str(t.dtype) or "uint" in str(t.dtype):
            raise ValidationError(
                "Diffusion times must be signed integers with the density batch shape."
            )
        checked = _checked(
            ops.array(t, real_value=True),
            (t >= 1) & (t <= self.steps),
            ops,
            "Diffusion timestep outside [1,steps].",
        )
        t = (
            t.clamp(1, self.steps).to(ops.xp.int64)
            if backend == "torch"
            else ops.xp.clip(t, 1, self.steps).astype("int32")
        )
        retention = ops.array(self.retention, real_value=True)
        identity = ops.array(np.eye(d) / d)

        def at(index):
            alpha = retention[index][..., None, None]
            return alpha * rho + (1 - alpha) * identity + checked[..., None, None] * 0

        return at(t), at(t - 1)

    def loss(self, denoiser, rho, times, *, backend="numpy", like=None):
        ops = _TensorOps(backend, like)
        noisy, target = self.training_pair(rho, times, backend=backend, like=like)
        predicted = _validate_density(
            denoiser(noisy, ops.array(times, real_value=True) / self.steps),
            1 << self.channel.memory_qubits,
            ops,
        )
        return ops.real(
            ops.xp.einsum("...ab,...ab->...", predicted - target, ops.conj(predicted - target))
        )

    def sample(self, denoiser, batch_size, *, backend="numpy", like=None):
        batch = integer(batch_size, "batch_size", 1)
        self.channel._admit(batch)
        ops = _TensorOps(backend, like)
        d = 1 << self.channel.memory_qubits
        rho = ops.array(np.broadcast_to(np.eye(d) / d, (batch, d, d)).copy())
        for time in range(self.steps, 0, -1):
            rho = _validate_density(
                denoiser(rho, ops.array(np.full(batch, time / self.steps), real_value=True)), d, ops
            )
        return rho
