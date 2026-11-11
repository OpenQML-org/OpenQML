"""Full-rank trainable POVMs with Cholesky whitening and native derivatives."""

import math
from dataclasses import dataclass, replace

import numpy as np

from .compute import memory_budget
from .errors import ResourceError, ValidationError
from .ir import integer
from .parameters import real
from .quantum_channels import _real_tensor, _validate_density
from .simulator import _TensorOps


@dataclass(frozen=True)
class _PreparedPOVM:
    weights: object
    inverse: object
    alpha: object
    assignment: object
    ops: object
    depolarization: float
    max_memory_mb: float

    def to(self, target):
        if self.ops.backend != "torch":
            raise ValidationError("Only Torch prepared frames support device transfer.")
        weights = self.weights.to(target)
        return replace(
            self,
            weights=weights,
            inverse=self.inverse.to(target),
            alpha=self.alpha.to(target),
            assignment=self.assignment.to(target),
            ops=_TensorOps("torch", weights.real),
        )

    def probabilities(self, rho):
        ops = self.ops
        dimension, classes = self.inverse.shape[-1], self.weights.shape[0]
        if rho.shape[-2:] != (dimension, dimension):
            raise ValidationError("POVM density shape does not match its frame.")
        if (
            32 * math.prod(rho.shape[:-2]) * classes * dimension**2
            > memory_budget(self.max_memory_mb) * 2**20
        ):
            raise ResourceError("POVM batch workspace exceeds max_memory_mb.")
        value = ops.einsum("...ab,kbc,kac->...k", rho, self.weights, ops.conj(self.weights))
        anchors = ops.einsum("...ab,bj,aj->...j", rho, self.inverse, ops.conj(self.inverse))
        value = ops.real(value + self.alpha * ops.matmul(anchors, self.assignment))
        return (1 - self.depolarization) * value + self.depolarization / classes


@dataclass(frozen=True)
class QuantumPOVMPlan:
    memory_qubits: int = 1
    outcomes: int = 2
    regularization: float = 0.01
    depolarization: float = 0.001
    max_memory_mb: float = 256

    def __post_init__(self):
        for name, minimum in (("memory_qubits", 1), ("outcomes", 2)):
            object.__setattr__(self, name, integer(getattr(self, name), name, minimum))
        for name in ("regularization", "depolarization", "max_memory_mb"):
            object.__setattr__(self, name, real(getattr(self, name)))
        if self.regularization <= 0 or not 0 <= self.depolarization < 1 or self.max_memory_mb <= 0:
            raise ValidationError("Invalid POVM regularization, depolarization or memory budget.")
        if (
            self.memory_qubits > int(memory_budget(self.max_memory_mb) * 2**20).bit_length()
            or 256 * self.outcomes * 4**self.memory_qubits
            > memory_budget(self.max_memory_mb) * 2**20
        ):
            raise ResourceError("POVM frame exceeds max_memory_mb.")

    @property
    def parameter_shape(self):
        d = 1 << self.memory_qubits
        return (self.outcomes, d, d, 2)

    def prepare(self, frame, *, backend="numpy", like=None):
        ops = _TensorOps(backend, frame if like is None else like)
        frame = _real_tensor(frame, ops)
        if frame.shape != self.parameter_shape:
            raise ValidationError("POVM frame must have shape [outcomes,d,d,2].")
        a = frame[..., 0] + 1j * frame[..., 1]
        gram = ops.einsum("kac,kbc->ab", a, ops.conj(a))
        d = gram.shape[-1]
        alpha = self.regularization * (1 + ops.real(ops.einsum("ii->", gram)) / d)
        identity = ops.array(np.eye(d))
        inverse = ops.solve(ops.xp.linalg.cholesky(gram + alpha * identity), identity)
        weights = ops.matmul(inverse, a)
        assignment = np.zeros((d, self.outcomes))
        assignment[np.arange(d), np.arange(d) % self.outcomes] = 1
        return _PreparedPOVM(
            weights,
            inverse,
            alpha,
            ops.array(assignment),
            ops,
            self.depolarization,
            self.max_memory_mb,
        )

    def probabilities(self, frame, rho, *, backend="numpy"):
        prepared = self.prepare(frame, backend=backend)
        return prepared.probabilities(_validate_density(rho, 1 << self.memory_qubits, prepared.ops))
