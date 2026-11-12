"""Private matrix-product projections for hybrid model feature maps."""

import math
from dataclasses import dataclass, field

import numpy as np

from ..errors import ResourceError, ValidationError
from ..ir.core import integer
from ..sim.dense import _TensorOps
from .kernel import compute_operation, memory_budget


def _modes(values, name):
    if isinstance(values, (str, bytes, dict, set, frozenset)):
        raise ValidationError(f"{name} must be an ordered sequence of positive integers.")
    values = tuple(integer(x, name, 1) for x in values)
    if not values:
        raise ValidationError("Tensor-train mode lists cannot be empty.")
    return values


@dataclass(frozen=True)
class TensorTrainPlan:
    input_modes: tuple
    output_modes: tuple
    bond_dimension: object = 8
    in_features: int | None = None
    out_features: int | None = None
    max_memory_mb: float = 256
    max_work: int = 2_000_000
    ranks: tuple = field(init=False)
    core_shapes: tuple = field(init=False)
    num_parameters: int = field(init=False)
    direction: str = field(init=False)
    working_bytes_per_row: int = field(init=False)

    @classmethod
    def from_features(cls, in_features, out_features, **kwargs):
        a, b = integer(in_features, "in_features", 1), integer(out_features, "out_features", 1)
        bits_a, bits_b = (a - 1).bit_length(), (b - 1).bit_length()
        count = max(1, (max(bits_a, bits_b) + 1) // 2)

        def modes(bits):
            result = [1] * count
            for i in reversed(range(count)):
                local = min(2, bits)
                result[i] = 1 << local
                bits -= local
            return tuple(result)

        return cls(modes(bits_a), modes(bits_b), in_features=a, out_features=b, **kwargs)

    def __post_init__(self):
        ins, outs = (
            _modes(self.input_modes, "input_modes"),
            _modes(self.output_modes, "output_modes"),
        )
        if len(ins) != len(outs):
            raise ValidationError("Input/output mode counts must match.")
        budget = memory_budget(self.max_memory_mb)
        if budget <= 0:
            raise ValidationError("max_memory_mb must be positive.")
        integer(self.max_work, "max_work", 1)
        a, b = math.prod(ins), math.prod(outs)
        af = a if self.in_features is None else integer(self.in_features, "in_features", 1)
        bf = b if self.out_features is None else integer(self.out_features, "out_features", 1)
        if af > a or bf > b:
            raise ValidationError("Feature dimensions cannot exceed the physical modes.")
        if isinstance(self.bond_dimension, (int, np.integer)):
            cap = integer(self.bond_dimension, "bond_dimension", 1)
            caps = (cap,) * (len(ins) - 1)
        else:
            caps = (
                _modes(self.bond_dimension, "bond_dimension")
                if len(ins) > 1
                else tuple(self.bond_dimension)
            )
            if len(caps) != len(ins) - 1:
                raise ValidationError("Provide one bond dimension per internal cut.")
        pairs = tuple(i * o for i, o in zip(ins, outs, strict=True))
        ranks = (
            (1,)
            + tuple(
                min(c, math.prod(pairs[:k]), math.prod(pairs[k:])) for k, c in enumerate(caps, 1)
            )
            + (1,)
        )
        shapes = tuple(
            (ranks[k], i, o, ranks[k + 1]) for k, (i, o) in enumerate(zip(ins, outs, strict=True))
        )
        params = sum(math.prod(s) for s in shapes)
        if params > self.max_work or 16 * params >= budget * 2**20:
            raise ResourceError("Tensor-train parameters exceed the work/memory budget.")

        def peak(im, om, rs):
            remaining, prefix, maximum = math.prod(im), 1, math.prod(im)
            for k, (i, o) in enumerate(zip(im, om, strict=True)):
                remaining //= i
                prefix *= o
                maximum = max(maximum, prefix * rs[k + 1] * remaining)
            return maximum

        left = peak(ins, outs, ranks)
        right = peak(ins[::-1], outs[::-1], ranks[::-1])
        per_row = 8 * (a + b + 4 * min(left, right))
        if 16 * params + per_row > budget * 2**20:
            raise ResourceError("One tensor-train row exceeds max_memory_mb.")
        for name, value in (
            ("input_modes", ins),
            ("output_modes", outs),
            ("ranks", ranks),
            ("core_shapes", shapes),
            ("num_parameters", params),
            ("in_features", af),
            ("out_features", bf),
            ("direction", "left" if left <= right else "right"),
            ("working_bytes_per_row", per_row),
            ("max_memory_mb", budget),
        ):
            object.__setattr__(self, name, value)

    @property
    def batch_capacity(self):
        return max(
            1,
            int(memory_budget(self.max_memory_mb) * 2**20 - 16 * self.num_parameters)
            // self.working_bytes_per_row,
        )

    def to_dict(self):
        return {
            "representation": "matrix_product_operator",
            "input_modes": list(self.input_modes),
            "output_modes": list(self.output_modes),
            "ranks": list(self.ranks),
            "parameters": self.num_parameters,
            "dense_parameters": self.in_features * self.out_features,
            "direction": self.direction,
            "working_bytes_per_row": self.working_bytes_per_row,
            "batch_capacity": self.batch_capacity,
            "padding": "zero_input; crop_output",
        }

    @compute_operation
    def apply(self, cores, inputs, *, backend="numpy"):
        cores = tuple(cores)
        if len(cores) != len(self.core_shapes) or any(
            tuple(c.shape) != s for c, s in zip(cores, self.core_shapes, strict=True)
        ):
            raise ValidationError("MPO cores must match the plan's rank/mode shapes.")
        if inputs.ndim < 1 or inputs.shape[-1] != self.in_features:
            raise ValidationError("MPO input feature width differs from the plan.")
        if backend == "numpy" and (
            np.asarray(inputs).dtype.kind not in "fiu"
            or any(np.asarray(c).dtype.kind not in "fiu" for c in cores)
        ):
            raise ValidationError("Classical MPO inputs and cores must be real numeric arrays.")
        rows = math.prod(inputs.shape[:-1])
        if rows > self.batch_capacity:
            raise ResourceError("MPO contraction exceeds max_memory_mb; reduce the batch.")
        ops = _TensorOps(backend, cores[0] if backend != "numpy" else None)
        x = ops.array(inputs, real_value=True)
        local = tuple(ops.array(c, real_value=True) for c in cores)
        total = math.prod(self.input_modes)
        if total != self.in_features:
            zero = ops.xp.broadcast_to(
                ops.array(0, real_value=True), x.shape[:-1] + (total - self.in_features,)
            )
            x = (
                ops.xp.cat((x, zero), dim=-1)
                if backend == "torch"
                else ops.xp.concatenate((x, zero), axis=-1)
            )
        shape = x.shape[:-1]
        ins, outs = self.input_modes, self.output_modes
        if self.direction == "right":
            batch = len(shape)
            x = ops.transpose(
                x.reshape(shape + ins),
                tuple(range(batch)) + tuple(range(batch + len(ins) - 1, batch - 1, -1)),
            )
            ins, outs = ins[::-1], outs[::-1]
            local = tuple(ops.transpose(c, (3, 1, 2, 0)) for c in local[::-1])
        remaining, prefix, rank = math.prod(ins), 1, 1
        state = x.reshape((rows, 1, 1, remaining))
        for core, i, o in zip(local, ins, outs, strict=True):
            remaining //= i
            state = ops.einsum(
                "bpaij,aioz->bpozj", state.reshape((rows, prefix, rank, i, remaining)), core
            )
            prefix *= o
            rank = core.shape[-1]
            state = state.reshape((rows, prefix, rank, remaining))
        result = state.reshape(shape + outs)
        if self.direction == "right":
            batch = len(shape)
            result = ops.transpose(
                result, tuple(range(batch)) + tuple(range(batch + len(outs) - 1, batch - 1, -1))
            )
        return result.reshape(shape + (math.prod(self.output_modes),))[..., : self.out_features]
