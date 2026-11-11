"""Matrix-product operators with native contractions and certified TT-SVD import."""

import math
from dataclasses import dataclass, field

import numpy as np

from .compute import get_compute_kernel, memory_budget
from .errors import ResourceError, ValidationError
from .ir import integer
from .parameters import real
from .simulator import _TensorOps


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


@dataclass(frozen=True, eq=False)
class TensorTrainMatrix:
    plan: TensorTrainPlan
    cores: tuple
    error_bound: float = 0.0
    source_norm: float = 0.0

    def __post_init__(self):
        if not isinstance(self.plan, TensorTrainPlan):
            raise ValidationError("Provide a TensorTrainPlan.")
        raw = tuple(np.asarray(c) for c in self.cores)
        if any(c.dtype.kind not in "fiu" for c in raw):
            raise ValidationError("Classical MPO cores must be real numeric arrays.")
        cores = tuple(np.array(c, dtype=float, copy=True) for c in raw)
        if len(cores) != len(self.plan.core_shapes) or any(
            c.shape != s or not np.isfinite(c).all()
            for c, s in zip(cores, self.plan.core_shapes, strict=True)
        ):
            raise ValidationError("MPO cores must be finite real arrays matching the plan.")
        for c in cores:
            c.setflags(write=False)
        object.__setattr__(self, "cores", cores)
        if real(self.error_bound) < 0 or real(self.source_norm) < 0:
            raise ValidationError("Compression norms/bounds must be nonnegative.")

    @classmethod
    def from_dense(
        cls,
        weight,
        *,
        input_modes=None,
        output_modes=None,
        max_bond_dimension=None,
        relative_error=None,
        max_memory_mb=256,
        max_work=2_000_000,
    ):
        value = np.asarray(weight)
        if (
            value.ndim != 2
            or value.dtype.kind not in "fiu"
            or not np.isfinite(value).all()
            or min(value.shape) < 1
        ):
            raise ValidationError(
                "Dense weights must be a finite real (out_features,in_features) matrix."
            )
        out_features, in_features = value.shape
        if (input_modes is None) != (output_modes is None):
            raise ValidationError("Specify both input_modes and output_modes.")
        if input_modes is None:
            template = TensorTrainPlan.from_features(
                in_features,
                out_features,
                bond_dimension=1,
                max_memory_mb=max_memory_mb,
                max_work=max_work,
            )
            ins, outs = template.input_modes, template.output_modes
        else:
            ins, outs = _modes(input_modes, "input_modes"), _modes(output_modes, "output_modes")
            if (
                len(ins) != len(outs)
                or math.prod(ins) < in_features
                or math.prod(outs) < out_features
            ):
                raise ValidationError("Dense import modes do not fit the source matrix.")
        budget = memory_budget(max_memory_mb)
        if budget <= 0 or 64 * math.prod(ins) * math.prod(outs) > budget * 2**20:
            raise ResourceError("TT-SVD dense import/workspace exceeds max_memory_mb.")
        cap = (
            None
            if max_bond_dimension is None
            else integer(max_bond_dimension, "max_bond_dimension", 1)
        )
        tolerance = None if relative_error is None else real(relative_error)
        if tolerance is not None and not 0 <= tolerance <= 1:
            raise ValidationError("relative_error must be in [0,1].")
        from ._norms import vector_norm

        value = value.astype(float)
        scale = float(np.max(np.abs(value))) or 1.0
        value = value / scale
        unit_norm = float(vector_norm(value, np, "numpy"))
        norm = unit_norm * scale
        matrix = np.zeros((math.prod(outs), math.prod(ins)))
        matrix[:out_features, :in_features] = value
        d = len(ins)
        axes = tuple(j for i in range(d) for j in (d + i, i))
        tail = matrix.reshape(outs + ins).transpose(axes)
        cores, ranks, error_squared, rank = [], [], 0.0, 1
        allowance = 0.0 if tolerance is None else (tolerance * unit_norm) ** 2 / max(1, d - 1)
        for k, (i, o) in enumerate(zip(ins[:-1], outs[:-1], strict=True)):
            remaining = math.prod(ins[k + 1 :]) * math.prod(outs[k + 1 :])
            u, s, vh = get_compute_kernel().svd(
                tail.reshape(rank * i * o, remaining), full_matrices=False
            )
            selected = len(s)
            tails = np.concatenate((np.cumsum((s[::-1] ** 2))[::-1], [0.0]))
            if tolerance is not None:
                selected = next((r for r in range(1, len(s) + 1) if tails[r] <= allowance), len(s))
            if cap is not None:
                selected = min(selected, cap)
            error_squared += float(tails[selected])
            cores.append(u[:, :selected].reshape(rank, i, o, selected))
            tail = s[:selected, None] * vh[:selected]
            ranks.append(selected)
            rank = selected
        cores.append(tail.reshape(rank, ins[-1], outs[-1], 1) * scale)
        unit_bound = math.sqrt(error_squared)
        bound = unit_bound * scale
        if tolerance is not None and unit_bound > tolerance * unit_norm:
            raise ResourceError("The rank cap cannot satisfy the requested relative_error.")
        plan = TensorTrainPlan(ins, outs, tuple(ranks), in_features, out_features, budget, max_work)
        return cls(plan, tuple(cores), bound, norm)

    def __call__(self, inputs):
        return self.plan.apply(self.cores, np.asarray(inputs), backend="numpy")

    def to_dense(self, *, max_memory_mb=None):
        budget = self.plan.max_memory_mb if max_memory_mb is None else memory_budget(max_memory_mb)
        a, b = self.plan.in_features, self.plan.out_features
        if budget <= 0 or 24 * a * b > budget * 2**20:
            raise ResourceError("Dense MPO export exceeds max_memory_mb.")
        result = np.empty((b, a))
        chunk = min(
            self.plan.batch_capacity, max(1, int(budget * 2**20 - 8 * a * b) // max(8 * a, 1))
        )
        for start in range(0, a, chunk):
            size = min(chunk, a - start)
            basis = np.zeros((size, a))
            basis[np.arange(size), start + np.arange(size)] = 1
            result[:, start : start + size] = self(basis).T
        return result

    def to_dict(self):
        return {
            **self.plan.to_dict(),
            "frobenius_error_bound": self.error_bound,
            "source_frobenius_norm": self.source_norm,
            "bound_scope": "TT-SVD truncation only; excludes floating_point_roundoff",
        }
