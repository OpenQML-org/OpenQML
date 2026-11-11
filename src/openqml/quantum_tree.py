"""Deterministic quantum tree tensor networks in a real Pauli-transfer basis.

Each node is a two-qubit unitary followed by a partial trace. The discarded
output's local SU(2) gauge is removed: 12 angles cover the SU(4) pooling family.
Bloch contractions retain coherence and unitary-induced entanglement without
postselection, SVD, complex state buffers or an exponentially large wavefunction.
"""

import hashlib
import json
import math
from dataclasses import dataclass, field

import numpy as np

from .compute import compute_operation, memory_budget
from .errors import ResourceError, ValidationError
from .ir import CircuitBuilder, integer
from .parameters import Parameter, real
from .simulator import _TensorOps


def _stack(values, ops):
    return (
        ops.xp.stack(tuple(values), dim=-1)
        if ops.backend == "torch"
        else ops.xp.stack(values, axis=-1)
    )


def _euler(vector, angles, ops):
    """Active RZ(a), RY(b), RZ(c), in circuit application order."""
    xp = ops.xp
    x, y, z = (vector[..., i] for i in range(3))
    a, b, c = (angles[..., i] for i in range(3))
    ca, sa, cb, sb, cc, sc = xp.cos(a), xp.sin(a), xp.cos(b), xp.sin(b), xp.cos(c), xp.sin(c)
    x, y = ca * x - sa * y, sa * x + ca * y
    x, z = cb * x + sb * z, -sb * x + cb * z
    return _stack((cc * x - sc * y, sc * x + cc * y, z), ops)


def _pool(left, right, angles, ops):
    left, right = _euler(left, angles[..., :3], ops), _euler(right, angles[..., 3:6], ops)
    x, y, z = (left[..., i] for i in range(3))
    u, v, w = (right[..., i] for i in range(3))
    a, b, c = (angles[..., i] for i in (6, 7, 8))
    ca, sa, cb, sb, cc, sc = (
        ops.xp.cos(a),
        ops.xp.sin(a),
        ops.xp.cos(b),
        ops.xp.sin(b),
        ops.xp.cos(c),
        ops.xp.sin(c),
    )
    # Exact Heisenberg transfer through exp[-i(a XX+b YY+c ZZ)/2].
    out = _stack(
        (
            x * cb * cc + z * v * sb * cc - y * w * cb * sc + u * sb * sc,
            y * ca * cc - z * u * sa * cc + x * w * ca * sc + v * sa * sc,
            z * ca * cb + y * u * sa * cb - x * v * ca * sb + w * sa * sb,
        ),
        ops,
    )
    return _euler(out, angles[..., 9:12], ops)


@dataclass(frozen=True, eq=False)
class QuantumTreePlan:
    num_qubits: int
    num_outputs: int = 4
    max_memory_mb: float = 256
    max_work: int = 2_000_000
    levels: tuple = field(init=False, repr=False)
    num_parameters: int = field(init=False)

    def __post_init__(self):
        n = integer(self.num_qubits, "num_qubits", 1)
        heads = integer(self.num_outputs, "num_outputs", 1)
        integer(self.max_work, "max_work", 1)
        budget = memory_budget(self.max_memory_mb)
        if budget <= 0:
            raise ValidationError("max_memory_mb must be positive.")
        parameters = heads * (12 * (n - 1) + 2)
        if parameters > self.max_work or 16 * parameters + 128 * n > budget * 2**20:
            raise ResourceError(
                "Quantum tree descriptors/parameters exceed the work or memory budget."
            )
        roots, levels = list(range(n)), []
        while len(roots) > 1:
            pairs = tuple(zip(roots[::2], roots[1::2], strict=False))
            levels.append(pairs)
            roots = [a for a, _ in pairs] + (roots[-1:] if len(roots) % 2 else [])
        object.__setattr__(self, "levels", tuple(levels))
        object.__setattr__(self, "num_parameters", parameters)
        object.__setattr__(self, "max_memory_mb", budget)

    @property
    def fingerprint(self):
        return hashlib.sha256(
            json.dumps((self.num_qubits, self.num_outputs, "ZYZ-KAK-left-trace-v1")).encode()
        ).hexdigest()

    def to_dict(self):
        return {
            "representation": "unitary_partial_trace_quantum_tree",
            "num_qubits": self.num_qubits,
            "num_outputs": self.num_outputs,
            "parameters": self.num_parameters,
            "depth": len(self.levels),
            "virtual_dimension": 2,
            "operator_bond_dimension": 4,
            "arithmetic": "real_Pauli_transfer; linear_in_sites_times_trees",
            "exactness": "no_postselection_or_intentional_truncation; roundoff_remains",
        }

    def initial_values(self, rng=None, *, scale=0.02):
        scale = real(scale)
        if scale < 0:
            raise ValidationError("scale must be nonnegative.")
        rng = np.random.default_rng() if rng is None else rng
        result = rng.normal(0, scale, self.num_parameters)
        count = 12 * self.num_outputs * (self.num_qubits - 1)
        if count:
            result[:count].reshape(self.num_outputs, self.num_qubits - 1, 12)[..., 6:9] += (
                math.pi / 4
            )
        result[count:] = math.pi / 4
        return result

    @compute_operation
    def evaluate(self, weights, inputs, *, backend="numpy"):
        if backend == "numpy":
            weights, inputs = np.asarray(weights), np.asarray(inputs)
            if (
                weights.dtype.kind not in "fiu"
                or inputs.dtype.kind not in "fiu"
                or not np.isfinite(weights).all()
                or not np.isfinite(inputs).all()
            ):
                raise ValidationError("Quantum tree values must be finite and real.")
        if (
            inputs.ndim < 1
            or inputs.shape[-1] != self.num_qubits
            or weights.ndim < 1
            or weights.shape[-1] != self.num_parameters
        ):
            raise ValidationError("Quantum tree input/weight feature widths differ from the plan.")
        try:
            shape = np.broadcast_shapes(weights.shape[:-1], inputs.shape[:-1])
        except ValueError as exc:
            raise ValidationError("Quantum tree batch dimensions must broadcast.") from exc
        if (
            96 * max(1, math.prod(shape)) * self.num_outputs * self.num_qubits
            + 16 * self.num_parameters
        ) > memory_budget(self.max_memory_mb) * 2**20:
            raise ResourceError(
                "Quantum tree native buffers exceed max_memory_mb; reduce the batch."
            )
        ops = _TensorOps(backend, weights if backend != "numpy" else None)
        w, x = ops.array(weights, real_value=True), ops.array(inputs, real_value=True)
        n, heads = self.num_qubits, self.num_outputs
        count = 12 * heads * (n - 1)
        nodes = w[..., :count].reshape(w.shape[:-1] + (heads, n - 1, 12))
        axes = w[..., count:].reshape(w.shape[:-1] + (heads, 2))
        initial = _stack((ops.xp.sin(x), x * 0, ops.xp.cos(x)), ops)
        state = initial[..., None, :, :]
        offset = 0
        for level in self.levels:
            pairs = len(level)
            pooled = _pool(
                state[..., : 2 * pairs : 2, :],
                state[..., 1 : 2 * pairs : 2, :],
                nodes[..., offset : offset + pairs, :],
                ops,
            )
            if state.shape[-2] % 2:
                tail = ops.xp.broadcast_to(state[..., -1:, :], pooled.shape[:-2] + (1, 3))
                state = (
                    ops.xp.cat((pooled, tail), dim=-2)
                    if backend == "torch"
                    else ops.xp.concatenate((pooled, tail), axis=-2)
                )
            else:
                state = pooled
            offset += pairs
        root = state[..., 0, :]
        polar, azimuth = axes[..., 0], axes[..., 1]
        value = (
            root[..., 0] * ops.xp.sin(polar) * ops.xp.cos(azimuth)
            + root[..., 1] * ops.xp.sin(polar) * ops.xp.sin(azimuth)
            + root[..., 2] * ops.xp.cos(polar)
        )
        return ops.xp.broadcast_to(value, shape + (heads,))

    def to_spec(self):
        from .layers import LayerSpec
        from .observables import PauliString

        n, heads = self.num_qubits, self.num_outputs
        if heads * heads * n > self.max_work:
            raise ResourceError("Exported forest observable descriptions exceed max_work.")
        operations = heads * (13 * n - 10)
        if 512 * operations + 2 * heads * heads * n > memory_budget(self.max_memory_mb) * 2**20:
            raise ResourceError("Exported quantum forest IR exceeds max_memory_mb.")
        inputs = tuple(f"tn_x{i:08d}" for i in range(n))
        weights = tuple(f"tn_w{i:08d}" for i in range(self.num_parameters))
        builder = CircuitBuilder(n * heads)
        for h in range(heads):
            for q, name in enumerate(inputs):
                builder.ry(Parameter(name), h * n + q)
            offset = 0
            for level in self.levels:
                for left, right in level:
                    base = (h * (n - 1) + offset) * 12
                    values = tuple(Parameter(weights[base + i]) for i in range(12))
                    a, b = h * n + left, h * n + right
                    for wire, start in ((a, 0), (b, 3)):
                        builder.rz(values[start], wire).ry(values[start + 1], wire).rz(
                            values[start + 2], wire
                        )
                    builder.rxx(values[6], a, b).ryy(values[7], a, b).rzz(values[8], a, b)
                    builder.rz(values[9], a).ry(values[10], a).rz(values[11], a)
                    offset += 1
            base = 12 * heads * (n - 1) + 2 * h
            builder.rz(-Parameter(weights[base + 1]), h * n).ry(-Parameter(weights[base]), h * n)
        observables = tuple(
            PauliString("I" * (h * n) + "Z" + "I" * (n * heads - h * n - 1)) for h in range(heads)
        )
        return LayerSpec(builder.build(), inputs, weights, observables)
