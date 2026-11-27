"""Exact local readouts of two-body IQP circuit hypernetworks.

Independent implementation of the commuting family in arXiv:2609.24657,
Eq. (5). One-site expectation values use exact local trigonometric contractions
with work proportional to the circuit's vertices and edges.
"""

import hashlib
import json
import math
from collections.abc import Mapping, Set
from dataclasses import dataclass, field

import numpy as np

from ..compute.kernel import memory_budget
from ..errors import ResourceError, ValidationError
from ..ir.core import CircuitBuilder, integer
from ..ir.parameters import Parameter
from ..sim.dense import _TensorOps


@dataclass(frozen=True)
class IQPGraph:
    num_qubits: int
    edges: tuple | None = None
    max_memory_mb: float = 256
    max_work: int = 2_000_000
    _groups: tuple = field(init=False, repr=False, compare=False)
    _order: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        n = integer(self.num_qubits, "num_qubits", 1)
        work = integer(self.max_work, "max_work", 1)
        budget = memory_budget(self.max_memory_mb)
        if budget <= 0:
            raise ValidationError("max_memory_mb must be positive.")
        capacity = int(budget * 2**20)
        if n > work or 256 * n > capacity:
            raise ResourceError("IQP graph descriptors exceed work/memory budget.")
        if self.edges is None:
            stride = 3 if n >= 7 else 2
            offsets = (1, stride) if n >= 5 else (1,) if n > 1 else ()
            count = 1 if n == 2 else n * len(offsets)
            if n + 2 * count > work or 256 * n + 512 * count > capacity:
                raise ResourceError("Default IQP graph exceeds work/memory budget.")
            edges = sorted(
                {tuple(sorted((q, (q + step) % n))) for q in range(n) for step in offsets}
            )
        else:
            if isinstance(self.edges, (str, bytes, Mapping, Set)):
                raise ValidationError("IQP edges require an ordered iterable.")
            edges, seen = [], set()
            try:
                for edge in self.edges:
                    if len(edges) >= work or 256 * n + 512 * (len(edges) + 1) > capacity:
                        raise ResourceError("IQP edges exceed work/memory budget.")
                    a, b = edge
                    a, b = integer(a, "edge vertex"), integer(b, "edge vertex")
                    pair = tuple(sorted((a, b)))
                    if a == b or max(a, b) >= n or pair in seen:
                        raise ValidationError("IQP edges must be unique, loop-free and in range.")
                    seen.add(pair)
                    edges.append(pair)
            except (TypeError, ValueError) as exc:
                raise ValidationError("IQP edges require pairs of wire indices.") from exc
        if n + 2 * len(edges) > work or 256 * n + 512 * len(edges) > capacity:
            raise ResourceError("IQP graph exceeds work/memory budget.")
        incident = [[] for _ in range(n)]
        for index, (a, b) in enumerate(edges):
            incident[a].append(index)
            incident[b].append(index)
        buckets = {}
        for q, entries in enumerate(incident):
            buckets.setdefault(len(entries), []).append((q, tuple(entries)))
        groups, permutation = [], []
        for degree, entries in sorted(buckets.items()):
            nodes = tuple(q for q, _ in entries)
            groups.append((nodes, tuple(e for _, e in entries), degree))
            permutation.extend(nodes)
        inverse = [0] * n
        for i, q in enumerate(permutation):
            inverse[q] = i
        object.__setattr__(self, "num_qubits", n)
        object.__setattr__(self, "edges", tuple(edges))
        object.__setattr__(self, "max_memory_mb", budget)
        object.__setattr__(self, "_groups", tuple(groups))
        object.__setattr__(self, "_order", tuple(inverse))

    @property
    def num_coordinates(self):
        return 3 * self.num_qubits + len(self.edges)

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps((self.num_qubits, self.edges)).encode()).hexdigest()

    def to_dict(self):
        degrees = [0] * self.num_qubits
        for a, b in self.edges:
            degrees[a] += 1
            degrees[b] += 1
        return {
            "family": "H-RZ-RZZ-H; local spherical readout axes",
            "num_qubits": self.num_qubits,
            "edges": [list(e) for e in self.edges],
            "max_degree": max(degrees),
            "fingerprint": self.fingerprint,
            "coordinate_order": "alpha, omega, polar, azimuth",
            "scope": "local_expectations_only; no_shot_or_truncation_approximation",
            "reference_gradient_variance": [math.ldexp(1 / 3, -d) for d in degrees],
            "variance_assumptions": "independent_uniform_angles_and_uniform_S2_axes; not_task_loss",
        }

    def _coordinates(self, coordinates, backend, like):
        if like is None and backend != "numpy":
            like = coordinates
        ops = _TensorOps(backend, like)
        if backend == "numpy":
            raw = np.asarray(coordinates)
            if raw.dtype.kind not in "iuf" or not np.isfinite(raw).all():
                raise ValidationError("IQP coordinates require finite real arrays.")
        value = ops.array(coordinates, real_value=True)
        if value.ndim < 1 or value.shape[-1] != self.num_coordinates:
            raise ValidationError(f"Expected (..., {self.num_coordinates}) IQP coordinates.")
        rows = math.prod(value.shape[:-1])
        if (
            64 * max(1, rows) * (self.num_coordinates + self.num_qubits)
            > memory_budget(self.max_memory_mb) * 2**20
        ):
            raise ResourceError("IQP batched coordinate/readout workspace exceeds max_memory_mb.")
        return ops, value

    def evaluate(self, coordinates, *, backend="numpy", like=None, _indices=None):
        ops, value = self._coordinates(coordinates, backend, like)
        n, m = self.num_qubits, len(self.edges)
        alpha, omega = value[..., :n], value[..., n : n + m]
        polar, azimuth = value[..., n + m : 2 * n + m], value[..., 2 * n + m :]
        products = []
        for i, (nodes, edges, degree) in enumerate(self._groups):
            if degree:
                index = _indices[i] if _indices is not None else _index(edges, ops)
                factors = ops.xp.cos(omega[..., index])
                products.append(
                    factors.prod(dim=-1) if backend == "torch" else factors.prod(axis=-1)
                )
            else:
                products.append(alpha[..., : len(nodes)] * 0 + 1)
        combined = (
            ops.xp.cat(products, dim=-1)
            if backend == "torch"
            else ops.xp.concatenate(products, axis=-1)
        )
        order = _indices[-1] if _indices is not None else _index(self._order, ops)
        multiplier = combined[..., order]
        return (
            ops.xp.cos(polar) * ops.xp.cos(alpha)
            - ops.xp.sin(polar) * ops.xp.sin(azimuth) * ops.xp.sin(alpha)
        ) * multiplier

    def prepare_z_pairs(self, pairs=None):
        """Compile exact Z_i Z_j readouts of circuit(), in the requested pair order."""
        pairs = self.edges if pairs is None else pairs
        if isinstance(pairs, (str, bytes, Mapping, Set)):
            raise ValidationError("Provide an ordered iterable of IQP wire pairs.")
        neighbors = [{} for _ in range(self.num_qubits)]
        for edge, (a, b) in enumerate(self.edges):
            neighbors[a][b] = neighbors[b][a] = edge
        buckets = {}
        work = 0
        try:
            for row, (a, b) in enumerate(pairs):
                a, b = integer(a, "pair wire"), integer(b, "pair wire")
                if a == b or max(a, b) >= self.num_qubits:
                    raise ValidationError("IQP readout pairs need distinct in-range wires.")
                left, right = neighbors[a], neighbors[b]
                work += 1 + len(left) + len(right)
                if work > self.max_work or 256 * work > memory_budget(self.max_memory_mb) * 2**20:
                    raise ResourceError("IQP pair descriptors exceed work/memory budget.")
                common = sorted(left.keys() & right.keys())
                single = tuple(left[q] for q in left if q != b and q not in right)
                single += tuple(right[q] for q in right if q != a and q not in left)
                group = buckets.setdefault((len(single), len(common)), [])
                group.append(
                    (
                        row,
                        a,
                        b,
                        single,
                        tuple(left[q] for q in common),
                        tuple(right[q] for q in common),
                    )
                )
        except (TypeError, ValueError) as exc:
            raise ValidationError("Provide ordered pairs of IQP wire indices.") from exc
        groups, order = [], []
        for entries in buckets.values():
            rows, *selectors = zip(*entries, strict=True)
            order.extend(rows)
            groups.append(tuple(selectors))
        inverse = tuple(int(i) for i in np.argsort(order))
        return IQPPairReadout(self, tuple(groups), inverse, work)

    def circuit(self):
        """Export the prepared IQP state, excluding the adaptive readout axes."""
        if 3 * self.num_qubits + len(self.edges) > self.max_work:
            raise ResourceError("IQP IR construction exceeds max_work.")
        builder = CircuitBuilder(self.num_qubits)
        for q in range(self.num_qubits):
            builder.h(q)
        for q in range(self.num_qubits):
            builder.rz(Parameter(f"alpha_{q}"), q)
        for e, (a, b) in enumerate(self.edges):
            builder.rzz(Parameter(f"omega_{e}"), a, b)
        for q in range(self.num_qubits):
            builder.h(q)
        return builder.build()


@dataclass(frozen=True)
class IQPPairReadout:
    graph: IQPGraph
    _groups: tuple = field(repr=False)
    _order: tuple = field(repr=False)
    _work: int = field(repr=False)

    def evaluate(self, coordinates, *, backend="numpy", like=None):
        """Return [..., pairs] in the computational basis, before adaptive readout rotations."""
        ops, value = self.graph._coordinates(coordinates, backend, like)
        work = math.prod(value.shape[:-1]) * self._work
        if (
            work > self.graph.max_work
            or 64 * work > memory_budget(self.graph.max_memory_mb) * 2**20
        ):
            raise ResourceError("IQP pair evaluation exceeds work/memory budget.")
        if not self._order:
            return value[..., :0]
        n = self.graph.num_qubits
        omega = value[..., n : n + len(self.graph.edges)]
        xp = ops.xp

        def cosine_product(angle):
            factors = xp.cos(angle)
            return factors.prod(dim=-1) if backend == "torch" else factors.prod(axis=-1)

        outputs = []
        for left, right, single, common_left, common_right in self._groups:
            a, b = value[..., _index(left, ops)], value[..., _index(right, ops)]
            plus, minus = xp.cos(a + b), xp.cos(a - b)
            if common_left[0]:
                u = omega[..., _index(common_left, ops)]
                v = omega[..., _index(common_right, ops)]
                plus = plus * cosine_product(u + v)
                minus = minus * cosine_product(u - v)
            pair = 0.5 * (plus + minus)
            if single[0]:
                pair = pair * cosine_product(omega[..., _index(single, ops)])
            outputs.append(pair)
        joined = xp.cat(outputs, dim=-1) if backend == "torch" else xp.concatenate(outputs, axis=-1)
        return joined[..., _index(self._order, ops)]


def _index(values, ops):
    if ops.backend == "torch":
        return ops.xp.as_tensor(values, dtype=ops.xp.long, device=ops.device)
    return np.asarray(values, dtype=np.intp) if ops.backend == "numpy" else ops.xp.asarray(values)
