"""Hard-cardinality portfolio QUBOs and constraint-preserving CD/XY circuits."""

import itertools
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from ..compute import get_compute_kernel, memory_budget
from ..errors import ResourceError, ValidationError
from ..ir import CircuitBuilder, integer
from ..ir.observables import Hamiltonian, PauliString
from ..ir.parameters import Parameter, real
from ..qml.layers import LayerSpec
from ._array import Ops, admission


def _fixed(value, shape=None):
    try:
        a = np.asarray(value)
        if a.dtype.kind not in "fiu" or (shape is not None and a.shape != shape):
            raise ValidationError("Invalid fixed financial estimate shape or numeric type.")
        a = np.array(a, dtype=float, copy=True)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValidationError(
            "QUBO construction needs fixed real estimates; detach device tensors explicitly."
        ) from exc
    if not np.all(np.isfinite(a)):
        raise ValidationError("Fixed financial estimates must be finite.")
    a.setflags(write=False)
    return a


def _bits(value, n):
    a = np.asarray(value)
    if a.shape != (n,) or a.dtype.kind not in "biu" or not np.all((a == 0) | (a == 1)):
        raise ValidationError("Selections require a binary vector with one entry per asset.")
    return a.astype(np.int8)


class SelectionResult(NamedTuple):
    selection: np.ndarray
    cost: float


class PortfolioSamples(NamedTuple):
    selection: np.ndarray
    cost: float
    feasible_shots: int
    invalid_shots: int
    conditional_mean_cost: float


@dataclass(frozen=True)
class CardinalityProblem:
    matrix: object
    linear: object
    cardinality: int
    offset: float = 0
    max_memory_mb: float = 256

    def __post_init__(self):
        linear = _fixed(self.linear)
        if linear.ndim != 1 or not len(linear):
            raise ValidationError("A cardinality QUBO requires a nonempty asset vector.")
        matrix = _fixed(self.matrix, (len(linear), len(linear)))
        if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
            raise ValidationError("QUBO matrix must be symmetric.")
        k = integer(self.cardinality, "cardinality", 0)
        if k > len(linear):
            raise ValidationError("Cardinality exceeds the asset universe.")
        admission(3 * matrix.size + 4 * linear.size, self.max_memory_mb)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "linear", linear)
        object.__setattr__(self, "cardinality", k)
        object.__setattr__(self, "offset", real(self.offset))
        object.__setattr__(self, "max_memory_mb", memory_budget(self.max_memory_mb))

    @property
    def num_assets(self):
        return len(self.linear)

    def cost(self, selection, *, require_feasible=True):
        x = _bits(selection, self.num_assets)
        if require_feasible and int(x.sum()) != self.cardinality:
            raise ValidationError("Selection violates the hard cardinality constraint.")
        return float(
            get_compute_kernel().matmul(get_compute_kernel().matmul(x, self.matrix), x)
            + get_compute_kernel().matmul(self.linear, x)
            + self.offset
        )

    def condition(self, indices, fixed_selection):
        """Exact conditional QUBO, including cross-block costs and global offset."""
        indices = tuple(integer(i, "free asset", 0) for i in indices)
        if not indices or len(set(indices)) != len(indices) or max(indices) >= self.num_assets:
            raise ValidationError("Free assets must be nonempty, unique and in range.")
        fixed = _bits(fixed_selection, self.num_assets)
        outside = np.asarray(
            [i for i in range(self.num_assets) if i not in set(indices)], dtype=int
        )
        free = np.asarray(indices)
        k = self.cardinality - int(fixed[outside].sum())
        if not 0 <= k <= len(free):
            raise ValidationError("Fixed holdings leave an infeasible conditional cardinality.")
        x = fixed[outside]
        linear = self.linear[free] + (
            get_compute_kernel().matmul(2 * self.matrix[np.ix_(free, outside)], x)
            if len(outside)
            else 0
        )
        offset = self.offset + (
            float(
                get_compute_kernel().matmul(
                    get_compute_kernel().matmul(x, self.matrix[np.ix_(outside, outside)]), x
                )
                + get_compute_kernel().matmul(self.linear[outside], x)
            )
            if len(outside)
            else 0
        )
        return CardinalityProblem(
            self.matrix[np.ix_(free, free)], linear, k, offset, self.max_memory_mb
        )

    def merge(self, indices, local_selection, fixed_selection):
        indices = tuple(indices)
        conditional = self.condition(indices, fixed_selection)
        local = _bits(local_selection, len(indices))
        conditional.cost(local)
        result = _bits(fixed_selection, self.num_assets)
        result[np.asarray(indices)] = local
        self.cost(result)
        return result

    def exact(self, *, max_states=100_000):
        limit = integer(max_states, "max_states", 1)
        if math.comb(self.num_assets, self.cardinality) > limit:
            raise ResourceError(
                "Exact cardinality search exceeds max_states; condition a smaller block."
            )
        best = None
        for chosen in itertools.combinations(range(self.num_assets), self.cardinality):
            x = np.zeros(self.num_assets, np.int8)
            x[list(chosen)] = 1
            cost = self.cost(x)
            if best is None or cost < best.cost:
                best = SelectionResult(x, cost)
        return best

    def refine(self, selection, *, max_swaps=32, max_work=100_000_000):
        x = _bits(selection, self.num_assets)
        self.cost(x)
        count = integer(max_swaps, "max_swaps", 0)
        work = self.num_assets**2 + count * self.cardinality * (self.num_assets - self.cardinality)
        if work > integer(max_work, "max_work", 1):
            raise ResourceError("Cardinality-preserving local refinement exceeds max_work.")
        product = get_compute_kernel().matmul(self.matrix, x)
        for _ in range(count):
            held, absent = np.flatnonzero(x), np.flatnonzero(1 - x)
            if not len(held) or not len(absent):
                break
            i, j = held[:, None], absent[None, :]
            delta = (
                2 * (product[j] - product[i])
                + self.matrix[i, i]
                + self.matrix[j, j]
                - 2 * self.matrix[i, j]
                + self.linear[j]
                - self.linear[i]
            )
            row, column = np.unravel_index(np.argmin(delta), delta.shape)
            if delta[row, column] >= 0:
                break
            old, new = held[row], absent[column]
            x[old], x[new] = 0, 1
            product = product + self.matrix[:, new] - self.matrix[:, old]
        return SelectionResult(x, self.cost(x))

    def qaoa(self, **options):
        return CardinalityQAOAPlan(self, **options)


@dataclass(frozen=True)
class PortfolioSelection:
    expected_returns: object
    covariance: object
    cardinality: int
    risk_aversion: float = 1
    previous: object = None
    fees: object = 0
    turnover: float = 0
    ridge: float = 0
    max_memory_mb: float = 256
    problem: CardinalityProblem = field(init=False, repr=False)

    def __post_init__(self):
        mean = _fixed(self.expected_returns)
        if mean.ndim != 1 or not len(mean):
            raise ValidationError("Expected returns must have one fixed value per asset.")
        n, k = len(mean), integer(self.cardinality, "cardinality", 1)
        if k > n:
            raise ValidationError("Cardinality exceeds the asset universe.")
        covariance = _fixed(self.covariance, (n, n))
        if not np.allclose(covariance, covariance.T, atol=1e-12, rtol=1e-10) or np.linalg.eigvalsh(
            covariance
        )[0] < -1e-10 * max(1, np.max(abs(covariance))):
            raise ValidationError("Portfolio covariance must be symmetric positive semidefinite.")
        previous = _fixed(np.zeros(n) if self.previous is None else self.previous, (n,))
        fees = _fixed(np.broadcast_to(self.fees, (n,)))
        values = tuple(real(getattr(self, name)) for name in ("risk_aversion", "turnover", "ridge"))
        if min(values) < 0 or np.any(previous < 0) or np.any(fees < 0):
            raise ValidationError(
                "Portfolio risk, trading penalties and previous holdings must be nonnegative."
            )
        risk, turnover, ridge = values
        matrix = 0.5 * risk * covariance / k**2 + np.eye(n) * (ridge + turnover) / (2 * k**2)
        linear = (
            -mean / k - turnover * previous / k + fees * (abs(1 / k - previous) - abs(previous))
        )
        offset = 0.5 * turnover * float(get_compute_kernel().matmul(previous, previous)) + float(
            get_compute_kernel().matmul(fees, abs(previous))
        )
        object.__setattr__(
            self, "problem", CardinalityProblem(matrix, linear, k, offset, self.max_memory_mb)
        )
        object.__setattr__(self, "cardinality", k)
        for name, value in (
            ("expected_returns", mean),
            ("covariance", covariance),
            ("previous", previous),
            ("fees", fees),
        ):
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class CardinalityQAOAPlan:
    problem: CardinalityProblem
    depth: int = 3
    topology: str = "ring"
    counterdiabatic: str = "local"
    initial_selection: object = None
    initial_mixing: float = np.pi / 4
    max_operations: int = 100_000
    spec: LayerSpec = field(init=False, repr=False)
    _initial: tuple = field(init=False, repr=False)
    _cd_pairs: tuple = field(init=False, repr=False)
    energy_scale: float = field(init=False)

    def __post_init__(self):
        if not isinstance(self.problem, CardinalityProblem):
            raise ValidationError("Provide a CardinalityProblem.")
        depth, limit = (
            integer(self.depth, "depth", 1),
            integer(self.max_operations, "max_operations", 1),
        )
        if self.topology not in {"line", "ring"} or self.counterdiabatic not in {
            "none",
            "local",
            "full",
        }:
            raise ValidationError(
                "Use line/ring topology and none/local/full counterdiabatic correction."
            )
        n, problem = self.problem.num_assets, self.problem
        if n > limit:
            raise ResourceError("Circuit width exceeds max_operations.")
        selection = np.zeros(n, np.int8)
        selection[
            np.argsort(problem.linear + np.diag(problem.matrix), kind="stable")[
                : problem.cardinality
            ]
        ] = 1
        if self.initial_selection is not None:
            selection = _bits(self.initial_selection, n)
        problem.cost(selection)
        h = -0.5 * (problem.linear + problem.matrix.sum(axis=1))
        coupling = problem.matrix * 0.5
        constant = (
            problem.offset
            + 0.5 * float(problem.linear.sum())
            + 0.25 * float(problem.matrix.sum() + np.trace(problem.matrix))
        )
        norm = float(np.sum(abs(h)) + np.sum(abs(np.triu(coupling, 1))))
        norm = norm if norm > 0 else 1
        edges = list(zip(range(n - 1), range(1, n), strict=True))
        if self.topology == "ring" and n > 2:
            edges.append((n - 1, 0))
        cd_pairs = []
        if self.counterdiabatic != "none":
            for i, j in edges:
                if h[i] != h[j]:
                    cd_pairs.append((i, j, None, float((h[i] - h[j]) / norm)))
                if self.counterdiabatic == "full":
                    for k in range(n):
                        if k not in (i, j) and coupling[i, k] != coupling[j, k]:
                            cd_pairs.append(
                                (i, j, k, float((coupling[i, k] - coupling[j, k]) / norm))
                            )
        terms = [(constant, {})] if constant else []
        terms += [(float(c), {i: "Z"}) for i, c in enumerate(h) if c]
        terms += [
            (float(coupling[i, j]), {i: "Z", j: "Z"})
            for i in range(n)
            for j in range(i + 1, n)
            if coupling[i, j]
        ]
        if not terms:
            terms = [(0.0, {})]
        descriptor = (n + 128) * len(terms) + 128 * len(cd_pairs)
        if descriptor > problem.max_memory_mb * 2**20:
            raise ResourceError("QAOA descriptors exceed max_memory_mb.")
        builder, parameters, initial = CircuitBuilder(n), [], []
        count = 0

        def emit(gate, *wires, angle=None):
            nonlocal count
            count += 1
            if count > limit or descriptor + 512 * count > problem.max_memory_mb * 2**20:
                raise ResourceError("QAOA circuit exceeds operation or descriptor budget.")
            builder.gate(gate, *wires, params=() if angle is None else (angle,))

        def rotation(sites, angle):
            wires = tuple(sorted(sites))
            for q in wires:
                if sites[q] == "Y":
                    emit("sdg", q)
                if sites[q] in {"X", "Y"}:
                    emit("h", q)
            pairs = list(zip(wires, wires[1:], strict=False))
            for a, b in pairs:
                emit("cx", a, b)
            emit("rz", wires[-1], angle=angle)
            for a, b in reversed(pairs):
                emit("cx", a, b)
            for q in reversed(wires):
                if sites[q] in {"X", "Y"}:
                    emit("h", q)
                if sites[q] == "Y":
                    emit("s", q)

        def mixer(angle):
            for i, j in edges:
                emit("rxx", i, j, angle=angle)
                emit("ryy", i, j, angle=angle)

        for q in np.flatnonzero(selection):
            emit("x", int(q))
        mixing = real(self.initial_mixing)
        if mixing and edges:
            mixer(mixing)
        phase = 0
        for layer in range(depth):
            fraction = (layer + 0.5) / depth
            if np.any(h) or np.any(np.triu(coupling, 1)) or constant:
                name = f"gamma_{layer}"
                parameters.append(name)
                initial.append(fraction / depth)
                gamma = Parameter(name)
                phase = phase - gamma * (constant / norm)
                for i, c in enumerate(h):
                    if c:
                        emit("rz", i, angle=2 * c / norm * gamma)
                for i in range(n):
                    for j in range(i + 1, n):
                        if coupling[i, j]:
                            emit("rzz", i, j, angle=2 * coupling[i, j] / norm * gamma)
            if edges:
                name = f"beta_{layer}"
                parameters.append(name)
                initial.append((1 - fraction) / depth)
                mixer(Parameter(name))
            if cd_pairs:
                name = f"cd_{layer}"
                parameters.append(name)
                initial.append(0.0)
                alpha = Parameter(name)
                for i, j, k, c in cd_pairs:
                    first, second = {i: "Y", j: "X"}, {i: "X", j: "Y"}
                    if k is not None:
                        first[k] = second[k] = "Z"
                    rotation(first, 2 * c * alpha)
                    rotation(second, -2 * c * alpha)
        circuit = builder.build().with_global_phase(phase)
        observables = []
        for coefficient, sites in terms:
            word = ["I"] * n
            for q, p in sites.items():
                word[q] = p
            observables.append(PauliString("".join(word), coefficient))
        object.__setattr__(
            self,
            "spec",
            LayerSpec(circuit, (), tuple(parameters), (Hamiltonian(tuple(observables)),)),
        )
        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "_initial", tuple(initial))
        object.__setattr__(self, "_cd_pairs", tuple(cd_pairs))
        object.__setattr__(self, "energy_scale", norm)

    def to_spec(self):
        return self.spec

    def to_dict(self):
        return dict(
            architecture="cardinality_preserving_portfolio_QAOA",
            num_qubits=self.problem.num_assets,
            cardinality=self.problem.cardinality,
            depth=self.depth,
            topology=self.topology,
            counterdiabatic=self.counterdiabatic,
            energy_scale=self.energy_scale,
            cd_pairs=len(self._cd_pairs),
            operations=len(self.spec.circuit.operations),
            parameters=len(self.spec.weight_parameters),
            initialization="feasible_basis_plus_XY; not_uniform_Dicke",
            guarantee="ideal_unitary_cardinality",
        )

    def initial_parameters(self, *, dtype="float64"):
        if np.dtype(dtype) not in {np.dtype("float32"), np.dtype("float64")}:
            raise ValidationError("QAOA parameters require float32/64.")
        return np.asarray(self._initial, dtype=dtype)

    def bound(self, weights):
        values = _fixed(weights, (len(self.spec.weight_parameters),))
        return self.spec.circuit.bind(dict(zip(self.spec.weight_parameters, values, strict=True)))

    def expectation(self, weights, *, backend=None, **options):
        from ..sim.dense import expectations

        ops = Ops(weights)
        w = ops.real(weights)
        if w.shape[-1:] != (len(self.spec.weight_parameters),):
            raise ValidationError("QAOA weight width differs from the circuit.")
        if not self.spec.weight_parameters:
            return ops.sum(w) * 0
        bindings = {name: w[..., i] for i, name in enumerate(self.spec.weight_parameters)}
        options.setdefault("max_memory_mb", self.problem.max_memory_mb)
        return expectations(
            self.spec.circuit,
            self.spec.observables,
            bindings,
            backend=ops.name if backend is None else backend,
            **options,
        )[..., 0]

    def sample(self, weights, *, shots=1024, backend="auto", **options):
        from ..sim.dense import sample

        options.setdefault("max_memory_mb", self.problem.max_memory_mb)
        return self.decode_counts(
            sample(self.bound(weights), shots=shots, backend=backend, **options)
        )

    def decode_counts(self, counts):
        if not isinstance(counts, Mapping) or not counts:
            raise ValidationError("Provide nonempty canonical q0-MSB sample counts.")
        feasible, invalid, total_cost, best = 0, 0, 0.0, None
        for word, count in counts.items():
            count = integer(count, "shot count", 0)
            if (
                not isinstance(word, str)
                or len(word) != self.problem.num_assets
                or set(word) - {"0", "1"}
            ):
                raise ValidationError(
                    "Counts must contain q0-MSB bit strings matching the asset universe."
                )
            if not count:
                continue
            x = np.fromiter((int(q) for q in word), np.int8)
            if int(x.sum()) != self.problem.cardinality:
                invalid += count
                continue
            cost = self.problem.cost(x)
            feasible += count
            total_cost += count * cost
            if best is None or cost < best.cost:
                best = SelectionResult(x, cost)
        if not feasible:
            raise ValidationError(
                "No feasible cardinality sample; inspect hardware noise or use a smaller circuit."
            )
        return PortfolioSamples(best.selection, best.cost, feasible, invalid, total_cost / feasible)
