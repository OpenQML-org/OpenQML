"""Causal scenario trees, conditional risk and funded contingent policies."""

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from ..compute.kernel import compute_operation, memory_budget
from ..errors import ResourceError, ValidationError
from ..ir.core import integer
from ..ir.parameters import real
from ._array import Ops, admission, rows
from .execution_risk import execution_risk_projection
from .expectiles import expectile
from .risk import expected_shortfall
from .transport import _metric_values


class TreeRiskResult(NamedTuple):
    value: object
    node_values: object
    node_mass: object


class ScenarioTreeResult(NamedTuple):
    wealth: object
    positions: object
    cash: object
    costs: object
    scale: object
    robust_risk: object
    terminal_wealth: object
    risk: TreeRiskResult


@dataclass(frozen=True, eq=False)
class ScenarioTreePlan:
    """Finite paths [S,H,N] and information-node labels [S,H+1]."""

    scenarios: object
    node_ids: object
    max_memory_mb: float = 256
    max_work: int = 1_000_000
    node_keys: tuple = field(init=False)
    _members: tuple = field(init=False, repr=False)
    _parents: tuple = field(init=False, repr=False)
    _children: tuple = field(init=False, repr=False)
    _edges: tuple = field(init=False, repr=False)
    _leaves: tuple = field(init=False, repr=False)
    _decisions: tuple = field(init=False, repr=False)
    _decision_index: dict = field(init=False, repr=False)

    def __post_init__(self):
        budget = memory_budget(self.max_memory_mb)
        work = integer(self.max_work, "max_work", 1)
        try:
            paths, labels = np.asarray(self.scenarios, dtype=float), np.asarray(self.node_ids)
        except (TypeError, ValueError, OverflowError, RuntimeError) as exc:
            raise ValidationError(
                "Tree paths and labels require static rectangular arrays."
            ) from exc
        if (
            paths.ndim != 3
            or min(paths.shape) < 1
            or not np.isfinite(paths).all()
            or np.any(paths < -1)
        ):
            raise ValidationError("Tree paths require finite arithmetic returns [S,H,N] >= -1.")
        s, h, n = paths.shape
        if (
            labels.shape != (s, h + 1)
            or labels.dtype.kind not in "iu"
            or np.any(labels < 0)
            or np.any(labels[:, 0] != labels[0, 0])
            or len(set(labels[:, -1].tolist())) != s
        ):
            raise ValidationError("Node labels require one root and a distinct terminal per path.")
        if s * h * (n + 4) > work:
            raise ResourceError("Scenario tree preprocessing exceeds max_work.")
        admission(4 * s * h * n + 96 * s * (h + 1), budget)
        keys, groups, lookup = [], [], {}
        for t in range(h + 1):
            for path in range(s):
                key = t, int(labels[path, t])
                if key not in lookup:
                    lookup[key] = len(keys)
                    keys.append(key)
                    groups.append([])
                groups[lookup[key]].append(path)
        parents, children, edges = [-1] * len(keys), [[] for _ in keys], [None] * len(keys)
        for index, ((stage, _), members) in enumerate(zip(keys, groups, strict=True)):
            if stage == 0:
                continue
            previous = labels[members, stage - 1]
            incoming = paths[members, stage - 1]
            if np.any(previous != previous[0]) or np.any(incoming != incoming[0]):
                raise ValidationError(
                    "Information nodes must share one parent and observed return."
                )
            parent = lookup[stage - 1, int(previous[0])]
            parents[index] = parent
            children[parent].append(index)
            edge = incoming[0].copy()
            edge.setflags(write=False)
            edges[index] = edge
        paths, labels = paths.copy(), labels.copy()
        for value in (paths, labels):
            value.setflags(write=False)
        decisions = tuple(i for i, key in enumerate(keys) if key[0] < h)
        leaves = tuple(lookup[h, int(labels[i, h])] for i in range(s))
        for name, value in (
            ("scenarios", paths),
            ("node_ids", labels),
            ("max_memory_mb", budget),
            ("max_work", work),
            ("node_keys", tuple(keys)),
            ("_members", tuple(tuple(g) for g in groups)),
            ("_parents", tuple(parents)),
            ("_children", tuple(tuple(c) for c in children)),
            ("_edges", tuple(edges)),
            ("_leaves", leaves),
            ("_decisions", decisions),
            ("_decision_index", {node: i for i, node in enumerate(decisions)}),
        ):
            object.__setattr__(self, name, value)

    @property
    def num_decisions(self):
        return len(self._decisions)

    @property
    def decision_keys(self):
        return tuple(self.node_keys[i] for i in self._decisions)

    def _masses(self, p, ops):
        return tuple(ops.sum(p[..., list(group)]) for group in self._members)

    def _conditional(self, node, mass, ops):
        values = ops.stack(tuple(mass[child] for child in self._children[node]))
        total = mass[node]
        return ops.xp.where(
            (total > 0)[..., None],
            values / ops.xp.where(total > 0, total, 1)[..., None],
            ops.xp.ones_like(values) / len(self._children[node]),
        )

    @compute_operation
    def nested_risk(self, losses, probabilities=None, *, risk_measure="cvar", confidence=0.95):
        """Backward composition of conditional CVaR or coherent expectiles."""
        confidence = real(confidence)
        if (
            not isinstance(risk_measure, str)
            or risk_measure not in {"cvar", "expectile"}
            or not ((0.5 if risk_measure == "expectile" else 0) <= confidence < 1)
        ):
            raise ValidationError("Use cvar confidence in [0,1) or expectile level in [0.5,1).")
        ops = Ops(losses, probabilities)
        values = ops.real(losses)
        s = self.scenarios.shape[0]
        if values.ndim < 1 or values.shape[-1] != s:
            raise ValidationError("Tree losses require one terminal value per scenario.")
        p = ops.probabilities(probabilities, s)
        prefix = ops.broadcast_shape(values.shape[:-1], p.shape[:-1])
        admission(rows(prefix) * (8 * len(self.node_keys) + 8 * s), self.max_memory_mb)
        p, values = ops.broadcast(p, prefix + (s,)), ops.broadcast(values, prefix + (s,))
        mass = self._masses(p, ops)
        result = [None] * len(self.node_keys)
        for i, leaf in enumerate(self._leaves):
            result[leaf] = values[..., i]
        for node in reversed(self._decisions):
            children = ops.stack(tuple(result[c] for c in self._children[node]))
            conditional = self._conditional(node, mass, ops)
            result[node] = (
                expected_shortfall(children, confidence=confidence, probabilities=conditional)
                if risk_measure == "cvar"
                else expectile(
                    children,
                    level=confidence,
                    probabilities=conditional,
                    max_memory_mb=self.max_memory_mb,
                )
            )
        return TreeRiskResult(result[0], ops.stack(result), ops.stack(mass))

    @compute_operation
    def rollout(
        self,
        proposals,
        probabilities=None,
        *,
        previous=0,
        fees=0.001,
        impact=0,
        budget=0.02,
        radius=0,
        cash_returns=0,
        risk_measure="cvar",
        confidence=0.95,
        transport_metric=None,
    ):
        """One funded, risk-constrained policy row per information node."""
        ops = Ops(
            proposals,
            probabilities,
            previous,
            fees,
            impact,
            budget,
            radius,
            cash_returns,
            *_metric_values(transport_metric),
        )
        s, _, n = self.scenarios.shape
        d, total_nodes = self.num_decisions, len(self.node_keys)
        w, held, fee, quadratic = (ops.real(v) for v in (proposals, previous, fees, impact))
        if w.ndim < 2 or w.shape[-2:] != (d, n):
            raise ValidationError("Tree proposals require [...,num_decisions,num_assets].")
        if held.ndim and held.shape[-1] != n:
            raise ValidationError("Initial holding width must match the assets.")
        if any(v.ndim > 1 or (v.ndim and v.shape != (n,)) for v in (fee, quadratic)):
            raise ValidationError("Tree fees and impact require scalars or asset vectors.")
        controls = [ops.real(v) for v in (budget, radius, cash_returns)]
        if any(v.ndim and v.shape[-1] not in (1, d) for v in controls):
            raise ValidationError("Tree controls require scalars or a final decision-node axis.")
        p = ops.probabilities(probabilities, s)
        metric_prefix = ()
        if transport_metric is not None:
            diagonal = ops.real(transport_metric.diagonal)
            factors = (
                None if transport_metric.factors is None else ops.real(transport_metric.factors)
            )
            metric_prefix = ops.broadcast_shape(
                diagonal.shape[:-1], () if factors is None else factors.shape[:-2]
            )
        prefix = ops.broadcast_shape(
            w.shape[:-2],
            held.shape[:-1] if held.ndim else (),
            p.shape[:-1],
            metric_prefix,
            *(v.shape[:-1] if v.ndim else () for v in controls),
        )
        admission(rows(prefix) * (24 * s + 24 * total_nodes * (n + 4)), self.max_memory_mb)
        w, held, p = (
            ops.broadcast(v, prefix + shape) for v, shape in ((w, (d, n)), (held, (n,)), (p, (s,)))
        )
        bounds, radii, cash_rate = [ops.broadcast(v, prefix + (d,)) for v in controls]
        eps = 2.0 ** (-52 if "64" in str(w.dtype) else -23)
        held = ops.check(
            held,
            (held >= 0) & (ops.sum(held) <= 1 + 16 * n * eps)[..., None],
            "Initial holdings must be funded and long-only.",
        )
        held = held / ops.maximum(ops.sum(held, keepdims=True), 1)
        mass = self._masses(p, ops)
        wealth, positions, cash, costs = ([None] * total_nodes for _ in range(4))
        scales, risks = [None] * d, [None] * d
        for node in range(total_nodes):
            if node == 0:
                current, cash_balance = held, 1 - ops.sum(held)
            else:
                parent = self._parents[node]
                current = positions[parent] * (1 + ops.real(self._edges[node]))
                cash_balance = cash[parent] * (1 + cash_rate[..., self._decision_index[parent]])
            before = ops.sum(current) + cash_balance
            before = ops.check(
                before,
                ops.xp.isfinite(before) & (before >= 0),
                "Scenario-tree wealth must remain finite and nonnegative.",
            )
            wealth[node] = before
            if not self._children[node]:
                positions[node], cash[node], costs[node] = current, cash_balance, before * 0
                continue
            index = self._decision_index[node]
            returns = np.stack([self._edges[child] for child in self._children[node]])
            result = execution_risk_projection(
                w[..., index, :],
                returns,
                previous=current / ops.xp.where(before > 0, before, 1)[..., None],
                probabilities=self._conditional(node, mass, ops),
                fees=fee,
                impact=quadratic,
                budget=bounds[..., index],
                radius=radii[..., index],
                cash_return=cash_rate[..., index],
                risk_measure=risk_measure,
                confidence=confidence,
                transport_metric=transport_metric,
                max_memory_mb=self.max_memory_mb,
            )
            positions[node], cash[node], costs[node] = (
                before[..., None] * result.positions,
                before * result.cash,
                before * result.costs,
            )
            scales[index] = ops.xp.where(before > 0, result.scale, 0)
            risks[index] = ops.xp.where(before > 0, result.robust_risk, 0)
        terminal = ops.stack(tuple(wealth[leaf] for leaf in self._leaves))
        nested = self.nested_risk(1 - terminal, p, risk_measure=risk_measure, confidence=confidence)
        return ScenarioTreeResult(
            ops.stack(wealth),
            ops.stack(positions, axis=-2),
            ops.stack(cash),
            ops.stack(costs),
            ops.stack(scales),
            ops.stack(risks),
            terminal,
            nested,
        )
