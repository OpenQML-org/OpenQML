"""Independent maximality, after-cost ledger and native hypergradient checks."""

import numpy as np
import pytest

import openqml as oq

RETURNS = np.array([[-0.12, -0.03], [0.04, -0.08], [0.1, 0.02], [-0.05, 0.01]])
PROBABILITIES = np.array([0.2, 0.3, 0.4, 0.1])


def _risk(loss, p, level, measure, margin):
    p = p / p.sum()
    if measure == "cvar":
        order = np.argsort(loss)[::-1]
        mass = p[order]
        admitted = np.minimum(mass, np.maximum(1 - level - np.cumsum(mass) + mass, 0))
        return np.dot(admitted, loss[order]) / (1 - level) + margin / (1 - level)
    low, high = min(loss) - 1, max(loss) + 1 + level * margin / (1 - level)
    for _ in range(120):
        middle = (low + high) / 2
        residual = (
            level * np.dot(p, np.maximum(loss - middle, 0))
            - (1 - level) * np.dot(p, np.maximum(middle - loss, 0))
            + level * margin
        )
        if residual > 0:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def _reference(scale, w, previous, r, p, fees, impact, cash_return, level, measure, radius, metric):
    positions = scale * w
    trades = positions - previous
    costs = np.sum(fees * np.abs(trades) + impact * trades**2)
    cash = 1 - positions.sum() - costs
    losses = costs - r @ positions - cash_return * cash
    margin = radius * np.sqrt(positions @ np.linalg.solve(metric, positions))
    return _risk(losses, p, level, measure, margin), cash, costs


@pytest.mark.parametrize(
    "measure,level", [("cvar", 0), ("cvar", 0.8), ("expectile", 0.5), ("expectile", 0.9)]
)
@pytest.mark.parametrize("cash_return", [0, 0.003])
def test_sorted_quadratic_projection_matches_independent_feasible_prefix(
    measure, level, cash_return
):
    f = oq.finance
    w, previous = np.array([0.4, 0.5]), np.array([0.3, 0.2])
    fees, impact = np.array([0.001, 0.002]), np.array([0.02, 0.03])
    diagonal, factors = np.array([0.7, 1.2]), np.array([[0.2, -0.3]])
    metric = np.diag(diagonal) + factors.T @ factors
    options = dict(
        previous=previous,
        fees=fees,
        impact=impact,
        probabilities=PROBABILITIES,
        risk_measure=measure,
        confidence=level,
        budget=0.015,
        radius=0.001,
        cash_return=cash_return,
        transport_metric=f.WassersteinMetric(diagonal, factors),
    )
    result = f.execution_risk_projection(w, RETURNS, **options)

    def evaluate(scale):
        return _reference(
            scale,
            w,
            previous,
            RETURNS,
            PROBABILITIES,
            fees,
            impact,
            cash_return,
            level,
            measure,
            0.001,
            metric,
        )

    if evaluate(1)[0] <= 0.015 and evaluate(1)[1] >= 0:
        expected = 1.0
    else:
        low, high = 0.0, 1.0
        for _ in range(90):
            midpoint = (low + high) / 2
            risk, cash, _ = evaluate(midpoint)
            if risk <= 0.015 and cash >= 0:
                low = midpoint
            else:
                high = midpoint
        expected = (low + high) / 2
    assert result.scale == pytest.approx(expected, abs=4e-13)
    risk, cash, cost = evaluate(float(result.scale))
    assert result.robust_risk == pytest.approx(risk, abs=2e-14)
    assert result.cash == pytest.approx(cash, abs=2e-14)
    assert result.costs == pytest.approx(cost, abs=2e-14)
    assert risk <= 0.015 + 2e-14 and cash >= -2e-14
    assert result.positions.sum() + result.cash + result.costs == pytest.approx(1, abs=2e-14)
    np.testing.assert_allclose(result.weights * (1 - result.costs), result.positions, atol=2e-14)


def test_funding_bound_context_batches_and_zero_proposals():
    f = oq.finance
    w = np.array([[0.4, 0.6], [0.0, 0.9]])
    result = f.execution_risk_projection(w, np.zeros((3, 2)), previous=0, fees=0.01, budget=0.5)
    expected = np.minimum(1, 1 / ((1 + 0.01) * w.sum(-1)))
    np.testing.assert_allclose(result.scale, expected, atol=2e-15)
    np.testing.assert_allclose(result.cash, 1 - expected * w.sum(-1) * (1 + 0.01), atol=2e-15)
    assert (
        f.execution_risk_projection(
            [0.0, 0.0], RETURNS, previous=[0.2, 0.3], budget=0.02
        ).positions.sum()
        == 0
    )


@pytest.mark.parametrize("measure,level", [("cvar", 0.8), ("expectile", 0.9)])
def test_returned_post_cost_targets_reproduce_self_financing_ledger(measure, level):
    f = oq.finance
    previous = np.array([0.3, 0.2])
    fee, impact = np.array([0.001, 0.002]), np.array([0.02, 0.03])
    result = f.execution_risk_projection(
        [0.4, 0.5],
        RETURNS,
        previous=previous,
        fees=fee,
        impact=impact,
        risk_measure=measure,
        confidence=level,
        budget=0.015,
        cash_return=0.003,
    )
    count = len(RETURNS)
    target = np.broadcast_to(result.weights, (count, 1, 2))
    state = f.TradingState(
        np.broadcast_to(previous, (count, 2)),
        np.full(count, 1 - previous.sum()),
        target,
        np.ones((count, 1), dtype=bool),
    )
    ledger = f.SelfFinancingPlan(fees=fee, impact=impact).rollout(
        RETURNS[:, None, :], target, state=state, cash_returns=0.003
    )
    np.testing.assert_allclose(ledger.costs[:, 0], result.costs, atol=3e-14)
    loss = 1 - ledger.wealth[:, 0]
    nominal = _risk(loss, np.ones(count) / count, level, measure, 0)
    assert nominal == pytest.approx(float(result.nominal_risk), abs=4e-14)


def _objective(values, backend):
    f = oq.finance
    state = oq.statevector(
        oq.Circuit(1).ry(oq.Parameter("theta"), 0), {"theta": values[0]}, backend=backend
    )
    p = state.real**2 + state.imag**2
    result = f.execution_risk_projection(
        [0.4, 0.5],
        RETURNS[:2],
        previous=[0.3, 0.2],
        fees=values[1:3],
        impact=values[3:5],
        probabilities=p,
        risk_measure="expectile",
        confidence=0.8,
        radius=values[5],
        budget=0.018,
    )
    return -0.03 * result.positions[0] + 0.01 * result.positions[1]


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_quantum_probabilities_fees_impact_and_radius_keep_native_derivatives(backend):
    lib = pytest.importorskip(backend)
    xp = lib if backend == "torch" else lib.numpy
    dtype = xp.float64 if backend == "torch" else xp.float32
    base = np.array([1.4, 0.005, 0.003, 0.04, 0.02, 0.001])
    values = xp.asarray(base, dtype=dtype)
    if backend == "torch":
        values.requires_grad_()
        assert lib.autograd.gradcheck(lambda v: _objective(v, backend), (values,))
        assert lib.autograd.gradgradcheck(lambda v: _objective(v, backend), (values,))
        gradient = lib.autograd.grad(_objective(values, backend), values)[0].detach().numpy()
    else:
        gradient = np.asarray(lib.jit(lib.grad(lambda v: _objective(v, backend)))(values))

    def reference(v):
        p = np.array([np.cos(v[0] / 2) ** 2, np.sin(v[0] / 2) ** 2])
        w, previous = np.array([0.4, 0.5]), np.array([0.3, 0.2])
        low, high = 0.0, 1.0
        for _ in range(90):
            middle = (low + high) / 2
            risk, cash, _ = _reference(
                middle,
                w,
                previous,
                RETURNS[:2],
                p,
                v[1:3],
                v[3:5],
                0,
                0.8,
                "expectile",
                v[5],
                np.eye(2),
            )
            if risk <= 0.018 and cash >= 0:
                low = middle
            else:
                high = middle
        return (-0.03 * w[0] + 0.01 * w[1]) * (low + high) / 2

    expected = []
    for i in range(6):
        delta = np.eye(6)[i] * 1e-6
        expected.append((reference(base + delta) - reference(base - delta)) / 2e-6)
    assert np.isfinite(gradient).all() and (np.abs(gradient) > 1e-8).all()
    np.testing.assert_allclose(gradient, expected, rtol=3e-3, atol=3e-7)


def test_zero_initial_impact_has_a_nonzero_correct_one_sided_derivative():
    torch = pytest.importorskip("torch")
    impact = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    function = oq.finance.execution_risk_projection

    def objective(value):
        return function(
            [0.4, 0.5],
            RETURNS,
            previous=[0.3, 0.2],
            fees=0.001,
            impact=value,
            budget=0.015,
            confidence=0.8,
        ).positions.sum()

    gradient = torch.autograd.grad(objective(impact), impact)[0]
    difference = (objective(torch.tensor(1e-6, dtype=impact.dtype)) - objective(impact)) / 1e-6
    assert abs(float(gradient)) > 1e-5
    torch.testing.assert_close(gradient, difference, rtol=2e-4, atol=1e-7)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_compiled_funding_checks_survive_native_execution(backend):
    lib = pytest.importorskip(backend)
    xp = lib if backend == "torch" else lib.numpy
    project = oq.finance.execution_risk_projection
    run = oq.compile_compute(
        lambda w: project(w, RETURNS, previous=[0.3, 0.2], budget=0.015),
        backend=backend,
        **({"compiler": "eager", "fullgraph": True} if backend == "torch" else {}),
    )
    result = run(xp.asarray([0.4, 0.5]))
    assert float(result.robust_risk) <= 0.015 + 1e-6
    with pytest.raises(oq.ValidationError):
        run(xp.asarray([-0.4, 0.5]))


@pytest.mark.parametrize(
    "options",
    [
        {"fees": -1},
        {"impact": -0.1},
        {"fees": 0.5, "impact": 0.25},
        {"previous": [1.0, 1.0]},
        {"budget": 0.00001},
        {"radius": -1},
        {"risk_measure": "bad"},
        {"cash_return": -2},
        {"max_memory_mb": 1e-9},
    ],
)
def test_invalid_execution_domains_and_infeasible_liquidation_fail(options):
    kwargs = dict(previous=[0.3, 0.2], fees=0.01)
    kwargs.update(options)
    with pytest.raises((oq.ValidationError, oq.ResourceError)):
        oq.finance.execution_risk_projection([0.4, 0.5], RETURNS, **kwargs)
