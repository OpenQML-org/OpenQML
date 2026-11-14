"""Independent conditional-risk and funded causal-policy references."""

import numpy as np
import pytest

import openqml as oq

PATHS = np.array(
    [
        [[-0.06, 0.02], [-0.12, 0.01]],
        [[-0.06, 0.02], [0.08, -0.04]],
        [[0.05, -0.03], [-0.04, -0.09]],
        [[0.05, -0.03], [0.12, 0.06]],
    ]
)
NODES = np.array([[0, 1, 0], [0, 1, 1], [0, 2, 2], [0, 2, 3]])
P = np.array([0.1, 0.2, 0.3, 0.4])
W = np.array([[0.4, 0.3], [0.55, 0.2], [0.2, 0.5]])


def _risk(values, probabilities, level, measure, margin=0):
    probabilities = probabilities / sum(probabilities)
    if measure == "cvar":
        order = np.argsort(values)[::-1]
        remainder, total = 1 - level, 0.0
        for index in order:
            taken = min(remainder, probabilities[index])
            total += taken * values[index]
            remainder -= taken
        return (total + margin) / (1 - level)
    low, high = min(values) - 1, max(values) + 1 + level * margin / (1 - level)
    for _ in range(100):
        middle = (low + high) / 2
        value = level * np.dot(probabilities, np.maximum(values - middle, 0))
        value -= (1 - level) * np.dot(probabilities, np.maximum(middle - values, 0))
        value += level * margin
        if value > 0:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def _project(proposal, returns, p, previous, fee, impact, budget, radius, rate, measure, level):
    def evaluate(scale):
        positions = scale * proposal
        trade = positions - previous
        cost = np.sum(fee * np.abs(trade) + impact * trade**2)
        cash = 1 - positions.sum() - cost
        losses = cost - returns @ positions - rate * cash
        risk = _risk(losses, p, level, measure, scale * radius * np.linalg.norm(proposal))
        return positions, cash, cost, risk

    low, high = 0.0, 1.0
    if evaluate(1)[1] >= 0 and evaluate(1)[3] <= budget:
        low = 1.0
    else:
        for _ in range(100):
            middle = (low + high) / 2
            _, cash, _, risk = evaluate(middle)
            if cash >= 0 and risk <= budget:
                low = middle
            else:
                high = middle
    return (*evaluate(low), low)


@pytest.mark.parametrize("measure,level", [("cvar", 0.6), ("expectile", 0.7)])
def test_tree_wealth_risk_and_costs_match_independent_conditional_bisection_and_ledger(
    measure, level
):
    plan = oq.finance.ScenarioTreePlan(PATHS, NODES)
    options = dict(
        previous=[0.3, 0.2],
        fees=[0.001, 0.002],
        impact=[0.005, 0.008],
        budget=0.045,
        radius=0.0005,
        cash_returns=0.002,
        risk_measure=measure,
        confidence=level,
    )
    result = plan.rollout(W, P, **options)
    before = [1.0, None, None]
    positions, cash, costs, scales = [], [], [], []
    groups = [[0, 1, 2, 3], [0, 1], [2, 3]]
    for node, indices in enumerate(groups):
        if node == 0:
            previous = np.array(options["previous"])
            returns, p = PATHS[[0, 2], 0], np.array([P[:2].sum(), P[2:].sum()])
        else:
            current = positions[0] * (1 + PATHS[indices[0], 0])
            balance = cash[0] * (1 + options["cash_returns"])
            before[node] = current.sum() + balance
            previous = current / before[node]
            returns, p = PATHS[indices, 1], P[indices]
        position, balance, cost, risk, scale = _project(
            W[node],
            returns,
            p,
            previous,
            np.array(options["fees"]),
            np.array(options["impact"]),
            options["budget"],
            options["radius"],
            options["cash_returns"],
            measure,
            level,
        )
        positions.append(position * before[node])
        cash.append(balance * before[node])
        costs.append(cost * before[node])
        scales.append(scale)
        assert risk <= options["budget"] + 1e-10
    terminal = np.array(
        [
            np.sum(positions[1 + i // 2] * (1 + PATHS[i, 1]))
            + cash[1 + i // 2] * (1 + options["cash_returns"])
            for i in range(4)
        ]
    )
    losses = 1 - terminal
    conditional = np.array(
        [_risk(losses[:2], P[:2], level, measure), _risk(losses[2:], P[2:], level, measure)]
    )
    expected_risk = _risk(conditional, np.array([P[:2].sum(), P[2:].sum()]), level, measure)
    np.testing.assert_allclose(result.terminal_wealth, terminal, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.costs[:3], costs, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.scale, scales, rtol=1e-12, atol=1e-12)
    assert float(result.risk.value) == pytest.approx(expected_risk, abs=1e-12)
    np.testing.assert_allclose(
        result.positions.sum(-1) + result.cash + result.costs, result.wealth, atol=2e-15
    )
    ledger = oq.finance.SelfFinancingPlan(fees=options["fees"], impact=options["impact"])
    for i in range(4):
        branch = 1 + i // 2
        targets = np.array([positions[j] / (before[j] - costs[j]) for j in (0, branch)])
        account = ledger.rollout(
            np.concatenate((np.zeros((1, 2)), PATHS[i])),
            np.concatenate((targets, np.zeros((1, 2)))),
            initial_weights=options["previous"],
            cash_returns=[0, options["cash_returns"], options["cash_returns"]],
        )
        assert account.wealth[-1] == pytest.approx(terminal[i], abs=2e-12)


def test_nested_tail_risk_is_time_consistent_and_distinct_from_terminal_cvar():
    plan = oq.finance.ScenarioTreePlan(PATHS, NODES)
    losses = np.array([0.0, 10.0, 4.0, 4.0])
    nested = plan.nested_risk(losses, [0.25] * 4, confidence=0.5)
    assert nested.value == pytest.approx(10.0)
    assert oq.finance.expected_shortfall(losses, confidence=0.5) == pytest.approx(7.0)
    for measure, level in (("cvar", 0), ("expectile", 0.5)):
        actual = plan.nested_risk(losses, P, risk_measure=measure, confidence=level)
        assert actual.value == pytest.approx(np.dot(losses, P))
    empty = plan.nested_risk(losses, [0, 0, 0.3, 0.7], confidence=0.5)
    assert np.isfinite(empty.node_values).all() and empty.value == pytest.approx(4.0)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_batched_causal_policies_keep_native_quantum_probabilities_and_higher_derivatives(backend):
    lib = pytest.importorskip(backend)
    xp = lib if backend == "torch" else lib.numpy
    dtype = xp.float64 if backend == "torch" else xp.float32
    a, b = oq.Parameter("a"), oq.Parameter("b")
    circuit = oq.Circuit(2).ry(a, 0).ry(b, 1).cx(0, 1)
    scenarios = oq.finance.ScenarioPlan(PATHS)
    plan = scenarios.tree(NODES)
    value = xp.asarray([0.8, 1.1, 0.2, -0.1, -0.3, 0.4, 0.1, -0.2, -0.3, 0.2, -0.5], dtype=dtype)

    def objective(v):
        state = oq.statevector(circuit, {"a": v[0], "b": v[1]}, backend=backend)
        p = state.real**2 + state.imag**2
        proposal = 0.4 / (1 + xp.exp(-v[2:8])).reshape((3, 2))
        fee, impact, radius = (
            xp.log(1 + xp.exp(v[i])) * scale for i, scale in ((8, 0.001), (9, 0.01), (10, 0.001))
        )
        result = plan.rollout(
            proposal,
            p,
            previous=[0.3, 0.2],
            fees=fee,
            impact=impact,
            radius=radius,
            budget=0.1,
            risk_measure="expectile",
            confidence=0.7,
        )
        score = scenarios.energy_score(p, xp.asarray(PATHS[1], dtype=dtype))
        return result.risk.value + 0.02 * result.robust_risk.sum() + 0.2 * score

    if backend == "torch":
        value.requires_grad_()
        assert lib.autograd.gradcheck(objective, (value,))
        assert lib.autograd.gradgradcheck(objective, (value,))
        gradient = lib.autograd.grad(objective(value), value)[0].detach().numpy()
    else:
        gradient = np.asarray(lib.jit(lib.grad(objective))(value))
    assert np.isfinite(gradient).all() and (np.abs(gradient) > 1e-8).all()
    p = xp.asarray(np.stack((P, P[::-1])), dtype=dtype)
    result = plan.rollout(
        xp.asarray(W, dtype=dtype), p, budget=xp.asarray([[0.1], [0.2]], dtype=dtype)
    )
    assert result.terminal_wealth.shape == (2, 4) and result.positions.shape == (2, 7, 2)


def test_learned_transport_metric_selects_native_execution_for_host_policies():
    torch = pytest.importorskip("torch")
    diagonal = torch.tensor([[1.1, 0.9], [0.8, 1.3]], dtype=torch.float64, requires_grad=True)
    tree = oq.finance.ScenarioTreePlan(PATHS, NODES)
    result = tree.rollout(
        W, P, radius=0.001, budget=0.1, transport_metric=oq.finance.WassersteinMetric(diagonal)
    )
    assert isinstance(result.terminal_wealth, torch.Tensor)
    assert result.terminal_wealth.shape == (2, 4)
    gradient = torch.autograd.grad(result.robust_risk.sum(), diagonal)[0]
    assert torch.isfinite(gradient).all() and (gradient.abs() > 1e-8).all()


def test_tree_rejects_incompatible_information_and_preserves_bankruptcy():
    bad = PATHS.copy()
    bad[1, 0, 0] += 0.01
    with pytest.raises(oq.ValidationError, match="observed return"):
        oq.finance.ScenarioTreePlan(bad, NODES)
    with pytest.raises(oq.ResourceError):
        oq.finance.ScenarioTreePlan(PATHS, NODES, max_work=2)
    with pytest.raises(oq.ValidationError):
        oq.finance.ScenarioTreePlan(PATHS, NODES.astype(float))
    labels = NODES.copy()
    labels[-1, -1] = labels[0, -1]
    with pytest.raises(oq.ValidationError):
        oq.finance.ScenarioTreePlan(PATHS, labels)
    paths = PATHS.copy()
    paths[:2, 0] = -1
    tree = oq.finance.ScenarioTreePlan(paths, NODES)
    result = tree.rollout([[1.0, 0.0]] * 3, P, previous=[1, 0], fees=0, budget=2)
    np.testing.assert_array_equal(result.terminal_wealth[:2], 0)
    assert result.scale[1] == 0 and result.robust_risk[1] == 0
    assert np.isfinite(result.risk.node_values).all()


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_tree_checks_survive_native_compilation(backend):
    lib = pytest.importorskip(backend)
    xp = lib if backend == "torch" else lib.numpy
    tree = oq.finance.ScenarioTreePlan(PATHS, NODES)
    call = oq.compile_compute(
        lambda w: tree.rollout(w, budget=0.2),
        backend=backend,
        **({"compiler": "eager", "fullgraph": True} if backend == "torch" else {}),
    )
    value = call(xp.asarray(W))
    assert np.isfinite(np.asarray(value.risk.value)).all()
    with pytest.raises(oq.ValidationError):
        call(xp.asarray(-W))
