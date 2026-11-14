"""Horizon allocation KKT equations, temporal coupling and native implicit AD."""

import numpy as np
import pytest

import openqml as oq
from openqml.finance import FactorRisk, MultiPeriodPlan, ScenarioPlan


def _risk(horizon=3, assets=3):
    rng = np.random.default_rng(41)
    return FactorRisk(
        np.zeros((horizon, assets)),
        rng.normal(size=(horizon, 2, assets)) * 0.1,
        np.full((horizon, 2), 0.5),
        np.full((horizon, assets), 0.3),
    )


def _dense(mean, risk, previous, turnover, ridge):
    h, n = mean.shape
    matrix = np.zeros((h * n, h * n))
    for t in range(h):
        part = slice(t * n, (t + 1) * n)
        matrix[part, part] = (
            np.diag(risk.diagonal[t] + ridge)
            + risk.factors[t].T @ np.diag(risk.factor_weights[t]) @ risk.factors[t]
        )
    difference = np.eye(h) - np.eye(h, k=-1)
    matrix += turnover * np.kron(difference.T @ difference, np.eye(n))
    linear = mean.copy()
    linear[0] += turnover * previous
    constraints = np.kron(np.eye(h), np.ones((1, n)))
    kkt = np.block([[matrix, constraints.T], [constraints, np.zeros((h, h))]])
    return np.linalg.solve(kkt, np.concatenate((linear.ravel(), np.ones(h))))[: h * n].reshape(h, n)


def test_coupled_horizon_matches_independent_dense_kkt_and_uses_future_predictions():
    mean = np.array([[0.1, 0.12, 0.13], [0.14, 0.07, 0.11], [0.08, 0.15, 0.13]])
    previous, risk = np.array([0.2, 0.3, 0.5]), _risk()
    plan = MultiPeriodPlan(ridge=0.1, turnover=0.4, tolerance=1e-10)
    result = plan.solve(mean, risk, previous=previous)
    expected = _dense(mean, risk, previous, 0.4, 0.1)
    np.testing.assert_allclose(result.weights, expected, atol=1e-10)
    assert result.converged and result.residual < 1e-10
    np.testing.assert_array_equal(result.first_weights, result.weights[0])
    changed = mean.copy()
    changed[-1, 0] += 0.1
    assert (
        np.linalg.norm(
            plan.solve(changed, risk, previous=previous).first_weights - result.first_weights
        )
        > 0.001
    )
    profile = MultiPeriodPlan(max_work=100_000_000_000).resource_profile(64, 1024, 4)
    assert profile["primal_workspace_bytes"] < 256 * 2**20


def test_caps_zero_and_full_budgets_batching_and_stationary_risk():
    risk = FactorRisk(np.zeros(3), np.empty((0, 3)), np.empty(0), np.ones(3) * 0.2)
    mean = np.array([[[10, 1, -10], [10, 1, -10]], [[-10, 1, 10], [-10, 1, 10]]], dtype=float)
    result = MultiPeriodPlan(ridge=0.1).solve(mean, risk, caps=[0.6, 0.3, 0.1], budget=[0, 1])
    np.testing.assert_allclose(result.weights[:, 0], 0, atol=1e-10)
    np.testing.assert_allclose(result.weights[:, 1], [[0.6, 0.3, 0.1]] * 2, atol=1e-10)
    single = MultiPeriodPlan(ridge=0.1, turnover=0).solve(mean[:, :1], risk, caps=0.7)
    np.testing.assert_allclose(single.weights.sum(-1), 1, atol=1e-10)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_native_horizon_gradients_match_finite_changes_in_risk_and_turnover(backend):
    lib = pytest.importorskip(backend)
    xp = lib if backend == "torch" else lib.numpy
    dtype = xp.float64 if backend == "torch" else xp.float32
    mean = xp.asarray([[0.1, 0.2, 0.15], [0.2, 0.1, 0.17], [0.15, 0.13, 0.11]], dtype=dtype)
    plan = MultiPeriodPlan(ridge=0.1, tolerance=1e-7, max_iterations=80)
    baseline = _risk()
    coefficients = xp.asarray([[0.2, -0.4, 0.7], [0.4, 0.3, -0.1], [0.8, -0.3, 0.2]], dtype=dtype)

    def function(value):
        risk = FactorRisk(
            xp.asarray(baseline.mean, dtype=dtype),
            xp.asarray(baseline.factors, dtype=dtype) * value[0],
            xp.asarray(baseline.factor_weights, dtype=dtype) * value[1],
            xp.asarray(baseline.diagonal, dtype=dtype) * value[2],
        )
        result = plan.solve(
            mean, risk, previous=xp.asarray([0.3, 0.4, 0.3], dtype=dtype), turnover=value[3]
        )
        return (coefficients * result.weights).sum()

    value = xp.asarray([0.9, 1.1, 0.8, 0.2], dtype=dtype)
    if backend == "torch":
        value.requires_grad_()
        gradient = lib.autograd.grad(function(value), value)[0].detach().numpy()
        assert lib.autograd.gradcheck(function, (value,), eps=1e-5, atol=1e-5)
        assert lib.autograd.gradgradcheck(function, (value,), eps=1e-5, atol=1e-5)
    else:
        gradient = np.asarray(lib.jit(lib.grad(function))(value))
        np.testing.assert_allclose(lib.jit(function)(value), function(value), rtol=1e-5)
        assert np.isfinite(np.asarray(lib.jit(lib.jacfwd(lib.grad(function)))(value))).all()
    expected = []
    for i in range(4):
        direction = xp.asarray(np.eye(4)[i] * 0.001, dtype=dtype)
        expected.append(float((function(value + direction) - function(value - direction)) / 0.002))
    np.testing.assert_allclose(gradient, expected, rtol=0.02, atol=3e-5)


def test_torch_horizon_fullgraph_and_forward_derivative():
    torch = pytest.importorskip("torch")
    mean = torch.tensor(
        [[0.1, 0.2, 0.15], [0.2, 0.1, 0.17]], dtype=torch.float64, requires_grad=True
    )
    risk = FactorRisk(*(torch.tensor(v, dtype=mean.dtype) for v in _risk(2)))
    plan = MultiPeriodPlan(ridge=0.1, max_iterations=40, cg_iterations=20, tolerance=1e-8)

    def function(value):
        return plan.solve(value, risk).weights

    actual = torch.compile(function, backend="aot_eager", fullgraph=True)(mean)
    expected = function(mean)
    torch.testing.assert_close(actual, expected)
    tangent = torch.tensor([[0.3, 0.1, -0.4], [0.7, -0.2, 0.4]], dtype=mean.dtype)
    torch.testing.assert_close(
        torch.func.jvp(function, (mean,), (tangent,))[1],
        (function(mean + 1e-5 * tangent) - function(mean - 1e-5 * tangent)) / 2e-5,
        rtol=1e-4,
        atol=1e-7,
    )
    torch.testing.assert_close(
        torch.autograd.grad(actual.square().sum(), mean)[0],
        torch.autograd.grad(expected.square().sum(), mean)[0],
    )


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_born_path_probabilities_train_through_horizon_decision(backend):
    lib = pytest.importorskip(backend)
    xp = lib if backend == "torch" else lib.numpy
    dtype = xp.float64 if backend == "torch" else xp.float32
    scenarios = np.array(
        [
            [[0.1, -0.04], [0.04, 0.07]],
            [[-0.06, 0.09], [0.05, -0.08]],
            [[0.02, 0.05], [-0.03, 0.1]],
            [[-0.04, 0.02], [0.08, -0.03]],
        ]
    )
    scenario = ScenarioPlan(scenarios)
    allocation = MultiPeriodPlan(ridge=0.1, turnover=0.1, max_iterations=80)
    logits = xp.asarray([0.1, 0.3, -0.2, 0.4], dtype=dtype)

    def loss(value):
        probabilities = xp.exp(value) / xp.exp(value).sum()
        result = scenario.horizon_decision(probabilities, allocation=allocation)
        return (result.first_weights * xp.asarray([0.1, -0.03], dtype=dtype)).sum()

    if backend == "torch":
        logits.requires_grad_()
        gradient = lib.autograd.grad(loss(logits), logits)[0].detach().numpy()
    else:
        gradient = np.asarray(lib.jit(lib.grad(loss))(logits))
    assert np.isfinite(gradient).all() and np.linalg.norm(gradient) > 1e-5
    expected = []
    for i in range(4):
        direction = xp.asarray(np.eye(4)[i] * 0.001, dtype=dtype)
        expected.append(float((loss(logits + direction) - loss(logits - direction)) / 0.002))
    np.testing.assert_allclose(gradient, expected, rtol=0.02, atol=2e-6)


def test_horizon_rejects_infeasible_domains_and_exhausted_solver_budgets():
    mean = np.ones((2, 3))
    risk = _risk(2)
    with pytest.raises(oq.ValidationError):
        MultiPeriodPlan().solve(mean, risk, caps=0.1)
    with pytest.raises(oq.ValidationError):
        MultiPeriodPlan().solve(mean, risk, turnover=-1)
    with pytest.raises(oq.ValidationError):
        MultiPeriodPlan().solve(mean[0], risk)
    with pytest.raises(oq.ResourceError):
        MultiPeriodPlan(max_memory_mb=0.001).solve(mean, risk)
    with pytest.raises(oq.ConvergenceError):
        MultiPeriodPlan(cg_iterations=1, max_iterations=1, tolerance=1e-12).solve(
            np.array([[0.1, -0.2, 0.4], [0.3, 0.2, -0.1]]), risk
        )
