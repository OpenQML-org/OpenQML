"""Exact cash-scaling risk bounds and native scenario/probability derivatives."""

import gc
import sys

import numpy as np
import pytest

import openqml as oq


@pytest.fixture(autouse=True)
def release_compilations():
    yield
    if "jax" in sys.modules:
        sys.modules["jax"].clear_caches()
    gc.collect()


def data():
    return np.array([[-0.12, -0.03], [0.04, -0.08], [0.1, 0.02], [-0.05, 0.01]])


@pytest.mark.parametrize("confidence", [0, 0.5, 0.95])
@pytest.mark.parametrize("cash_return", [0, 0.005])
def test_projection_meets_recomputed_robust_budget_and_is_maximal(confidence, cash_return):
    weights, scenarios, p = np.array([0.4, 0.5]), data(), np.array([0.2, 0.3, 0.4, 0.1])
    result = oq.finance.risk_budget_projection(
        weights,
        scenarios,
        probabilities=p,
        confidence=confidence,
        budget=0.015,
        radius=0.0005,
        cash_return=cash_return,
    )
    losses = -scenarios @ result.weights - result.cash * cash_return
    risk = oq.finance.expected_shortfall(losses, confidence=confidence, probabilities=p)
    robust = risk + 0.0005 * np.linalg.norm(result.weights) / (1 - confidence)
    assert risk == pytest.approx(float(result.cvar), abs=2e-14)
    assert robust == pytest.approx(float(result.robust_cvar), abs=2e-14)
    assert robust <= 0.015 + 1e-12
    assert result.cash >= 0 and sum(result.weights) + result.cash == pytest.approx(1)
    assert 0 <= result.scale <= 1
    if result.scale < 1:
        larger = weights * (result.scale + 1e-5)
        losses = -scenarios @ larger - (1 - larger.sum()) * cash_return
        bound = oq.finance.expected_shortfall(losses, confidence=confidence, probabilities=p)
        bound += 0.0005 * np.linalg.norm(larger) / (1 - confidence)
        assert bound > 0.015


def test_safe_zero_risk_negative_losses_broadcast_radii_and_cash_only():
    f = oq.finance.risk_budget_projection
    np.testing.assert_array_equal(f([0.3, 0.2], np.zeros((3, 2))).weights, [0.3, 0.2])
    np.testing.assert_array_equal(
        f([0.3, 0.2], np.ones((3, 2)) * 0.01, budget=0).weights, [0.3, 0.2]
    )
    assert f([0.0, 0.0], data(), budget=0).cash == 1
    weights = np.array([[0.4, 0.5], [0.2, 0.3]])
    result = f(weights, data(), radius=np.array([0.001, 0.002]), budget=np.array([0.02, 0.03]))
    assert result.weights.shape == (2, 2)
    assert np.all(result.robust_cvar <= result.budget + 1e-12)


@pytest.mark.parametrize(
    "settings",
    [
        {"confidence": 1},
        {"radius": -0.1},
        {"budget": -0.01},
        {"weights": [-0.1, 0.3]},
        {"weights": [0.8, 0.8]},
        {"scenarios": [[1, 2, 3]]},
        {"probabilities": [0, 0, 0, 0]},
        {"max_memory_mb": 1e-8},
    ],
)
def test_invalid_risk_domains_and_memory_are_rejected(settings):
    args = dict(weights=[0.4, 0.5], scenarios=data())
    args.update(settings)
    with pytest.raises((oq.ValidationError, oq.ResourceError)):
        oq.finance.risk_budget_projection(**args)


def test_torch_probability_scenario_budget_and_contextual_radius_higher_derivatives():
    torch = pytest.importorskip("torch")
    weights = torch.tensor([0.4, 0.5], dtype=torch.float64, requires_grad=True)
    scenarios = torch.tensor(data(), dtype=torch.float64, requires_grad=True)
    probability = torch.tensor([0.2, 0.3, 0.4, 0.1], dtype=torch.float64, requires_grad=True)
    radius = torch.tensor(0.0005, dtype=torch.float64, requires_grad=True)
    budget = torch.tensor(0.015, dtype=torch.float64, requires_grad=True)

    def fn(w, r, p, rho, b):
        return oq.finance.risk_budget_projection(
            w, r, probabilities=p, radius=rho, budget=b, confidence=0.6
        ).weights

    assert torch.autograd.gradcheck(
        fn, (weights, scenarios, probability, radius, budget), atol=1e-5
    )
    assert torch.autograd.gradgradcheck(
        fn, (weights, scenarios, probability, radius, budget), atol=3e-5
    )


def test_jax_jit_and_gradients_match_torch_including_contextual_radius():
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    values = (
        np.array([0.4, 0.5]),
        data(),
        np.array([0.2, 0.3, 0.4, 0.1]),
        np.array(0.0005),
        np.array(0.015),
    )

    def objective(w, r, p, rho, b):
        result = oq.finance.risk_budget_projection(
            w, r, probabilities=p, radius=rho, budget=b, confidence=0.6
        )
        return (result.weights**2).sum()

    tensors = tuple(torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in values)
    expected = torch.autograd.grad(objective(*tensors), tensors)
    received = jax.jit(jax.grad(objective, argnums=(0, 1, 2, 3, 4)))(
        *(jnp.asarray(v) for v in values)
    )
    for a, b in zip(expected, received, strict=True):
        np.testing.assert_allclose(a.detach(), b, atol=5e-6)


def test_factor_space_woodbury_matches_dense_with_zero_mass_and_native_hessians():
    from openqml.finance._array import Ops
    from openqml.finance._linear import linear_solve

    rng = np.random.default_rng(12)
    d = rng.uniform(0.2, 1, size=16)
    f = rng.normal(size=(3, 16)) * 0.1
    p = np.array([0.0, 0.3, 0.7])
    rhs = rng.normal(size=(16, 2))
    matrix = np.diag(d) + f.T @ np.diag(p) @ f
    actual = linear_solve(d, f, p, rhs, 1, 1e-10, Ops(d))
    np.testing.assert_allclose(actual, np.linalg.solve(matrix, rhs), atol=2e-14)
    torch = pytest.importorskip("torch")
    variables = tuple(
        torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in (d, f, p, rhs)
    )

    def solve(diagonal, factors, mass, target):
        return linear_solve(diagonal, factors, mass, target, 1, 1e-10, Ops(diagonal))

    assert torch.autograd.gradcheck(solve, variables, atol=1e-5)
    assert torch.autograd.gradgradcheck(solve, variables, atol=3e-5)
