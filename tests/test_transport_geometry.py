"""Anisotropic robust decisions retain duality, calibration inputs and native AD."""

import numpy as np
import pytest

import openqml as oq


@pytest.mark.parametrize("assets,rank", [(5, 2), (16, 2), (3, 0)])
def test_factor_transport_dual_norm_matches_dense_solve_and_support_function(assets, rank):
    rng = np.random.default_rng(41)
    w = rng.normal(size=assets)
    diagonal = rng.uniform(0.5, 2, size=assets)
    factors = rng.normal(size=(rank, assets)) / 3
    metric = oq.finance.WassersteinMetric(diagonal, factors if rank else None)
    actual = oq.finance.transport_dual_norm(w, metric, tolerance=1e-11)
    matrix = np.diag(diagonal) + factors.T @ factors
    solution = np.linalg.solve(matrix, w)
    expected = np.sqrt(w @ solution)
    assert actual == pytest.approx(expected, rel=2e-10)
    maximizing_delta = solution / actual
    assert maximizing_delta @ matrix @ maximizing_delta == pytest.approx(1, abs=2e-10)
    assert w @ maximizing_delta == pytest.approx(actual, abs=2e-10)


def test_contextual_metrics_and_native_radii_broadcast_into_maximal_cash_scaling():
    f = oq.finance
    proposal = np.array([0.4, 0.5])
    scenarios = np.array([[-0.12, -0.03], [0.04, -0.08], [0.1, 0.02], [-0.05, 0.01]])
    diagonal = np.array([[0.5, 2.0], [3.0, 0.8]])
    factors = np.array([[0.2, -0.4]])
    metric = f.WassersteinMetric(diagonal, factors)
    result = f.risk_budget_projection(
        proposal, scenarios, confidence=0.5, budget=0.015, radius=0.01, transport_metric=metric
    )
    assert result.weights.shape == (2, 2)
    for i in range(2):
        local = f.WassersteinMetric(diagonal[i], factors)
        risk = f.wasserstein_cvar(
            result.weights[i], scenarios, confidence=0.5, radius=0.01, transport_metric=local
        )
        assert risk == pytest.approx(float(result.robust_cvar[i]), abs=2e-8)
        assert risk <= 0.015 + 2e-8
        assert float(result.scale[i]) < 1
        larger = proposal * (result.scale[i] + 1e-4)
        assert (
            f.wasserstein_cvar(
                larger, scenarios, confidence=0.5, radius=0.01, transport_metric=local
            )
            > 0.015
        )


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_born_probabilities_metric_and_radius_remain_jointly_differentiable(backend):
    lib = pytest.importorskip(backend)
    xp = lib if backend == "torch" else lib.numpy
    dtype = xp.float64 if backend == "torch" else xp.float32
    values = xp.asarray([1.4, 0.7, 1.2, 0.2, -0.3, 0.01], dtype=dtype)
    f, circuit = oq.finance, oq.Circuit(1).ry(oq.Parameter("theta"), 0)
    scenarios = np.array([[-0.1, 0.02], [0.04, -0.03]])

    def objective(v):
        state = oq.statevector(circuit, {"theta": v[0]}, backend=backend)
        probabilities = state.real**2 + state.imag**2
        metric = f.WassersteinMetric(v[1:3], v[3:5].reshape((1, 2)))
        decision = f.risk_budget_projection(
            [0.45, 0.4],
            scenarios,
            probabilities=probabilities,
            confidence=0.25,
            budget=0.01,
            radius=v[5],
            transport_metric=metric,
        )
        return -decision.weights[0] * 0.03 + decision.weights[1] * 0.01

    if backend == "torch":
        values.requires_grad_()
        derivative = lib.autograd.grad(objective(values), values)[0].detach().numpy()
    else:
        derivative = np.asarray(lib.jit(lib.grad(objective))(values))
    expected, h = [], 1e-4
    for i in range(6):
        direction = np.eye(6)[i] * h
        base = values.detach().numpy() if backend == "torch" else np.asarray(values)
        plus, minus = (
            xp.asarray(base + direction, dtype=dtype),
            xp.asarray(base - direction, dtype=dtype),
        )
        left, right = objective(plus), objective(minus)
        left = float(left.detach()) if backend == "torch" else float(left)
        right = float(right.detach()) if backend == "torch" else float(right)
        expected.append((left - right) / (2 * h))
    assert np.all(np.isfinite(derivative)) and np.all(np.abs(derivative) > 1e-7)
    np.testing.assert_allclose(derivative, expected, rtol=5e-3, atol=3e-6)


def test_transport_implicit_derivatives_and_zero_subgradient_are_finite():
    torch = pytest.importorskip("torch")
    w = torch.tensor([0.3, -0.2, 0.4], dtype=torch.float64, requires_grad=True)
    diagonal = torch.tensor([0.7, 1.1, 0.9], dtype=torch.float64, requires_grad=True)
    factors = torch.tensor([[0.2, -0.3, 0.1]], dtype=torch.float64, requires_grad=True)

    def run(a, d, f):
        return oq.finance.transport_dual_norm(a, oq.finance.WassersteinMetric(d, f))

    assert torch.autograd.gradcheck(run, (w, diagonal, factors))
    assert torch.autograd.gradgradcheck(run, (w, diagonal, factors))
    zero = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    assert float(run(zero, diagonal, factors).detach()) == 0
    gradient = torch.autograd.grad(run(zero, diagonal, factors), (zero, diagonal, factors))
    assert all(torch.isfinite(g).all() and not bool(g.any()) for g in gradient)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_compiled_transport_retains_metric_domain_checks(backend):
    lib = pytest.importorskip(backend)
    xp = lib if backend == "torch" else lib.numpy
    w = xp.asarray([0.3, 0.5])
    norm, metric = oq.finance.transport_dual_norm, oq.finance.WassersteinMetric
    run = oq.compile_compute(
        lambda d: norm(w, metric(d)),
        backend=backend,
        **({"compiler": "eager", "fullgraph": True} if backend == "torch" else {}),
    )
    np.testing.assert_allclose(
        np.asarray(run(xp.asarray([1.0, 4.0]))), np.hypot(0.3, 0.25), rtol=2e-6
    )
    with pytest.raises(oq.ValidationError):
        run(xp.asarray([1.0, -1.0]))


@pytest.mark.parametrize(
    "diagonal,factors", [([1, 0], None), ([1, -1], None), ([1], None), ([1, 1], [0.2, 0.3])]
)
def test_invalid_metrics_are_rejected(diagonal, factors):
    with pytest.raises(oq.ValidationError):
        oq.finance.transport_dual_norm([0.3, 0.4], oq.finance.WassersteinMetric(diagonal, factors))


def test_transport_admits_work_before_broadcast_materialization():
    with pytest.raises(oq.ResourceError):
        oq.finance.transport_dual_norm(
            np.ones(1000), oq.finance.WassersteinMetric(np.ones(1000)), max_memory_mb=0.001
        )
