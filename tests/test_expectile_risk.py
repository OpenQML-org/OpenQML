"""Reference root, ambiguity witnesses and native end-to-end risk derivatives."""

import numpy as np
import pytest

import openqml as oq


def reference(losses, probabilities, level, margin=0):
    p = np.asarray(probabilities, dtype=float)
    p = p / p.sum()
    lo, hi = min(losses) - 1, max(losses) + 1 + level * margin / (1 - level)
    for _ in range(150):
        mid = (lo + hi) / 2
        residual = (
            level * np.dot(p, np.maximum(losses - mid, 0))
            - (1 - level) * np.dot(p, np.maximum(mid - losses, 0))
            + level * margin
        )
        if residual > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


@pytest.mark.parametrize("level", [0.5, 0.8, 0.98])
@pytest.mark.parametrize("radius", [0, 0.001, 0.2])
def test_sorted_nominal_and_robust_roots_match_independent_bisection(level, radius):
    f = oq.finance
    losses = np.array([0.1, -0.2, 0.4, 0.1, -0.3])
    p = np.array([0.1, 0, 0.3, 0.2, 0.4])
    nominal = f.expectile(losses, level=level, probabilities=p)
    robust = f.wasserstein_expectile(
        [1.0], -losses[:, None], level=level, probabilities=p, radius=radius
    )
    assert nominal == pytest.approx(reference(losses, p, level), abs=1e-13)
    assert robust == pytest.approx(reference(losses, p, level, radius), abs=1e-13)
    assert robust >= nominal - 1e-13
    assert f.expectile(3 * losses + 2, level=level, probabilities=p) == pytest.approx(
        3 * nominal + 2
    )


def test_wasserstein_expectile_is_approached_by_an_explicit_adversarial_distribution():
    # Move epsilon mass from the origin to radius/epsilon. This uses exactly
    # the stated W1 budget, and its expectile tends to the returned supremum.
    f, level, radius = oq.finance, 0.8, 0.02
    robust = f.wasserstein_expectile([1.0], [[0.0]], level=level, radius=radius)
    witness = f.expectile([0.0, radius / 1e-7], level=level, probabilities=[1 - 1e-7, 1e-7])
    assert witness <= robust
    assert robust == pytest.approx(level * radius / (1 - level), abs=1e-15)
    assert witness == pytest.approx(robust, rel=1e-6)


def test_contextual_anisotropic_risk_and_funded_scaling_are_maximal():
    f = oq.finance
    w = np.array([0.4, 0.5])
    scenarios = np.array([[-0.12, -0.03], [0.04, -0.08], [0.1, 0.02], [-0.05, 0.01]])
    p = np.array([0.2, 0.3, 0.4, 0.1])
    diagonal = np.array([[0.7, 1.2], [1.1, 0.8]])
    factors = np.array([[[0.2, -0.3]], [[0.1, 0.2]]])
    metric = f.WassersteinMetric(diagonal, factors)
    decision = f.expectile_budget_projection(
        w,
        scenarios,
        probabilities=p,
        level=0.8,
        budget=0.01,
        radius=0.002,
        cash_return=0.003,
        transport_metric=metric,
    )
    assert decision.weights.shape == (2, 2)
    for i in range(2):
        matrix = np.diag(diagonal[i]) + factors[i].T @ factors[i]
        dual = np.sqrt(decision.weights[i] @ np.linalg.solve(matrix, decision.weights[i]))
        losses = -scenarios @ decision.weights[i] - decision.cash[i] * 0.003
        exact = reference(losses, p, 0.8, 0.002 * dual)
        assert exact == pytest.approx(decision.robust_expectile[i], abs=1e-14)
        assert exact <= 0.01 + 1e-14
        larger = w * (decision.scale[i] + 1e-6)
        above = reference(
            -scenarios @ larger - (1 - larger.sum()) * 0.003,
            p,
            0.8,
            0.002 * np.sqrt(larger @ np.linalg.solve(matrix, larger)),
        )
        assert above > 0.01


def _objective(values, backend):
    f = oq.finance
    circuit = oq.Circuit(1).ry(oq.Parameter("theta"), 0)
    state = oq.statevector(circuit, {"theta": values[0]}, backend=backend)
    p = state.real**2 + state.imag**2
    metric = f.WassersteinMetric(values[1:3], values[3:5].reshape((1, 2)))
    decision = f.expectile_budget_projection(
        [0.4, 0.5],
        [[-0.12, -0.03], [0.04, -0.08]],
        level=0.8,
        probabilities=p,
        budget=0.01,
        radius=values[5],
        transport_metric=metric,
    )
    return -0.03 * decision.weights[0] + 0.01 * decision.weights[1]


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_quantum_probabilities_radius_and_metric_have_native_decision_gradients(backend):
    lib = pytest.importorskip(backend)
    xp = lib if backend == "torch" else lib.numpy
    dtype = xp.float64 if backend == "torch" else xp.float32
    values = xp.asarray([1.4, 0.7, 1.2, 0.2, -0.3, 0.01], dtype=dtype)
    if backend == "torch":
        values.requires_grad_()
        assert lib.autograd.gradcheck(lambda v: _objective(v, backend), (values,))
        assert lib.autograd.gradgradcheck(lambda v: _objective(v, backend), (values,))
        result = lib.autograd.grad(_objective(values, backend), values)[0].detach().numpy()
    else:
        result = np.asarray(lib.jit(lib.grad(lambda v: _objective(v, backend)))(values))
    base = values.detach().numpy() if backend == "torch" else np.asarray(values)

    def independent_objective(v):
        p = np.array([np.cos(v[0] / 2) ** 2, np.sin(v[0] / 2) ** 2])
        w = np.array([0.4, 0.5])
        scenarios = np.array([[-0.12, -0.03], [0.04, -0.08]])
        matrix = np.diag(v[1:3]) + np.outer(v[3:5], v[3:5])
        margin = v[5] * np.sqrt(w @ np.linalg.solve(matrix, w))
        robust = reference(-scenarios @ w, p, 0.8, margin)
        return (-0.03 * w[0] + 0.01 * w[1]) * min(1, 0.01 / robust)

    expected = []
    for i in range(6):
        # Differentiate an independent double-precision reference; float32
        # subtraction of nearly equal native losses hides small factor gradients.
        delta = np.eye(6)[i] * 1e-5
        expected.append(
            (independent_objective(base + delta) - independent_objective(base - delta)) / 2e-5
        )
    assert np.isfinite(result).all() and (np.abs(result) > 1e-8).all()
    np.testing.assert_allclose(result, expected, rtol=2e-3, atol=2e-8)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_compiled_risk_retains_probability_and_radius_checks(backend):
    lib = pytest.importorskip(backend)
    xp = lib if backend == "torch" else lib.numpy
    risk = oq.finance.wasserstein_expectile
    compiled = oq.compile_compute(
        lambda p, r: risk(
            xp.asarray([0.4, 0.5]),
            xp.asarray([[-0.12, -0.03], [0.04, -0.08]]),
            probabilities=p,
            radius=r,
            level=0.8,
        ),
        backend=backend,
        **({"compiler": "eager", "fullgraph": True} if backend == "torch" else {}),
    )
    assert np.isfinite(np.asarray(compiled(xp.asarray([0.4, 0.6]), xp.asarray(0.01))))
    with pytest.raises(oq.ValidationError):
        compiled(xp.asarray([0.4, -0.6]), xp.asarray(0.01))
    with pytest.raises(oq.ValidationError):
        compiled(xp.asarray([0.4, 0.6]), xp.asarray(-0.01))


def test_invalid_risk_domains_and_small_memory_fail_explicitly():
    f = oq.finance
    for options in (
        {"level": 0.4},
        {"level": 1},
        {"probabilities": [0, 0]},
        {"max_memory_mb": 1e-9},
    ):
        with pytest.raises((oq.ValidationError, oq.ResourceError)):
            f.expectile([0.1, 0.2], **options)
    with pytest.raises(oq.ValidationError):
        f.expectile([])
    with pytest.raises(oq.ValidationError):
        f.wasserstein_expectile([1.0], [[0.1], [0.2]], radius=-1)
    assert f.expectile([1e250, 1e250], level=0.99) == pytest.approx(1e250)
    assert f.expectile([0.0, 0.0], level=0.99) == 0
    root = f.expectile([1e-200, 1e200], level=0.99, probabilities=[1.0, 0.0])
    assert root / 1e-200 == pytest.approx(1.0)
    root = f.expectile([1e-310], level=1 - 1e-15)
    assert root / 1e-310 == pytest.approx(1.0)
