"""Finite-support entropic risk, funding and derivatives against independent references."""

import math

import numpy as np
import pytest

import openqml as oq


def test_binary_kl_reference_and_funded_execution():
    p, confidence = 0.08, 0.6
    lo, hi = p, 1.0
    rho = -math.log1p(-confidence)
    for _ in range(75):
        w = (lo + hi) / 2
        kl = w * math.log(w / p) + (1 - w) * math.log((1 - w) / (1 - p))
        if kl > rho:
            hi = w
        else:
            lo = w
    worst = np.array([lo, 1 - lo])
    returns = np.array([[-0.25], [0.08]])
    result = oq.finance.entropic_value_at_risk(
        -returns[:, 0], probabilities=[p, 1 - p], confidence=confidence
    )
    np.testing.assert_allclose(result.worst_probabilities, worst, atol=2e-14)
    np.testing.assert_allclose(result.value, worst @ -returns[:, 0], atol=2e-14)
    np.testing.assert_allclose(result.relative_entropy, rho, atol=1e-13)
    paths = oq.finance.ScenarioPlan(returns)
    decision = paths.execution_decision(
        [p, 1 - p],
        [1.0],
        fees=0.003,
        impact=0.01,
        budget=0.03,
        risk_measure="evar",
        confidence=confidence,
    )
    assert 0 < decision.scale < 1
    np.testing.assert_allclose(decision.cash + decision.positions.sum() + decision.costs, 1)
    np.testing.assert_allclose(worst @ (decision.costs - returns @ decision.positions), 0.03)
    with pytest.raises(oq.ValidationError, match="radius zero"):
        paths.execution_decision([p, 1 - p], [1.0], risk_measure="evar", radius=0.1)


def test_saturation_support_normalization_and_rectangular_tree():
    x = np.array([2.0, 2.0, -7.0, 1e300])
    p = np.array([[1.0, 3.0, 6.0, 0.0], [0.0, 0.0, 8.0, 0.0]])
    peak = oq.finance.entropic_value_at_risk(x, probabilities=p, confidence=0.9)
    np.testing.assert_allclose(peak.value, [2.0, -7.0])
    np.testing.assert_allclose(peak.worst_probabilities, [[0.25, 0.75, 0, 0], [0, 0, 1, 0]])
    np.testing.assert_allclose(peak.relative_entropy, [-np.log(0.4), 0.0])
    mean = oq.finance.entropic_value_at_risk(x, probabilities=p, confidence=0)
    np.testing.assert_allclose(mean.value, [-3.4, -7])
    np.testing.assert_allclose(mean.relative_entropy, 0)
    tree = oq.finance.ScenarioTreePlan(
        [[[-0.1], [-0.2]], [[-0.1], [0.2]], [[0.1], [0.3]], [[0.1], [-0.2]]],
        [[0, 1, 0], [0, 1, 1], [0, 2, 2], [0, 2, 3]],
    )
    losses, mass = [1, 3, 7, -1], [0.2, 0.3, 0.4, 0.1]
    nested = tree.nested_risk(losses, mass, risk_measure="evar", confidence=0.18)

    def risk(x, p):
        return oq.finance.entropic_value_at_risk(x, probabilities=p, confidence=0.18).value

    np.testing.assert_allclose(
        nested.value, risk([risk(losses[:2], mass[:2]), risk(losses[2:], mass[2:])], [0.5, 0.5])
    )
    with pytest.raises(oq.ValidationError):
        oq.finance.entropic_value_at_risk([1.0, 2.0], probabilities=[0.0, 0.0])
    with pytest.raises(oq.ResourceError):
        oq.finance.entropic_value_at_risk(np.ones(100), max_memory_mb=0.001)


def test_torch_risk_envelope_and_implicit_curvature():
    torch = pytest.importorskip("torch")
    x = torch.tensor([0.12, -0.2, 0.04], dtype=torch.double, requires_grad=True)
    p = torch.tensor([0.12, 0.5, 0.38], dtype=torch.double, requires_grad=True)

    def risk(x, p):
        return oq.finance.entropic_value_at_risk(x, probabilities=p, confidence=0.25)

    result = risk(x, p)
    torch.testing.assert_close(torch.autograd.grad(result.value, x)[0], result.worst_probabilities)
    assert torch.autograd.gradgradcheck(lambda x, p: risk(x, p).value, (x, p))
    assert torch.autograd.gradcheck(lambda x, p: risk(x, p).worst_probabilities, (x, p))


def test_jitted_born_distribution_gradient_matches_finite_differences():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    c = oq.Circuit(1).ry(oq.Parameter("a"), 0)
    z = oq.PauliTransferPlan(c, (oq.PauliString("Z"),))

    def risk(a, backend):
        z_value = z.evaluate({"a": a}, backend=backend)[..., 0]
        xp = jnp if backend == "jax" else np
        p = xp.stack(((1 + z_value) / 2, (1 - z_value) / 2))
        return oq.finance.entropic_value_at_risk(
            [-0.1, 0.3], probabilities=p, confidence=0.22
        ).value

    gradient = jax.jit(jax.grad(lambda a: risk(a, "jax")))(jnp.asarray(0.9))
    reference = (risk(0.90001, "numpy") - risk(0.89999, "numpy")) / 0.00002
    np.testing.assert_allclose(gradient, reference, atol=2e-6)
    jax.clear_caches()


def test_asset_attribution_sums_to_risk_and_deviation_for_batched_paths():
    paths = np.array(
        [
            [[-0.3, 0.1], [0.2, -0.3]],
            [[0.18, -0.11], [-0.3, 0.05]],
            [[0.2, 0.1], [0.08, 0.07]],
            [[-0.07, 0.08], [0.09, -0.18]],
        ]
    )
    plan = oq.finance.ScenarioPlan(paths)
    p = np.array([[0.2, 0.3, 0.4, 0.1], [0.4, 0.1, 0.2, 0.3]])
    weights = np.array([0.7, -0.2])
    for step in (None, 1):
        returns = plan.returns(step=step)

        def attribute(w, step=step):
            return plan.entropic_attribution(p, w, step=step, confidence=0.27)

        result = attribute(weights)
        np.testing.assert_allclose(result.contributions.sum(axis=-1), result.risk.value)
        nominal = -p @ returns @ weights
        np.testing.assert_allclose(
            result.deviation_contributions.sum(axis=-1), result.risk.value - nominal
        )
        for asset, bump in enumerate(np.eye(2) * 1e-6):
            derivative = (
                attribute(weights + bump).risk.value - attribute(weights - bump).risk.value
            ) / 2e-6
            np.testing.assert_allclose(result.marginal[:, asset], derivative, atol=1e-9)


def test_rare_extreme_low_loss_preserves_distinct_supported_maxima():
    confidence = 0.1
    p = np.array([0.001, 0.549, 0.45])
    risk = oq.finance.entropic_value_at_risk(
        [-1e308, 0, 1e-100], probabilities=p, confidence=confidence
    )
    binary = oq.finance.entropic_value_at_risk(
        [0, 1], probabilities=p[1:], confidence=1 - (1 - confidence) / p[1:].sum()
    )
    np.testing.assert_allclose(
        risk.worst_probabilities, [0, *binary.worst_probabilities], rtol=2e-13
    )
    np.testing.assert_allclose(risk.value / 1e-100, binary.value, rtol=2e-13)
