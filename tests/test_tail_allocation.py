import numpy as np
import pytest

from openqml import ResourceError, ValidationError
from openqml.finance import TailAllocationPlan

PATHS = np.array(
    [
        [[0.12, 0.03], [-0.04, 0.02]],
        [[-0.18, 0.04], [0.08, -0.03]],
        [[0.07, -0.08], [-0.13, 0.06]],
        [[0.02, 0.05], [0.04, 0.09]],
    ]
)


def _plan(**options):
    return TailAllocationPlan(confidence=0.8, smoothing=0.06, tolerance=2e-9, **options)


def test_path_tail_matches_independent_constrained_optimizer():
    scipy = pytest.importorskip("scipy.optimize")
    plan = _plan()
    probability = np.array([0.15, 0.25, 0.35, 0.25])
    caps = np.array([[0.8, 0.9], [0.7, 0.85]])
    budgets = np.array([0.9, 0.8])
    previous = np.array([0.4, 0.3])
    radius = 0.002
    result = plan.solve(
        PATHS,
        probabilities=probability,
        caps=caps,
        budget=budgets,
        previous=previous,
        radius=radius,
    )

    def objective(x):
        weights, threshold = x[:-1].reshape(2, 2), x[-1]
        losses = -np.einsum("shn,hn->s", PATHS, weights)
        soft = threshold + np.sum(
            probability * plan.smoothing * np.logaddexp(0, (losses - threshold) / plan.smoothing)
        ) / (1 - plan.confidence)
        soft += radius * np.linalg.norm(weights) / (1 - plan.confidence)
        changes = weights - np.concatenate((previous[None], weights[:-1]))
        return (
            plan.risk_aversion * soft
            + probability @ losses
            + plan.ridge * np.sum(weights**2) / 2
            + plan.turnover * np.sum(changes**2) / 2
            - plan.barrier * np.sum(np.log(weights) + np.log(caps - weights))
        )

    optimum = scipy.minimize(
        objective,
        np.array([0.45, 0.45, 0.4, 0.4, 0.1]),
        method="SLSQP",
        bounds=[(1e-7, cap - 1e-7) for cap in caps.flat] + [(None, None)],
        constraints={"type": "eq", "fun": lambda x: x[:-1].reshape(2, 2).sum(-1) - budgets},
        options={"ftol": 1e-13, "maxiter": 1000},
    )
    assert optimum.success
    np.testing.assert_allclose(result.weights, optimum.x[:-1].reshape(2, 2), atol=2e-6)
    np.testing.assert_allclose(result.objective, optimum.fun, atol=1e-11)
    np.testing.assert_allclose(result.weights.sum(-1), budgets, atol=2e-15)
    assert result.residual < 2e-9 and result.converged
    assert result.cvar <= result.robust_cvar <= result.smooth_risk
    assert result.smooth_risk - result.robust_cvar <= (
        plan.smoothing * np.log(2) / (1 - plan.confidence) + 1e-10
    )


def test_joint_temporal_dependence_changes_allocation():
    plan = _plan()
    shuffled = PATHS.copy()
    shuffled[:, 1] = shuffled[[2, 0, 3, 1], 1]
    np.testing.assert_allclose(PATHS.mean(0), shuffled.mean(0))
    original, changed = plan.solve(PATHS), plan.solve(shuffled)
    assert np.linalg.norm(original.weights - changed.weights) > 0.01


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_path_tail_batched_and_zero_probability_atoms(backend):
    if backend == "torch":
        torch = pytest.importorskip("torch")

        def array(x):
            return torch.as_tensor(x, dtype=torch.float64)
    elif backend == "jax":
        pytest.importorskip("jax")
        import jax.numpy as jnp

        array = jnp.asarray
    else:
        array = np.asarray
    plan = TailAllocationPlan(confidence=0.8, smoothing=0.06)
    returns = array(np.stack((PATHS, PATHS * 1.4)))
    p = array([[0.2, 0.3, 0.5, 0], [0.2, 0.3, 0.1, 0.4]])
    budgets = array([[0.8, 0.9], [0.7, 0.6]])
    batched = plan.solve(returns, probabilities=p, budget=budgets, radius=array([0, 0.002]))
    for i in range(2):
        separate = plan.solve(
            returns[i], probabilities=p[i], budget=budgets[i], radius=(0, 0.002)[i]
        )
        np.testing.assert_allclose(batched.weights[i], separate.weights, atol=3e-5)
    np.testing.assert_allclose(np.asarray(batched.weights).sum(-1), budgets, atol=2e-7)


def test_torch_path_tail_implicit_derivatives_and_jvp():
    torch = pytest.importorskip("torch")
    plan = _plan()
    data = (
        PATHS,
        [0.15, 0.25, 0.35, 0.25],
        [[0.8, 0.9], [0.7, 0.85]],
        [0.9, 0.8],
        [0.4, 0.3],
        0.02,
        0.002,
        0.06,
    )
    values = tuple(torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in data)

    def decision(r, p, c, b, previous, turnover, radius, smoothing):
        return plan.solve(
            r,
            probabilities=p,
            caps=c,
            budget=b,
            previous=previous,
            turnover=turnover,
            radius=radius,
            smoothing=smoothing,
        ).weights

    assert torch.autograd.gradcheck(decision, values, fast_mode=True, atol=2e-5, rtol=2e-4)
    assert torch.autograd.gradgradcheck(decision, values, fast_mode=True, atol=2e-4, rtol=3e-3)
    direction = tuple(torch.ones_like(x) * 0.001 for x in values)
    _, tangent = torch.func.jvp(decision, values, direction)
    delta = 1e-3
    finite = (
        decision(*(x + delta * d for x, d in zip(values, direction, strict=True)))
        - decision(*(x - delta * d for x, d in zip(values, direction, strict=True)))
    ) / (2 * delta)
    torch.testing.assert_close(tangent, finite, atol=2e-7, rtol=2e-4)


def test_jax_path_tail_jit_gradient_and_hessian():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    plan = TailAllocationPlan(confidence=0.8, smoothing=0.06)
    probability = jnp.array([0.15, 0.25, 0.35, 0.25])

    def loss(p):
        return jnp.sum(plan.solve(jnp.asarray(PATHS), probabilities=p, radius=0.002).weights ** 2)

    gradient = jax.jit(jax.grad(loss))(probability)
    hessian = jax.jit(jax.hessian(loss))(probability)
    assert np.all(np.isfinite(gradient)) and np.all(np.isfinite(hessian))
    delta = 1e-3
    expected = np.array(
        [
            (loss(probability.at[i].add(delta)) - loss(probability.at[i].add(-delta))) / (2 * delta)
            for i in range(4)
        ]
    )
    np.testing.assert_allclose(gradient, expected, atol=2e-4, rtol=3e-3)
    np.testing.assert_allclose(hessian, np.asarray(hessian).T, atol=1e-4)


def test_born_path_tail_decision_gradient():
    torch = pytest.importorskip("torch")
    from openqml.finance import ScenarioPlan

    scenarios = ScenarioPlan(PATHS)
    logits = torch.tensor([0.1, -0.3, 0.2, 0.6], dtype=torch.float64, requires_grad=True)
    decision = scenarios.tail_decision(torch.softmax(logits, -1), allocation=_plan())
    gradient = torch.autograd.grad(decision.weights.square().sum(), logits)[0]
    assert torch.isfinite(gradient).all() and gradient.norm() > 0


def test_path_tail_domains_and_admission():
    plan = _plan()
    for options in (
        {"caps": 0},
        {"budget": 0},
        {"budget": 2},
        {"radius": -0.1},
        {"smoothing": 0},
        {"turnover": -1},
    ):
        with pytest.raises(ValidationError):
            plan.solve(PATHS, **options)
    with pytest.raises(ResourceError):
        TailAllocationPlan(max_memory_mb=0.001).solve(PATHS)
    with pytest.raises(ResourceError):
        TailAllocationPlan(max_work=10).solve(PATHS)
    assert plan.resource_profile(4, 2, 2)["solver"].endswith("no_dense_KKT")
