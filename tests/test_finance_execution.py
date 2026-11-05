"""Exact tail selection, native gradients, and independent GBM barrier prices."""

import gc
import math
from dataclasses import replace

import numpy as np
import pytest

import openqml as oq

f = oq.finance


def _tail_reference(values, confidence):
    remaining, result = 1 - confidence, 0.0
    for loss in sorted(values, reverse=True):
        take = min(remaining, 1 / len(values))
        result += take * loss
        remaining -= take
        if remaining <= 1e-15:
            break
    return result / (1 - confidence)


@pytest.mark.parametrize("confidence", [0, 0.1, 0.5, 0.8, 0.9, 0.95, 0.999])
def test_partial_selection_matches_independent_fractional_mass_reference(confidence):
    values = np.array([[3, -2, 5, 4, 1, 5, -1], [-3, -2, -1, 0, 1, 2, 3]])
    expected = [_tail_reference(row, confidence) for row in values]
    np.testing.assert_allclose(
        f.expected_shortfall(values, confidence=confidence), expected, atol=1e-14
    )


def test_uniform_tail_does_not_sort_and_is_translation_equivariant(monkeypatch):
    losses = np.random.default_rng(21).normal(size=(3, 113))
    expected = f.expected_shortfall(losses, confidence=0.93)
    monkeypatch.setattr(np, "argsort", lambda *a, **k: pytest.fail("full sort was used"))
    np.testing.assert_array_equal(expected, f.expected_shortfall(losses, confidence=0.93))
    np.testing.assert_allclose(f.expected_shortfall(losses + 17, confidence=0.93), expected + 17)


@pytest.fixture
def native():
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield torch, jax
    finally:
        jax.config.update("jax_enable_x64", previous)
        jax.clear_caches()
        gc.collect()


def test_uniform_selection_native_values_and_gradients_equal_weighted_sort(native):
    torch, jax = native
    values = np.random.default_rng(32).normal(size=(2, 17))
    for p in (None, np.ones(17)):
        x = torch.tensor(values, requires_grad=True)
        result = f.expected_shortfall(x, confidence=0.83, probabilities=p)
        gradient = torch.autograd.grad(result.sum(), x)[0].numpy()

        def objective(v, probability=p):
            return f.expected_shortfall(v, confidence=0.83, probabilities=probability).sum()

        jvalue, jgradient = jax.jit(jax.value_and_grad(objective))(jax.numpy.asarray(values))
        np.testing.assert_allclose(float(result.sum().detach()), jvalue, atol=1e-12)
        np.testing.assert_allclose(gradient, jgradient, atol=1e-12)
        if p is None:
            reference_gradient = gradient
        else:
            np.testing.assert_allclose(gradient, reference_gradient, atol=1e-12)
    tied = torch.ones(11, dtype=torch.float64, requires_grad=True)
    gradient = torch.autograd.grad(f.expected_shortfall(tied, confidence=0.8), tied)[0]
    assert float(gradient.sum()) == pytest.approx(1)
    assert bool((gradient >= 0).all()) and bool((gradient <= 1 / 2.2 + 1e-12).all())


def test_native_tensors_win_over_python_constants_without_losing_gradients(native):
    torch, jax = native
    data = np.array([[0.01, -0.02], [-0.01, 0.03], [0.02, 0.01]])
    for xp in (torch, jax.numpy):
        returns = xp.asarray(data)
        risk = f.factor_risk(returns, probabilities=[0.2, 0.5, 0.3])
        assert type(risk.mean).__module__.startswith(("torch", "jax", "jaxlib"))
        np.testing.assert_allclose(risk.mean, np.array([0.003, 0.014]), atol=1e-12)
        allocation = f.allocate(returns, probabilities=[0.2, 0.5, 0.3])
        assert type(allocation.weights).__module__.startswith(("torch", "jax", "jaxlib"))
        price = f.price(xp.asarray([0, 10, 25]), probabilities=[0.2, 0.5, 0.3])
        assert float(price.value) == pytest.approx(12.5)
    values = torch.tensor([0.0, 10.0, 25.0], requires_grad=True)
    result = f.price(values, probabilities=[0.2, 0.5, 0.3])
    np.testing.assert_allclose(torch.autograd.grad(result.value, values)[0], [0.2, 0.5, 0.3])
    x = torch.tensor(data, requires_grad=True)
    loss = f.factor_risk(x, probabilities=[0.2, 0.5, 0.3]).mean.sum()
    np.testing.assert_allclose(torch.autograd.grad(loss, x)[0], [[0.2] * 2, [0.5] * 2, [0.3] * 2])
    with pytest.raises(oq.ValidationError, match="framework"):
        f.price(values, probabilities=jax.numpy.ones(3))
    negative_stride = np.array([0.2, 0.5, 0.3])[::-1]
    assert float(f.price(values, probabilities=negative_stride).value.detach()) == pytest.approx(10)


def test_native_operands_also_select_projection_scenario_and_trading_backends(native):
    torch, _ = native
    previous = torch.tensor([0.4, 0.6], dtype=torch.float64, requires_grad=True)
    weights = f.cost_aware_projection([0.41, 0.59], previous=previous, penalty=0.05)
    assert isinstance(weights, torch.Tensor)
    assert torch.isfinite(torch.autograd.grad(weights.square().sum(), previous)[0]).all()
    plan = f.ScenarioPlan(np.array([[0.01, -0.02], [-0.01, 0.03], [0.02, 0.01], [0, 0]]))
    payoffs = torch.tensor([0.0, 10.0, 20.0, 30.0], dtype=torch.float64, requires_grad=True)
    value = plan.expected_payoff([0.25] * 4, payoffs)
    np.testing.assert_allclose(torch.autograd.grad(value, payoffs)[0], [0.25] * 4)
    target = torch.tensor([0.005, -0.005], dtype=torch.float64, requires_grad=True)
    score = plan.energy_score([0.25] * 4, target)
    assert torch.isfinite(torch.autograd.grad(score, target)[0]).all()
    returns = torch.tensor(
        [[0.01, -0.02], [-0.01, 0.03], [0.02, 0.01]], dtype=torch.float64, requires_grad=True
    )
    result = f.SelfFinancingPlan().rollout(returns, [[0.5, 0.5]] * 3)
    assert isinstance(result.wealth, torch.Tensor)
    continued = f.SelfFinancingPlan().rollout(returns, [[0.5, 0.5]] * 3, state=result.state)
    gradient = torch.autograd.grad(continued.wealth.sum(), returns)[0]
    assert torch.isfinite(gradient).all() and bool((gradient != 0).any())


def test_native_float_precision_is_preserved_with_integer_probability_tensors(native):
    torch, jax = native
    values = torch.tensor([0.0, 10.0, 25.0], dtype=torch.float64, requires_grad=True)
    quote = f.price(values, probabilities=torch.tensor([1, 2, 1]))
    assert quote.value.dtype == torch.float64
    np.testing.assert_array_equal(torch.autograd.grad(quote.value, values)[0], [0.25, 0.5, 0.25])
    jvalue = jax.numpy.asarray([0.0, 10.0, 25.0], dtype=jax.numpy.float64)
    quote = f.price(jvalue, probabilities=jax.numpy.asarray([1, 2, 1]))
    assert quote.value.dtype == jax.numpy.float64
    assert float(quote.value) == pytest.approx(11.25)


def test_python_and_numpy_integer_probabilities_are_equivalent():
    for probabilities in ([1, 2, 1], np.array([1, 2, 1])):
        assert f.price([0, 10, 20], probabilities=probabilities).value == pytest.approx(10)


@pytest.mark.parametrize("direction", ["up", "down"])
@pytest.mark.parametrize("kind", ["call", "put"])
def test_conditional_barrier_in_out_parity_initial_hits_and_variance_reduction(direction, kind):
    barrier = 120 if direction == "up" else 80
    option = f.BarrierOption(100, 95, barrier, 0.2, 1, direction=direction, kind=kind)
    terminal = np.linspace(50, 150, 1001)
    survival = option.survival_probability(terminal)
    hit = option.crossing_probability(terminal)
    np.testing.assert_allclose(survival + hit, 1, atol=1e-15)
    vanilla = np.maximum(terminal - 95 if kind == "call" else 95 - terminal, 0)
    np.testing.assert_allclose(option(terminal) + replace(option, knock="in")(terminal), vanilla)
    outside = terminal >= barrier if direction == "up" else terminal <= barrier
    assert np.all(option(terminal)[outside] == 0)
    knocked = replace(option, spot=barrier)
    np.testing.assert_array_equal(knocked(terminal), 0 * terminal)
    np.testing.assert_array_equal(replace(knocked, knock="in")(terminal), vanilla)
    full_second_moment = np.mean(vanilla**2 * survival)
    conditional_second_moment = np.mean((vanilla * survival) ** 2)
    assert full_second_moment > conditional_second_moment


def _normal_cdf(x):
    return 0.5 * math.erfc(-x / math.sqrt(2))


def test_conditional_grid_price_matches_independent_absorbed_density_closed_form():
    spot, strike, barrier, sigma, time, rate = 100, 95, 125, 0.2, 1.2, 0.03
    variance = sigma * sigma * time
    mean, deviation = (rate - sigma * sigma / 2) * time, math.sqrt(variance)
    a, b = math.log(strike / spot), math.log(barrier / spot)

    def truncated_payoff(m):
        return spot * math.exp(m + variance / 2) * (
            _normal_cdf((b - m - variance) / deviation)
            - _normal_cdf((a - m - variance) / deviation)
        ) - strike * (_normal_cdf((b - m) / deviation) - _normal_cdf((a - m) / deviation))

    reference = math.exp(-rate * time) * (
        truncated_payoff(mean) - math.exp(2 * mean * b / variance) * truncated_payoff(mean + 2 * b)
    )
    edges = np.linspace(mean - 8 * deviation, mean + 8 * deviation, 65537)
    probability = np.diff([_normal_cdf((e - mean) / deviation) for e in edges])
    terminal = spot * np.exp((edges[1:] + edges[:-1]) / 2)
    option = f.BarrierOption(spot, strike, barrier, sigma, time)
    quote = f.price(option(terminal), probabilities=probability, discount=math.exp(-rate * time))
    assert quote.value == pytest.approx(reference, abs=2e-6)
    assert quote.oracle_queries == 0


def test_barrier_native_greeks_and_parameter_batch_axes(native):
    torch, jax = native
    terminal = np.array([92.0, 108.0, 116.0])
    volatility = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    option = f.BarrierOption(100, 90, 125, volatility, 1.3)
    value = option(terminal).sum()
    gradient = torch.autograd.grad(value, volatility)[0]

    def fun(vol):
        return f.BarrierOption(100, 90, 125, vol, 1.3)(terminal).sum()

    jgradient = jax.jit(jax.grad(fun))(jax.numpy.asarray(0.2))
    finite = (float(fun(0.200001)) - float(fun(0.199999))) / 2e-6
    assert float(gradient) == pytest.approx(finite, rel=1e-8)
    assert float(jgradient) == pytest.approx(finite, rel=1e-8)
    batch = f.BarrierOption(100, 90, 125, np.array([0.15, 0.2]), 1.3)(terminal)
    assert batch.shape == (2, 3)
    np.testing.assert_allclose(batch[1], option(terminal).detach(), atol=1e-12)


def test_barrier_uses_existing_coherent_pricing_without_path_registers():
    preparation = oq.Circuit(2).h(0).h(1)
    values = f.BarrierOption(100, 90, 125, 0.2, 1)(np.array([80, 95, 110, 130]))
    payoff = f.PayoffOraclePlan(preparation, values)
    assert payoff.oracle.preparation.num_qubits == 3
    quote = f.price(values, preparation=preparation, backend="numpy")
    assert quote.value == pytest.approx(values.mean()) and quote.oracle_queries == 0


@pytest.mark.parametrize(
    "field,value", [("spot", 0), ("volatility", 0), ("maturity", -1), ("strike", -1)]
)
def test_barrier_rejects_unsupported_model_domains(field, value):
    option = f.BarrierOption(100, 95, 120, 0.2, 1)
    with pytest.raises(oq.ValidationError):
        replace(option, **{field: value})([100])
