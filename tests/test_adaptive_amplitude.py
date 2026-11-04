"""Anytime coverage, actual Grover sampling and financially scaled precision."""

import itertools
import math
from fractions import Fraction

import pytest

import openqml as oq
from openqml.adaptive_amplitude import _confidence_sequence


@pytest.mark.parametrize("successes,shots", [(0, 8), (1, 8), (4, 8), (8, 8), (13, 29)])
def test_beta_mixture_boundaries_match_independent_exact_marginal(successes, shots):
    s, f = successes, shots - successes
    marginal = Fraction(
        math.factorial(2 * s) * math.factorial(2 * f),
        4**shots * math.factorial(s) * math.factorial(f) * math.factorial(shots),
    )
    threshold = math.log(20)
    lo, hi = _confidence_sequence(s, shots, threshold)
    for q in (lo, hi):
        if 0 < q < 1:
            log_e = math.log(float(marginal)) - s * math.log(q) - f * math.log1p(-q)
            assert log_e == pytest.approx(threshold, abs=2e-10)
    assert lo <= s / shots <= hi


@pytest.mark.parametrize("probability", [0.05, 0.25, 0.5, 0.9])
def test_time_uniform_coverage_by_complete_path_enumeration(probability):
    covered = 0.0
    for path in itertools.product((0, 1), repeat=9):
        successes, valid = 0, True
        for n, bit in enumerate(path, 1):
            successes += bit
            lo, hi = _confidence_sequence(successes, n, math.log(20))
            valid &= lo - 1e-14 <= probability <= hi + 1e-14
        if valid:
            covered += probability**successes * (1 - probability) ** (9 - successes)
    assert covered >= 0.95 - 1e-12


def test_adaptive_depth_and_stopping_keep_joint_coverage():
    plan = oq.AdaptiveAmplitudeEstimationPlan(max_power=1, confidence=0.9)
    theta = 0.41
    probability = math.sin(theta) ** 2
    covered, total = 0.0, 0.0
    for path in itertools.product((0, 1), repeat=8):
        records = {0: {"0": 0, "1": 0}, 1: {"0": 0, "1": 0}}
        ones, mass, valid = 0, 1.0, True
        for bit in path:
            depth = ones % 2
            rate = math.sin((2 * depth + 1) * theta) ** 2
            mass *= rate if bit else 1 - rate
            records[depth][str(bit)] += 1
            ones += bit
            try:
                result = plan.fit_counts(records)
                valid &= any(a - 1e-14 <= probability <= b + 1e-14 for a, b in result.intervals)
            except oq.BackendError:
                valid = False
        total += mass
        if valid:
            covered += mass
    assert total == pytest.approx(1)
    assert covered >= plan.confidence - 1e-12


def _oracle(p=0.23, **options):
    return oq.AmplitudeOracle(oq.Circuit(1).ry(2 * math.asin(math.sqrt(p)), 0), 0, **options)


def test_adaptive_execution_reaches_precision_with_actual_amplification():
    plan = oq.AdaptiveAmplitudeEstimationPlan(tolerance=0.025, confidence=0.99, max_queries=30_000)
    result = plan.run(_oracle(), backend="numpy", seed=41)
    assert result.converged and result.error_bound <= plan.tolerance
    assert any(a <= 0.23 <= b for a, b in result.intervals)
    assert any(m > 0 for m in result.powers)
    assert result.oracle_queries <= plan.max_queries
    assert result.oracle_queries == sum(
        n * (2 * m + 1) for n, m in zip(result.shots, result.powers, strict=True)
    )
    assert result.total_cost == result.oracle_queries


def test_cost_and_contrast_profiles_avoid_unproductive_deep_circuits():
    base = oq.AdaptiveAmplitudeEstimationPlan(tolerance=0.1)
    costs = (1,) + (1e9,) * (len(base.powers) - 1)
    expensive = oq.AdaptiveAmplitudeEstimationPlan(tolerance=0.1, costs=costs)
    noisy = oq.AdaptiveAmplitudeEstimationPlan(
        tolerance=0.1, visibility=(1,) + (0,) * (len(base.powers) - 1)
    )
    for plan in (expensive, noisy):
        result = plan.run(_oracle(), backend="numpy", seed=4)
        assert result.converged and result.powers == (0,)
    with pytest.raises(oq.ResourceError):
        oq.AdaptiveAmplitudeEstimationPlan(visibility=0).run(_oracle(), backend="numpy")


def test_budget_exhaustion_and_financial_absolute_error_are_explicit():
    oracle = _oracle()
    plan = oq.AdaptiveAmplitudeEstimationPlan(tolerance=0.001, max_queries=1)
    result = plan.run(oracle, backend="numpy", seed=2)
    assert not result.converged and result.oracle_queries == 1
    with pytest.raises(oq.ResourceError, match="achieved bound"):
        oq.finance.price(
            oracle=oracle,
            bounds=(-2, 3),
            error=0.001,
            max_queries=1,
            backend="numpy",
            method="quantum",
            seed=2,
        )
    partial = oq.finance.price(
        oracle=oracle,
        bounds=(-2, 3),
        error=0.001,
        max_queries=1,
        method="quantum",
        backend="numpy",
        seed=2,
        strict=False,
    )
    assert not partial.amplitude.converged
    quote = oq.finance.price(
        oracle=oracle, bounds=(-2, 3), error=0.1, method="quantum", backend="numpy", seed=41
    )
    assert max(abs(quote.value - q) for q in quote.interval) <= 0.1
    assert quote.interval[0] <= -2 + 5 * 0.23 <= quote.interval[1]


def test_large_error_and_constant_payoffs_do_not_launch_jobs(monkeypatch):
    import openqml.batch as batch

    monkeypatch.setattr(batch, "execute_batch", lambda *a, **kw: pytest.fail("Unnecessary job"))
    estimate = oq.AdaptiveAmplitudeEstimationPlan(tolerance=0.5).run(_oracle(), backend="numpy")
    assert estimate.converged and estimate.oracle_queries == 0
    constant = oq.finance.price([3, 3], preparation=oq.Circuit(1).h(0), discount=0.9)
    assert constant.value == pytest.approx(2.7) and constant.oracle_queries == 0
    exact = oq.finance.price([0, 10, 25], probabilities=[0.2, 0.5, 0.3], backend="missing-backend")
    assert exact.value == pytest.approx(12.5) and exact.oracle_queries == 0


def test_coherent_payoff_table_uses_same_adaptive_estimator():
    preparation = oq.Circuit(1).ry(0.7, 0)
    quote = oq.finance.price(
        [-1, 2],
        preparation=preparation,
        discount=0.97,
        error=0.1,
        method="quantum",
        backend="numpy",
        seed=3,
    )
    target = 0.97 * (-1 + 3 * math.sin(0.35) ** 2)
    assert quote.interval[0] <= target <= quote.interval[1]
    assert isinstance(quote.amplitude, oq.AdaptiveAmplitudeEstimate)


@pytest.mark.parametrize("backend", ["numpy", "auto"])
def test_auto_pricing_reuses_exact_preparation_without_amplification(backend, monkeypatch):
    def unnecessary(*args, **kwargs):
        pytest.fail("Exact pricing must not build or sample Grover circuits")

    import openqml.batch as batch

    monkeypatch.setattr(oq.AmplitudeOracle, "circuit", unnecessary)
    monkeypatch.setattr(batch, "execute_batch", unnecessary)
    preparation = oq.Circuit(1).ry(0.7, 0)
    table = oq.finance.price([-1, 2], preparation=preparation, discount=0.97, backend=backend)
    direct = oq.finance.price(oracle=_oracle(), bounds=(-2, 3), backend=backend)
    assert table.value == pytest.approx(0.97 * (-1 + 3 * math.sin(0.35) ** 2))
    assert direct.value == pytest.approx(-2 + 5 * 0.23)
    for quote in (table, direct):
        assert quote.confidence == 1 and quote.oracle_queries == 0
        assert quote.interval == (quote.value, quote.value)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(tolerance=0),
        dict(confidence=1),
        dict(max_queries=0),
        dict(costs=0),
        dict(visibility=-0.1),
    ],
)
def test_adaptive_domain_errors(kwargs):
    with pytest.raises(oq.ValidationError):
        oq.AdaptiveAmplitudeEstimationPlan(**kwargs)
