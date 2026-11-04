"""Coherent amplification, exact payoff encoding and statistical coverage."""

import math

import numpy as np
import pytest

import openqml as q
from openqml import finance as f


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_coherent_grover_probability_and_clean_workspace(width):
    circuit = q.Circuit(width)
    for wire in range(width):
        circuit = circuit.ry(0.31 + 0.23 * wire, wire).rz(-0.17, wire)
    for wire in range(width - 1):
        circuit = circuit.cx(wire, wire + 1)
    good = width // 2
    source = q.statevector(circuit)
    bits = (np.arange(2**width) >> (width - good - 1)) & 1
    probability = sum(abs(source[bits == 1]) ** 2)
    theta = math.asin(math.sqrt(probability))
    oracle = q.AmplitudeOracle(circuit, good)
    for power in (0, 1, 3, 5):
        amplified = oracle.circuit(power, measure=False)
        state = q.statevector(amplified)
        work = amplified.num_qubits - width
        probabilities = abs(state.reshape((2**width, 2**work))) ** 2
        np.testing.assert_allclose(probabilities[:, 1:], 0, atol=1e-25)
        observed = probabilities[bits == 1].sum()
        assert observed == pytest.approx(math.sin((2 * power + 1) * theta) ** 2, abs=1e-13)
        profile = oracle.resource_profile(power)
        assert profile["operations"] == len(oracle.circuit(power).operations)
        assert profile["oracle_queries"] == 2 * power + 1


@pytest.mark.parametrize("width", [1, 2, 3])
def test_signed_payoff_multiplexor_preserves_complex_amplitudes_exactly(width):
    rng = np.random.default_rng(14 + width)
    preparation = q.Circuit(width).with_global_phase(0.37)
    for wire in range(width):
        preparation = preparation.ry(0.4 + wire, wire).rz(0.2 * wire, wire)
    if width > 1:
        preparation = preparation.cx(0, width - 1)
    payoffs = rng.normal(size=2**width)
    plan = f.PayoffOraclePlan(preparation, payoffs, bounds=(-4, 4), discount=0.97)
    state = q.statevector(plan.oracle.circuit(0, measure=False)).reshape((-1, 2))
    amplitudes = q.statevector(preparation)
    marked = (payoffs + 4) / 8
    expected = amplitudes[:, None] * np.stack((np.sqrt(1 - marked), np.sqrt(marked)), axis=-1)
    np.testing.assert_allclose(state, expected, atol=1e-14)
    p = abs(amplitudes) ** 2
    assert plan.classical_value(p) == pytest.approx(0.97 * p @ payoffs)
    for wire in range(width):
        basis = q.Circuit(width).x(wire)
        local = f.PayoffOraclePlan(basis, payoffs, bounds=(-4, 4))
        marked_probability = abs(q.statevector(local.oracle.circuit(0, measure=False))[1::2]) ** 2
        assert marked_probability.sum() == pytest.approx(marked[1 << (width - wire - 1)])


def test_likelihood_noise_profile_aliases_and_finite_count_interval():
    plan = q.AmplitudeEstimationPlan.geometric(23, shots=10_000, confidence=0.99, visibility=0.8)
    probability = 0.317
    theta = math.asin(math.sqrt(probability))
    true = 0.1 + 0.8 * np.sin((2 * np.asarray(plan.powers) + 1) * theta) ** 2
    successes = np.rint(np.asarray(plan.shots) * true).astype(int)
    result = plan.fit(successes)
    assert result.probability == pytest.approx(probability, abs=2e-6)
    assert any(a <= probability <= b for a, b in result.intervals)
    grid = np.linspace(0, math.pi / 2, 150_001)
    predicted = 0.1 + 0.8 * np.sin(grid[:, None] * (2 * np.asarray(plan.powers) + 1)) ** 2
    reference = np.log(predicted) @ successes + np.log1p(-predicted) @ (
        np.asarray(plan.shots) - successes
    )
    assert result.log_likelihood >= reference.max() - 1e-8
    assert result.oracle_queries == sum(
        n * (2 * m + 1) for m, n in zip(plan.powers, plan.shots, strict=True)
    )
    ambiguous = q.AmplitudeEstimationPlan((0, 17), shots=(1, 1000)).fit(np.array([0, 500]))
    assert len(ambiguous.intervals) > 1
    blind = q.AmplitudeEstimationPlan((0, 2), visibility=0).fit(np.array([64, 64]))
    assert blind.interval == (0, 1)


@pytest.mark.parametrize("probability", [0, 0.01, 0.1, 0.4, 0.5, 0.8, 0.99, 1])
def test_finite_binomial_confidence_coverage_by_complete_enumeration(probability):
    plan = q.AmplitudeEstimationPlan((0,), shots=8, confidence=0.9)
    covered = 0.0
    for successes in range(9):
        result = plan.fit(np.array([successes]))
        if any(a - 1e-15 <= probability <= b + 1e-15 for a, b in result.intervals):
            covered += (
                math.comb(8, successes)
                * probability**successes
                * (1 - probability) ** (8 - successes)
            )
    assert covered >= plan.confidence - 1e-14


def test_multilevel_confidence_retains_true_alias_with_exact_joint_coverage():
    plan = q.AmplitudeEstimationPlan((0, 2), shots=3, confidence=0.9)
    theta = 0.31
    rates = np.sin(np.array([1, 5]) * theta) ** 2
    covered = 0.0
    for a in range(4):
        for b in range(4):
            mass = math.comb(3, a) * rates[0] ** a * (1 - rates[0]) ** (3 - a)
            mass *= math.comb(3, b) * rates[1] ** b * (1 - rates[1]) ** (3 - b)
            try:
                result = plan.fit(np.array([a, b]))
            except q.BackendError:
                continue
            if any(lo <= math.sin(theta) ** 2 <= hi for lo, hi in result.intervals):
                covered += mass
    assert covered >= plan.confidence


def test_quantum_pricing_executes_real_sampler_batch_and_transforms_confidence():
    preparation = q.Circuit(2).ry(1.13, 0).ry(0.73, 1).cx(0, 1)
    payoff = f.PayoffOraclePlan(preparation, [-2, -0.5, 1, 3], discount=0.96)
    plan = q.AmplitudeEstimationPlan.geometric(7, shots=512, confidence=0.999)
    value = payoff.estimate(plan, backend="numpy", seed=18)
    exact = payoff.classical_value(abs(q.statevector(preparation)) ** 2)
    assert any(lo <= exact <= hi for lo, hi in value.intervals)
    assert abs(value.value - exact) < 0.03
    assert value.oracle_queries == plan.resource_profile(payoff.oracle)["oracle_queries"]
    assert value.amplitude.shots == plan.shots
    with pytest.raises(q.ValidationError):
        plan.fit_counts([{"0": 1}] * len(plan.powers))


def test_known_mean_control_variate_eliminates_constant_residual_circuit(monkeypatch):
    import openqml.batch as batch

    monkeypatch.setattr(
        batch, "execute_batch", lambda *a, **kw: pytest.fail("Unnecessary quantum job")
    )
    control = np.array([0, 1, 2, 3])
    plan = f.PayoffOraclePlan(
        q.Circuit(2).h(0).h(1),
        2 * control + 3,
        control_values=control,
        control_expectation=1.5,
        control_coefficient=2,
        bounds=(-10, 10),
        discount=0.95,
    )
    assert plan.oracle is None
    result = plan.estimate()
    assert result.value == pytest.approx(5.7)
    assert result.interval == (result.value, result.value)
    assert result.oracle_queries == 0


def test_classical_payoff_path_does_not_compile_a_quantum_oracle():
    plan = f.PayoffOraclePlan(q.Circuit(2).h(0).h(1), [-2, 0, 1, 3], max_operations=1)
    assert plan.classical_value(np.ones(4)) == pytest.approx(0.5)
    assert plan._oracle is None
    with pytest.raises(q.ResourceError):
        _ = plan.oracle


def test_coherent_input_and_resource_failures_reject_before_expansion():
    for circuit in (
        q.Circuit(1).measure_all(),
        q.Circuit(1).ry(q.Parameter("theta"), 0),
        q.Circuit(1).reset(0),
    ):
        with pytest.raises(q.ValidationError):
            q.AmplitudeOracle(circuit, 0)
    huge = q.AmplitudeOracle(q.Circuit(10**12), 0)
    with pytest.raises(q.ResourceError):
        huge.circuit(1)
    with pytest.raises(q.ResourceError):
        q.AmplitudeEstimationPlan((0, 10**12))
    with pytest.raises(q.ResourceError):
        q.AmplitudeEstimationPlan(max_work=1)
    with pytest.raises(q.ValidationError):
        f.PayoffOraclePlan(q.Circuit(2), [0, 1])
    with pytest.raises(q.ValidationError):
        f.PayoffOraclePlan(q.Circuit(1), [-1, 1], bounds=(0, 1))
    with pytest.raises(q.ValidationError):
        f.PayoffOraclePlan(q.Circuit(1), [0, 1], control_values=[0, 1])
