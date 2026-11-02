import math

import pytest

import openqml as q


def test_compiled_layout_and_parallel_gate_timing_affect_the_noise_quote():
    model = q.CalibrationModel(
        {
            ("h", (0,)): q.GateCalibration(0.01, 2),
            ("x", (1,)): q.GateCalibration(0.02, 3),
            ("cx", (0, 1)): q.GateCalibration(0.05, 5),
        },
        {0: q.GateCalibration(0.03, 7), 1: q.GateCalibration(0.04, 6)},
        {0: (100, 200), 1: (100, 200)},
    )
    c = q.Circuit(2).h(0).x(1).cx(0, 1)
    estimate = model.estimate(c, (q.PauliString("ZZ"),))
    expected = 0.99 * 0.98 * 0.95 * 0.97 * 0.96 * math.exp(-(15 + 14) / 300)
    assert estimate.success_score == pytest.approx(expected)
    assert estimate.shot_seconds == 15  # Parallel first layer: max(2,3), then 5, then 7
    assert estimate.normalized_bias_estimate == pytest.approx(2 * (1 - expected))
    profile = estimate.profile(queue_seconds=10, cost_per_second=0.1)
    assert profile.noise_bias == estimate.normalized_bias_estimate
    assert "not_a_noise_certificate" in profile.source
    with pytest.raises(q.ValidationError, match="physical gate"):
        model.estimate(q.Circuit(2).h(1), (q.PauliString("ZZ"),))
    with pytest.raises(q.ValidationError, match="statically"):
        model.estimate(c.measure_all(), (q.PauliString("ZZ"),))


def test_zero_success_and_missing_readout_do_not_look_like_perfect_hardware():
    model = q.CalibrationModel({("x", (0,)): q.GateCalibration(1, 1)}, {})
    estimate = model.estimate(q.Circuit(1).x(0), ())
    assert estimate.success_score == 0 and estimate.normalized_bias_estimate == 2
    with pytest.raises(q.ValidationError, match="readout"):
        model.estimate(q.Circuit(1).x(0), (q.PauliString("Z"),))


@pytest.mark.parametrize("error,seconds", ((-1, 1), (1.01, 1), (0.1, -1), (float("nan"), 1)))
def test_invalid_calibration(error, seconds):
    with pytest.raises(q.ValidationError):
        q.GateCalibration(error, seconds)
