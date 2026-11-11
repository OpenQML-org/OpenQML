import numpy as np
import pytest

import openqml as oq


def test_readout_direct_observables_match_dense_inverse_and_joint_shot_error():
    assignment = np.array([[[0.9, 0.2], [0.1, 0.8]], [[0.8, 0.1], [0.2, 0.9]]])
    calibration = oq.ReadoutMitigation(assignment)
    distribution = np.array([0.4, 0.1, 0.2, 0.3])
    channel = np.kron(*assignment)
    counts = dict(
        zip(
            ("00", "01", "10", "11"),
            np.rint(channel @ distribution * 10000).astype(int),
            strict=True,
        )
    )
    observables = (
        oq.PauliString("ZI"),
        oq.PauliString("ZZ"),
        oq.Hamiltonian((oq.PauliString("ZI", 0.7), oq.PauliString("IZ", -0.4))),
        oq.PauliString("II", 2),
    )
    result = calibration.estimate(counts, observables)
    diagonal = np.array([[1, 1, -1, -1], [1, -1, -1, 1], [0.3, 1.1, -1.1, -0.3], [2] * 4])
    scores = diagonal @ np.linalg.inv(channel)
    frequency = np.array(list(counts.values()))
    mean = scores @ frequency / frequency.sum()
    error = np.sqrt(((scores - mean[:, None]) ** 2 @ frequency) / 10000 / 9999)
    np.testing.assert_allclose(result.values, diagonal @ distribution, atol=1e-14)
    np.testing.assert_allclose(result.values, mean, atol=1e-14)
    np.testing.assert_allclose(result.standard_errors, error, atol=1e-14)
    assert result.shots == 10000
    assert np.all(result.sampling_overhead >= 1)
    assignment[:] = 0  # Caller-owned calibration cannot mutate the retained estimator.
    np.testing.assert_allclose(calibration.estimate(counts, observables).values, mean, atol=1e-14)


def test_two_global_calibrations_recover_independent_marginals_and_preserve_overshoot():
    calibration = oq.ReadoutMitigation.from_counts(
        {"00": 720, "01": 180, "10": 80, "11": 20},
        {"00": 20, "01": 80, "10": 180, "11": 720},
    )
    np.testing.assert_allclose(calibration.assignment[:, 1, 0], [0.1, 0.2])
    np.testing.assert_allclose(calibration.assignment[:, 0, 1], [0.1, 0.2])
    result = calibration.estimate({"00": 100}, oq.PauliString("ZI"))
    assert result.values[0] > 1  # Statistical estimates must not be clipped.
    assert result.standard_errors[0] == 0
    assert calibration.estimate({"00": 1}, oq.PauliString("ZI")).standard_errors is None


def test_readout_sparse_counts_do_not_expand_hilbert_space_and_obey_budgets():
    width = 128
    assignment = np.broadcast_to([[0.99, 0.01], [0.01, 0.99]], (width, 2, 2))
    calibration = oq.ReadoutMitigation(assignment, max_memory_mb=1)
    result = calibration.estimate(
        {"0" * width: 80, "1" * width: 20}, oq.PauliString("Z" + "I" * (width - 1))
    )
    np.testing.assert_allclose(result.values, [0.6 / 0.98])
    with pytest.raises(oq.ResourceError, match="max_overhead"):
        oq.ReadoutMitigation(assignment, max_overhead=2).estimate(
            {"0" * width: 1}, oq.PauliString("Z" * width)
        )
    with oq.compute_context(policy=oq.ComputePolicy(max_memory_mb=0.001)):
        with pytest.raises(oq.ResourceError):
            oq.ReadoutMitigation(assignment)
    with pytest.raises(oq.ResourceError):
        oq.ReadoutMitigation(assignment, max_work=1)


@pytest.mark.parametrize("assignment", [[], [[[0.5, 0.5], [0.5, 0.5]]], [[[1, 1], [1, 1]]]])
def test_invalid_or_singular_readout_calibration_fails(assignment):
    with pytest.raises((oq.ValidationError, oq.NumericalError)):
        oq.ReadoutMitigation(assignment)


@pytest.mark.parametrize("counts", [{"0": -1}, {"0": True}, {"0": 0}, {"x": 1}, {"00": 1}])
def test_readout_count_contract(counts):
    calibration = oq.ReadoutMitigation(np.eye(2)[None])
    with pytest.raises(oq.ValidationError):
        calibration.estimate(counts, oq.PauliString("Z"))


def test_readout_rejects_non_diagonal_observables():
    with pytest.raises(oq.ValidationError, match="I/Z"):
        oq.ReadoutMitigation(np.eye(2)[None]).estimate({"0": 5}, oq.PauliString("X"))


def test_richardson_cancels_known_noise_orders_and_propagates_uncertainty():
    scales = np.array([1.0, 2, 3])
    values = 0.83 - 0.1 * scales + 0.02 * scales**2
    errors = np.array([0.01, 0.02, 0.03])
    result = oq.zero_noise_extrapolate(scales, values, order=2, standard_errors=errors)
    assert result.values == pytest.approx(0.83)
    np.testing.assert_allclose(result.weights, [3, -3, 1], atol=1e-14)
    np.testing.assert_allclose(result.residuals, 0, atol=1e-14)
    assert result.standard_errors == pytest.approx(np.linalg.norm([0.03, -0.06, 0.03]))
    assert result.sampling_overhead == pytest.approx(49)
    assert oq.zero_noise_extrapolate(scales, values).standard_errors is None


def test_extrapolation_of_physical_amplitude_damping_reduces_bias_at_low_noise():
    scales = np.array([1.0, 2, 3])
    circuits = [
        oq.Circuit(1).x(0).amplitude_damping(-np.expm1(-0.05 * scale), 0) for scale in scales
    ]
    measured = oq.expectations_batch(circuits, [oq.PauliString("Z")], backend="numpy")
    np.testing.assert_allclose(measured[:, 0], 1 - 2 * np.exp(-0.05 * scales), atol=1e-14)
    result = oq.zero_noise_extrapolate(scales, measured, order=2)
    assert abs(result.values[0] + 1) < abs(measured[0, 0] + 1) / 100


@pytest.mark.parametrize("scales", [[1, 1, 2], [0, 1, 2], [-1, 1, 2], [1, np.inf, 2]])
def test_extrapolation_rejects_invalid_physical_scales(scales):
    with pytest.raises(oq.ValidationError):
        oq.zero_noise_extrapolate(scales, [0.8, 0.7, 0.6])


def test_batched_weighted_extrapolation_agrees_with_generalized_least_squares():
    scales = np.array([1, 2, 3, 5.0])
    values = np.array([[0.2, 0.5], [0.1, 0.4], [-0.1, 0.3], [-0.2, 0.25]])
    errors = np.array([[0.01, 0.03], [0.02, 0.02], [0.015, 0.04], [0.03, 0.01]])
    result = oq.zero_noise_extrapolate(scales, values, standard_errors=errors)
    design = np.column_stack([np.ones(4), scales])
    for j in range(2):
        inverse = np.diag(errors[:, j] ** -2)
        reference = np.linalg.solve(design.T @ inverse @ design, design.T @ inverse)[0]
        np.testing.assert_allclose(result.weights[:, j], reference, atol=1e-13)
        assert result.values[j] == pytest.approx(reference @ values[:, j])
    covariance = np.diag(errors[:, 0] ** 2) + 0.00005
    result = oq.zero_noise_extrapolate(scales, values, covariance=covariance)
    inverse = np.linalg.inv(covariance)
    weights = np.linalg.solve(design.T @ inverse @ design, design.T @ inverse)[0]
    np.testing.assert_allclose(result.values, weights @ values)
    np.testing.assert_allclose(result.standard_errors, np.sqrt(weights @ covariance @ weights))
    np.testing.assert_allclose(result.weights[:, 0], weights)


@pytest.mark.parametrize(
    "options, error",
    [
        ({"order": 3}, oq.ValidationError),
        ({"standard_errors": [0.1, 0, 0.1]}, oq.ValidationError),
        ({"covariance": np.ones((3, 3))}, oq.ValidationError),
        ({"covariance": np.eye(3), "standard_errors": [1, 1, 1]}, oq.ValidationError),
        ({"max_condition": 1}, oq.NumericalError),
        ({"max_overhead": 1}, oq.ResourceError),
        ({"max_memory_mb": 0.00001}, oq.ResourceError),
        ({"max_work": 1}, oq.ResourceError),
    ],
)
def test_extrapolation_guards(options, error):
    with pytest.raises(error):
        oq.zero_noise_extrapolate([1, 2, 3], [0.8, 0.7, 0.6], **options)


def test_width_policy_is_memory_derived_and_structured_readout_exceeds_24_qubits():
    assert oq.ExecutionOptions.for_statevector(2048).max_qubits == 25
    circuit = oq.Circuit(64).h(0).cx(0, 63)
    value = oq.expectations_batch(circuit, [oq.PauliString("Z" + "I" * 62 + "Z")], backend="cpu")
    np.testing.assert_allclose(value, [[1]], atol=1e-12)
