"""Projected final records permit efficient finite-shot adaptive readouts."""

import numpy as np
import pytest

from openqml import (
    BackendCapabilities,
    Circuit,
    ClassicalCondition,
    PauliString,
    ResourceError,
    ShotBackend,
    ValidationError,
    estimate,
    execute,
    execute_batch,
    expectations,
    plan_density,
    sample,
)


def history(length):
    c = Circuit(2)
    for i in range(length):
        c = c.h(0).measure(0, i).if_else(ClassicalCondition((i,), 1), Circuit(2).x(1)).reset(0)
    return c


def test_selected_bits_preserve_requested_order_zero_storage_and_last_write():
    c = Circuit(3).x(0).x(2).measure(0, 4).measure(1, 0).measure(2, 4)
    assert sample(c, 21) == {"00001": 21}
    result = execute(c, shots=21, classical_bits=(4, 0, 3))
    assert result.counts == {"100": 21} and result.classical_bits == (4, 0, 3)
    assert sample(c, 21, classical_bits=()) == {"": 21}


def test_marginal_retains_past_record_instead_of_sampling_final_quantum_basis():
    p = 0.2
    c = (
        Circuit(1)
        .ry(2 * np.arcsin(np.sqrt(p)), 0)
        .measure(0, 0)
        .reset(0)
        .if_else(
            ClassicalCondition((0,), 1),
            Circuit(1).x(0),
        )
        .x(0)
        .measure(0, 2)
    )
    counts = sample(c, 20_000, seed=7, classical_bits=(2, 0))
    assert set(counts) == {"10", "01"}
    assert counts["01"] / 20_000 == pytest.approx(p, abs=0.015)
    full = sample(c, 20_000, seed=9)
    assert full["100"] / 20_000 == pytest.approx(p, abs=0.015)


def test_long_history_projects_to_one_bit_with_two_live_branches():
    c = history(96)
    plan = plan_density(c, classical_bits=(95,), max_branches=2)
    assert plan.retains_classical and plan.classical_bits == (95,)
    assert plan.peak_branches == plan.output_branches == 2
    counts = sample(c, 2000, seed=3, classical_bits=(95,), max_branches=2)
    assert set(counts) == {"0", "1"} and sum(counts.values()) == 2000
    assert counts["1"] / 2000 == pytest.approx(0.5, abs=0.04)
    with pytest.raises(ResourceError):
        sample(c, 10)


def test_projecting_an_empty_record_marginal_needs_no_final_classical_branches():
    c = history(32)
    plan = plan_density(c, classical_bits=())
    assert plan.output_branches == 1
    assert sample(c, 17, classical_bits=()) == {"": 17}


@pytest.mark.parametrize("mixed", [False, True])
def test_sparse_billion_bit_storage_does_not_materialize_unused_output(mixed):
    c = Circuit(1).x(0).measure(0, 999_999_999)
    if mixed:
        c = c.x(0)
    result = execute(c, shots=10, classical_bits=(999_999_999, 0), max_memory_mb=0.01)
    assert result.counts == {"10": 10}
    if mixed:
        np.testing.assert_allclose(result.probabilities, [1, 0])
    else:
        np.testing.assert_allclose(result.probabilities, [0, 1])


def test_shot_estimator_and_shot_backend_discard_used_feedback_history():
    c = history(40)
    obs = [PauliString("ZI"), PauliString("IZ"), PauliString("ZZ")]
    report = estimate(c, obs, shots=4096, seed=7, max_branches=4)
    np.testing.assert_allclose(report.values, [[1, 0, 0]], atol=0.04)
    assert report.shots_per_circuit == 4096
    np.testing.assert_allclose(
        expectations(c, obs, backend=ShotBackend(shots=4096, seed=7), max_branches=4),
        [1, 0, 0],
        atol=0.04,
    )


def test_estimator_groups_different_register_widths_and_restores_input_order():
    a = Circuit(1).x(0).measure(0, 0).x(0)
    b = Circuit(1).x(0).measure(0, 4)
    report = estimate([b, a, b], [PauliString("Z")], shots=64, seed=7, max_branches=2)
    np.testing.assert_array_equal(report.values, [[-1], [1], [-1]])
    np.testing.assert_array_equal(report.covariance, np.zeros((3, 1, 1)))


def test_native_sampler_projection_happens_after_contract_validation_and_generator_once():
    class Sampler:
        name = "projection-sampler"
        capabilities = BackendCapabilities(native_sampling=True)

        def __init__(self):
            self.calls = 0

        def sample_batch(self, circuits, shots, *, options):
            self.calls += 1
            return [{"1001": shots} for c in circuits]

    device = Sampler()
    c = Circuit(1).x(0).measure(0, 0).measure(0, 3)
    assert sample(c, 9, backend=device, classical_bits=(3, 0, 2)) == {"110": 9}
    results = execute_batch([c, c], backend=device, shots=9, classical_bits=(b for b in (3, 0)))
    assert all(r.counts == {"11": 9} and r.classical_bits == (3, 0) for r in results)
    before = device.calls
    with pytest.raises(ValidationError):
        execute_batch([c, Circuit(1).measure(0, 0)], shots=9, backend=device, classical_bits=(3,))
    assert device.calls == before


@pytest.mark.parametrize("selection", [(0, 0), (-1,), (True,), (2,), 1, {0}, "0"])
def test_invalid_selections_reject_before_execution(selection):
    c = Circuit(1).measure(0, 1)
    for function in (
        lambda: execute(c, shots=10, classical_bits=selection),
        lambda: execute_batch([c], shots=10, classical_bits=selection),
        lambda: plan_density(c, classical_bits=selection),
    ):
        with pytest.raises(ValidationError):
            function()


def test_selection_requires_measurement_and_shots_and_empty_batches_validate_syntax():
    with pytest.raises(ValidationError):
        sample(Circuit(1), 10, classical_bits=())
    with pytest.raises(ValidationError):
        execute(Circuit(1).measure(0, 0), classical_bits=(0,))
    with pytest.raises(ValidationError):
        execute_batch([], shots=10, classical_bits=True)


def test_long_feedback_shot_layer_retains_trainable_weight_gradient():
    torch = pytest.importorskip("torch")
    from openqml import Parameter
    from openqml.interfaces.torch import QuantumLayer

    c = (
        history(24)
        .reset(1)
        .h(0)
        .measure(0, 24)
        .if_else(
            ClassicalCondition((24,), 1),
            Circuit(2).ry(Parameter("w"), 1),
        )
    )
    layer = QuantumLayer(
        c,
        (),
        ("w",),
        (PauliString("IZ"),),
        dtype=torch.float64,
        backend=ShotBackend(shots=8192, seed=7),
        max_branches=4,
    )
    with torch.no_grad():
        layer.weights.fill_(0.6)
    inputs = torch.empty((2, 0), dtype=torch.float64, requires_grad=True)
    output = layer(inputs)
    np.testing.assert_allclose(output.detach(), [[0.5 + 0.5 * np.cos(0.6)]] * 2, atol=0.025)
    output.sum().backward()
    assert layer.weights.grad.item() == pytest.approx(-np.sin(0.6), abs=0.05)
    assert inputs.grad is not None
