"""Synthetic quote/ledger contracts; these are not physical device measurements."""

import time
from concurrent.futures import ThreadPoolExecutor
from copy import copy

import numpy as np
import pytest

import openqml as q
from openqml import backends
from openqml.costs import check_cost, estimate_cost


class Device:
    def __init__(self, name, kind, *, calls=None, shots=100):
        self.name, self.shots = name, shots
        self.calls = [] if calls is None else calls
        self.capabilities = q.BackendCapabilities(
            device_type=kind,
            native_expectations=True,
            native_sampling=True,
            deterministic_expectations=kind != "QPU",
            exact_expectations=kind != "QPU",
        )

    def expectations_batch(self, circuits, observables, *, options):
        self.calls.append((len(circuits), self.shots))
        return q.expectations_batch(circuits, observables, backend="numpy", **options.__dict__)

    def with_shots(self, shots):
        result = copy(self)
        result.shots = shots
        return result

    def sample_batch(self, circuits, shots, *, options):
        self.calls.append((len(circuits), shots))
        return [{"0": shots} for _ in circuits]


@pytest.fixture
def devices(monkeypatch):
    devices = [Device("cost-qpu", "QPU"), Device("gpu", "GPU"), Device("cpu", "CPU")]
    for device in devices:
        monkeypatch.setitem(backends._REGISTRY, device.name, device)
    return devices


def profile(seconds, **kwargs):
    options = dict(queue_seconds=0, cost_per_second=0, noise_bias=0)
    options.update(kwargs)
    return q.ExecutionProfile(seconds, relative_uncertainty=0, shots=100, **options)


def test_whole_training_queue_cost_noise_and_gradient_counts_change_the_route(devices):
    circuit, obs = q.Circuit(1).ry(0.4, 0), (q.PauliString("Z"),)
    work = q.TrainingWorkload(batch_size=8, steps=10, gradient_evaluations=6)
    profiles = {
        "cost-qpu": profile(0.01, queue_seconds=100, max_batch_size=2),
        "gpu": profile(0.1),
        "cpu": profile(1),
    }
    scheduler = q.CostAwareBackend(profiles=profiles, workload=work)
    route = scheduler.plan(circuit, observables=obs)
    assert route.backend == "gpu" and not any(d.calls for d in devices)
    gpu = next(c for c in route.candidates if c.backend == "gpu")
    assert gpu.estimate["circuit_evaluations"] == 560
    assert gpu.estimate["seconds"] == pytest.approx(56)
    assert q.expectations(circuit, obs, backend=scheduler)[0] == pytest.approx(np.cos(0.4))
    assert devices[1].calls == [(1, 100)]
    profiles["cost-qpu"] = profile(0.01, cost_per_shot=0.01)
    assert (
        q.CostAwareBackend(profiles=profiles, workload=work).plan(circuit, observables=obs).device
        == "QPU"
    )
    limited = q.CostAwareBackend(
        profiles=profiles, workload=work, policy=q.SchedulingPolicy(max_cost=1)
    )
    assert limited.plan(circuit, observables=obs).device == "GPU"
    profiles["cost-qpu"] = profile(0.001, noise_bias=0.2)
    quality = q.CostAwareBackend(profiles=profiles, policy=q.SchedulingPolicy(max_rmse=0.1))
    assert quality.plan(circuit, observables=obs).device == "GPU"


def test_unquoted_qpu_is_not_treated_as_free_or_fast(devices, monkeypatch):
    scheduler = q.CostAwareBackend()
    monkeypatch.setitem(backends._REGISTRY, "auto", scheduler)
    c, obs = q.Circuit(1), (q.PauliString("Z"),)
    plan = scheduler.plan(c, observables=obs)
    assert plan.device == "GPU" and plan.decision == "unprofiled_local_fallback"
    assert plan.candidates[0].eligible is False
    assert q.PriorityBackend().plan(c, observables=obs).device == "QPU"
    strict = q.CostAwareBackend(policy=q.SchedulingPolicy(require_profiles=True))
    with pytest.raises(q.ResourceError, match="No cost/quality-feasible"):
        strict.plan(c, observables=obs)
    assert not any(d.calls for d in devices)


def test_shot_allocation_is_applied_and_noise_bias_is_not_fixed_by_shots(devices):
    calls = []

    def quote(request):
        calls.append(request)
        return profile(0.001, noise_bias=0.03, cost_per_shot=0.001)

    scheduler = q.CostAwareBackend(
        profiles={"cost-qpu": quote}, policy=q.SchedulingPolicy(max_rmse=0.05)
    )
    c, obs = q.Circuit(1), (q.PauliString("Z", 2),)
    with pytest.raises(q.ResourceError):
        scheduler.plan(c, observables=obs)
    obs = (q.PauliString("Z"),)
    plan = scheduler.plan(c, observables=obs)
    expected = 625  # 1 / (.05**2 - .03**2)
    assert next(v for v in plan.candidates if v.backend == "cost-qpu").estimate["shots"] in (
        expected,
        expected + 1,
    )
    q.expectations(c, obs, backend=scheduler)
    assert devices[0].calls[-1][1] in (expected, expected + 1)
    assert calls[-1].shots in (expected, expected + 1)


def test_sampling_uses_actual_requested_shots_in_forecast_and_submission(devices):
    scheduler = q.CostAwareBackend(profiles={"cost-qpu": profile(0.01, cost_per_shot=0.1)})
    plan = scheduler.plan(q.Circuit(1), task="sampling", shots=13)
    estimate = next(c for c in plan.candidates if c.backend == "cost-qpu").estimate
    assert estimate["physical_shots"] == 13 and estimate["cost"] == pytest.approx(1.3)
    assert q.sample(q.Circuit(1), 13, backend=scheduler) == {"0": 13}
    assert devices[0].calls == [(1, 13)]


def test_profile_expiry_and_forecast_uncertainty_are_admission_constraints(devices):
    expired = profile(0.01, observed_at=time.time() - 100, valid_for_seconds=1)
    scheduler = q.CostAwareBackend(profiles={"cost-qpu": expired})
    with pytest.raises(q.ResourceError, match="stale"):
        q.CostAwareBackend(
            profiles={"cost-qpu": expired}, policy=q.SchedulingPolicy(require_profiles=True)
        ).plan(q.Circuit(1))
    assert scheduler.plan(q.Circuit(1)).device != "QPU"
    request = q.SchedulingRequest(
        (q.Circuit(1),),
        (),
        "expectations",
        q.ExecutionOptions(),
        q.TrainingWorkload(batch_size=5, steps=3, gradient_evaluations=2),
    )
    quote = q.ExecutionProfile(
        1, max_batch_size=2, concurrency=2, queue_seconds=10, relative_uncertainty=0.5
    )
    cost = estimate_cost(quote, request, q.SchedulingPolicy(), exact=True)
    assert cost.provider_jobs == 24  # 3 + 5 jobs per sequential optimizer step
    assert cost.seconds == 3 * (23 + 36)  # (2 waves + 3 rows) + (3 waves + 6 rows)
    assert cost.upper_seconds == 2 * cost.seconds
    with pytest.raises(q.ResourceError, match="max_seconds"):
        check_cost(cost, q.SchedulingPolicy(max_seconds=cost.seconds))


def test_shift_workload_counts_occurrences_and_controlled_gate_recipes():
    x = q.Parameter("x")
    circuit = q.Circuit(2).ry(x, 0).crx(x, 0, 1).rz(x * 2, 1)
    plan = q.ShiftPlan(circuit, (q.PauliString("ZZ"),), backend="numpy")
    work = q.TrainingWorkload.parameter_shift(circuit, plan.observables, steps=12)
    assert plan.occurrence_count == 3 and plan.shift_evaluations == 8
    assert work.gradient_evaluations == 8 and work.steps == 12


def test_microbatch_phases_keep_primal_and_gradient_chunk_limits_separate():
    request = q.SchedulingRequest(
        (q.Circuit(1),),
        (q.PauliString("Z"),),
        "expectations",
        q.ExecutionOptions(),
        q.TrainingWorkload(
            batch_size=5,
            steps=2,
            gradient_evaluations=10,
            microbatch_size=2,
            gradient_chunk_size=64,
        ),
    )
    cost = estimate_cost(
        profile(0.01, cost_per_job=1, queue_seconds=3), request, q.SchedulingPolicy()
    )
    # Three microbatches each submit one primal and one gradient job per step.
    assert cost.provider_jobs == 12 and cost.cost == 12
    assert cost.circuit_evaluations == 110
    assert cost.seconds == pytest.approx(36 + 1.1)


def test_profile_batch_limit_and_readout_groups_match_actual_submissions(devices):
    device = devices[0]
    scheduler = q.CostAwareBackend(
        profiles={device.name: profile(0.01, max_batch_size=2, concurrency=2, cost_per_job=1)}
    )
    x = q.Parameter("x")
    circuit, obs = (
        q.Circuit(1).ry(x, 0),
        (q.PauliString("X"), q.PauliString("Y"), q.PauliString("Z")),
    )
    values = np.arange(5) * 0.2
    actual = q.expectations_batch(circuit, obs, [{"x": v} for v in values], backend=scheduler)
    np.testing.assert_allclose(actual[:, 0], np.sin(values), atol=1e-14)
    np.testing.assert_allclose(actual[:, 2], np.cos(values), atol=1e-14)
    assert sorted(n for n, _ in device.calls) == [1, 2, 2]
    quote = scheduler.plan([circuit] * 5, observables=obs)
    estimate = next(c.estimate for c in quote.candidates if c.backend == device.name)
    assert estimate["provider_jobs"] == 3 and estimate["physical_shots"] == 1500


def test_numpy_scalar_profiles_are_json_serializable(devices):
    import json

    scheduler = q.CostAwareBackend(
        profiles={devices[0].name: profile(np.float32(0.01), max_batch_size=np.int64(2))},
        workload=q.TrainingWorkload(steps=np.int64(3)),
    )
    json.dumps(scheduler.plan(q.Circuit(1), observables=(q.PauliString("Z"),)).to_dict())


def test_feedback_is_bounded_and_can_deprioritize_a_slow_device(devices):
    profiles = {"cost-qpu": profile(0.1), "gpu": profile(0.2)}
    feedback = q.ExecutionFeedback(capacity=2, window=3)
    scheduler = q.CostAwareBackend(profiles=profiles, feedback=feedback)
    c, obs = q.Circuit(1), (q.PauliString("Z"),)
    assert scheduler.plan(c, observables=obs).device == "QPU"
    from openqml.runtime import _feedback_key

    request = q.SchedulingRequest(
        (c,), obs, "expectations", q.ExecutionOptions(), q.TrainingWorkload()
    )
    key = (*_feedback_key(devices[0], request, 100), "declared")
    for _ in range(5):
        feedback.observe(key, 0.1, 2)
    assert feedback.correction(key) == (20, 20, 3)
    assert scheduler.plan(c, observables=obs).device == "GPU"


def test_spending_reservations_are_atomic_and_failed_calls_are_not_retried(devices):
    ledger = q.BudgetLedger(1)

    def reserve(_):
        try:
            ledger.reserve(0.25)
            return True
        except q.ResourceError:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(reserve, range(16))) == 4
    assert ledger.reserved == 1
    ledger = q.BudgetLedger(2)
    scheduler = q.CostAwareBackend(
        profiles={"cost-qpu": profile(0.01, cost_per_job=1)}, ledger=ledger
    )

    def fail(*args, **kwargs):
        raise q.BackendError("Provider accepted and failed")

    devices[0].expectations_batch = fail
    with pytest.raises(q.BackendError, match="accepted and failed"):
        q.expectations(q.Circuit(1), (q.PauliString("Z"),), backend=scheduler)
    assert ledger.reserved == 1 and devices[1].calls == []


@pytest.mark.parametrize(
    "kwargs",
    (
        {"queue_seconds": -1},
        {"relative_uncertainty": float("nan")},
        {"shots": 0},
        {"concurrency": 0},
        {"mitigation_overhead": 0.5},
    ),
)
def test_invalid_profiles_reject_before_execution(kwargs):
    with pytest.raises(q.ValidationError):
        q.ExecutionProfile(0.1, **kwargs)
