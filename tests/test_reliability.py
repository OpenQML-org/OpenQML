"""Failure boundaries, true residuals, scope isolation and submitted-job recovery."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

import openqml as oq
from openqml import compute
from openqml._native_training import solve_metric
from openqml._tensor_contraction import TensorNetworkPlan
from openqml.finance._array import Ops
from openqml.finance._linear import linear_solve
from openqml.recovery import CircuitBreaker, PendingBatch, PendingJob
from openqml.simulator import _TensorOps


@pytest.fixture(autouse=True)
def kernel_scope(monkeypatch):
    monkeypatch.setattr(compute, "_KERNEL", oq.ComputeKernel())


def test_independent_keys_compile_concurrently_and_same_keys_share_failures():
    kernel = oq.ComputeKernel()
    barrier = threading.Barrier(2)

    def compile_one(index):
        def build():
            barrier.wait(timeout=3)
            return (index,)

        return kernel.plan("parallel", index, build)

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(compile_one, (1, 2))) == [(1,), (2,)]
    with pytest.raises(oq.ValidationError, match="Recursive compilation"):
        kernel.plan("recursive", (), lambda: kernel.plan("recursive", (), lambda: ()))


def test_clear_cache_does_not_publish_an_inflight_old_plan():
    kernel, entered, finish = oq.ComputeKernel(), threading.Event(), threading.Event()

    def build():
        entered.set()
        assert finish.wait(3)
        return (1,)

    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(kernel.plan, "cleared", (), build)
        assert entered.wait(3)
        kernel.clear_cache()
        finish.set()
        assert result.result(timeout=3) == (1,)
    assert kernel.cache_info()["entries"] == 0


def test_contexts_are_isolated_across_threads_and_async_tasks():
    baseline = oq.get_compute_kernel()

    def worker(budget):
        with oq.compute_context(policy=oq.ComputePolicy(max_memory_mb=budget)) as kernel:
            assert oq.get_compute_kernel() is kernel
            return oq.ExecutionOptions(max_memory_mb=100).max_memory_mb

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(worker, (2, 7))) == [2, 7]

    async def task(budget):
        with oq.compute_context(policy=oq.ComputePolicy(max_memory_mb=budget)):
            await asyncio.sleep(0)
            return oq.get_compute_kernel().policy.max_memory_mb

    async def tasks():
        return await asyncio.gather(task(3), task(9))

    assert asyncio.run(tasks()) == [3, 9]
    assert oq.get_compute_kernel() is baseline
    captured = _TensorOps()
    changed = oq.configure_compute(policy=oq.ComputePolicy(max_memory_mb=1))
    assert captured.kernel is baseline and oq.get_compute_kernel() is changed


def test_runtime_scopes_do_not_replace_the_default_scheduler():
    default = oq.get_backend("auto")
    scheduler = oq.CostAwareBackend()
    with oq.runtime_context(backend=scheduler):
        assert oq.get_backend("auto") is scheduler
    assert oq.get_backend("auto") is default


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_compiled_errors_preserve_validation_and_numerical_types(backend):
    xp = pytest.importorskip(backend if backend == "torch" else "jax.numpy")

    def function(value):
        ops = Ops(value)
        return ops.check(value, value >= 0, "Expected nonnegative inputs.") ** 0.5

    options = {"compiler": "eager", "fullgraph": True} if backend == "torch" else {}
    fn = oq.compile_compute(function, backend=backend, **options)
    value = xp.asarray([1.0, 4.0]) if backend == "jax" else xp.tensor([1.0, 4.0])
    np.testing.assert_allclose(np.asarray(fn(value)), [1, 2])
    with pytest.raises(oq.ValidationError, match="nonnegative"):
        fn(-value)
    numeric = oq.compile_compute(lambda x: x / 0, backend=backend, **options)
    with pytest.raises(oq.NumericalError):
        numeric(value)


def _system():
    rng = np.random.default_rng(8)
    factors = rng.normal(size=(12, 12))
    diagonal, mass, rhs = np.linspace(0.3, 0.9, 12), np.ones(12) / 12, rng.normal(size=(12, 2))
    rhs[:, 1] = 0
    return diagonal, factors, mass, rhs


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_factor_solver_certifies_real_residual_and_rejects_iteration_exhaustion(backend):
    if backend == "torch":
        xp = pytest.importorskip("torch")
        values = tuple(xp.tensor(v, dtype=xp.float64) for v in _system())
    elif backend == "jax":
        xp = pytest.importorskip("jax.numpy")
        values = tuple(xp.asarray(v) for v in _system())
    else:
        values = _system()
    diagonal, factors, mass, rhs = values
    tolerance = 2e-6 if backend == "jax" and "32" in str(rhs.dtype) else 1e-10
    actual = linear_solve(*values, 128, tolerance, Ops(diagonal))
    d, f, p, b = _system()
    expected = np.linalg.solve(np.diag(d) + f.T @ (p[:, None] * f), b)
    np.testing.assert_allclose(np.asarray(actual), expected, atol=2e-5, rtol=2e-5)
    with pytest.raises(oq.ConvergenceError, match="true residual"):
        linear_solve(*values, 1, tolerance, Ops(diagonal))


def test_checked_jax_solver_propagates_nonconvergence_through_implicit_solve():
    jnp = pytest.importorskip("jax.numpy")
    d, f, p, rhs = (jnp.asarray(v) for v in _system())
    fn = oq.compile_compute(lambda b: linear_solve(d, f, p, b, 1, 1e-6, Ops(d)), backend="jax")
    with pytest.raises(oq.ConvergenceError):
        fn(rhs)


def test_native_metric_stops_at_convergence_and_rejects_uncertified_directions():
    calls = []
    ops = _TensorOps()

    def identity(value):
        calls.append(1)
        return value

    result = solve_metric(identity, {"x": np.ones(4)}, ops, max_iterations=32)
    assert bool(result.converged) and len(calls) == result.metric_products == 2

    def bad(value):
        return {"x": np.arange(1, 5) * value["x"]}

    with pytest.raises(oq.ConvergenceError):
        solve_metric(bad, {"x": np.ones(4)}, ops, max_iterations=1, tolerance=1e-12)


def test_pending_jobs_retry_result_only_and_resume_without_submission():
    class Job:
        attempts = 0

        def job_id(self):
            return "existing-job"

        def result(self):
            self.attempts += 1
            if self.attempts < 3:
                raise TimeoutError("temporary retrieval failure")
            return 4

    job = Job()
    pending = PendingJob(
        job,
        lambda value, _: value * 2,
        backend="qpu",
        metadata={},
        policy=oq.RecoveryPolicy(max_attempts=2, initial_delay=0),
    )
    with pytest.raises(oq.BackendJobError) as caught:
        pending.result()
    assert caught.value.job_id == "existing-job" and job.attempts == 2
    assert caught.value.resume() == 8 and job.attempts == 3
    assert pending.result() == 8 and job.attempts == 3


def test_qpu_breaker_rejects_new_work_before_submission_and_recovers():
    class Device:
        name = "recoverable-test-qpu"
        capabilities = oq.BackendCapabilities()
        attempts = 0

        def run(self, circuit, *, options):
            self.attempts += 1
            raise ConnectionError("provider unavailable")

    device = Device()
    wrapped = oq.QPUBackend(
        device, recovery=oq.RecoveryPolicy(failure_threshold=1, cooldown_seconds=60)
    )
    with pytest.raises(ConnectionError):
        wrapped.run(oq.Circuit(1), options=oq.ExecutionOptions())
    assert not wrapped.is_available()
    with pytest.raises(oq.BackendUnavailableError):
        wrapped.run(oq.Circuit(1), options=oq.ExecutionOptions())
    assert device.attempts == 1
    breaker = CircuitBreaker(oq.RecoveryPolicy(failure_threshold=1, cooldown_seconds=0))
    breaker.acquire()
    breaker.failure()
    breaker.acquire()
    assert not breaker.available()
    breaker.success()
    assert breaker.available()


def test_chunk_recovery_retains_completed_work_and_unknown_submissions_are_not_replayed():
    submissions = [0, 0]

    class Job:
        attempts = 0

        def job_id(self):
            return "chunk-job"

        def result(self):
            self.attempts += 1
            if self.attempts == 1:
                raise TimeoutError("retrieve later")
            return [2]

    job = Job()

    def first():
        submissions[0] += 1
        return [1]

    def second():
        submissions[1] += 1
        return PendingJob(
            job,
            lambda result, _: result,
            backend="qpu",
            metadata={},
            policy=oq.RecoveryPolicy(max_attempts=1),
        ).result()

    batch = PendingBatch([first, second], lambda chunks: sum(chunks, []))
    batch.take(0, first)
    batch.take(1, second)
    with pytest.raises(oq.BackendJobError) as caught:
        batch.assemble()
    assert caught.value.resume() == [1, 2]
    assert submissions == [1, 1] and job.attempts == 2

    def unknown():
        raise oq.BackendJobError("Unknown submission", stage="submission")

    blocked = PendingBatch([unknown], lambda chunks: chunks)
    blocked.take(0, unknown)
    with pytest.raises(oq.BackendJobError):
        blocked.result()


def test_recorded_qiskit_jobs_resume_by_identity_and_precision_views_share_compilation():
    pytest.importorskip("qiskit")
    from qiskit.providers.exceptions import JobError, JobTimeoutError

    assert oq.RecoveryPolicy().transient(JobTimeoutError("still waiting"))
    assert not oq.RecoveryPolicy().transient(JobError("terminal job failure"))

    class Job:
        attempts = 0

        def job_id(self):
            return "provider-job"

        def result(self):
            self.attempts += 1
            if self.attempts < 3:
                raise JobTimeoutError("retrieval timeout")
            return [SimpleNamespace(data=SimpleNamespace(evs=np.array([[0.25]])))]

    job = Job()

    class Estimator:
        submissions = 0

        def run(self, *args, **kwargs):
            self.submissions += 1
            return job

    estimator = Estimator()
    backend = oq.QiskitEstimatorBackend(
        estimator, recovery=oq.RecoveryPolicy(max_attempts=2, initial_delay=0)
    )
    with pytest.raises(oq.BackendJobError) as caught:
        backend.expectations_batch(
            [oq.Circuit(1)], [oq.PauliString("Z")], options=oq.ExecutionOptions()
        )
    receipt = caught.value.pending.to_dict()
    restored = oq.recover_job(receipt, lambda identifier: job)
    np.testing.assert_array_equal(restored, [[0.25]])
    assert estimator.submissions == 1 and job.attempts == 3
    assert backend.with_shots(256)._cache is backend._cache
    wrong = Job()
    wrong.job_id = lambda: "different-job"
    with pytest.raises(oq.ValidationError, match="job ID"):
        oq.recover_job(receipt, lambda identifier: wrong)
    assert wrong.attempts == 0


def test_runtime_workers_inherit_scopes_and_kernel_policy_is_read_only():
    from openqml.runtime import _ObservedBackend

    kernel = oq.ComputeKernel(oq.ComputePolicy(max_memory_mb=8))
    with pytest.raises(AttributeError):
        kernel.policy = oq.ComputePolicy()

    class Device:
        name = "scoped-local-device"
        capabilities = oq.BackendCapabilities(native_expectations=True, exact_expectations=True)

        def expectations_batch(self, circuits, observables, *, options):
            assert oq.get_compute_kernel() is kernel
            assert oq.get_backend("auto") is scheduler
            return np.zeros((len(circuits), len(observables)))

    profile = oq.ExecutionProfile(0.01, max_batch_size=1, concurrency=2, relative_uncertainty=0)
    scheduler = oq.CostAwareBackend(profiles={Device.name: profile})
    observed = _ObservedBackend(
        Device(), scheduler, profile, "expectations", (oq.PauliString("Z"),), None
    )
    with oq.compute_context(kernel=kernel), oq.runtime_context(backend=scheduler):
        result = observed.expectations_batch(
            [oq.Circuit(1)] * 3, (oq.PauliString("Z"),), options=oq.ExecutionOptions()
        )
    np.testing.assert_array_equal(result, np.zeros((3, 1)))


@pytest.mark.parametrize(
    "equation,shapes",
    [
        ("bij,bjk->bik", ((2, 32, 64), (1, 64, 16))),
        ("bpaij,aioz->bpozj", ((2, 3, 2, 4, 64), (2, 4, 8, 4))),
        ("...rn,...rc->...nc", ((2, 64, 32), (1, 64, 16))),
        ("ij,ij->", ((32, 64), (32, 64))),
        ("ij,jk->ik", ((32, 1), (64, 32))),
        ("...i,...i->...", ((2, 32, 64), (32, 64))),
    ],
)
def test_prepared_binary_gemms_match_einstein_contractions(equation, shapes):
    rng = np.random.default_rng(11)
    values = tuple(rng.normal(size=shape) for shape in shapes)
    kernel = oq.ComputeKernel()
    actual = kernel.einsum(equation, *values)
    expected = np.einsum(equation, *values)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=2e-12)


def test_broadcast_contraction_avoids_unadmitted_reshape_copies(monkeypatch):
    from openqml.compute import _BinaryGEMM

    left, right = np.arange(2048.0)[:, None], np.arange(2048.0)[None, :]
    kernel = oq.ComputeKernel(oq.ComputePolicy(max_memory_mb=1))

    def forbidden(*args):
        pytest.fail("Entered GEMM with unadmitted expanded operands")

    monkeypatch.setattr(_BinaryGEMM, "apply", forbidden)
    actual = kernel.einsum("ij,ij->", left, right)
    np.testing.assert_allclose(actual, left.sum() * right.sum(), rtol=1e-12)


def test_streaming_keeps_its_snapshot_without_leaking_it_to_the_consumer():
    p = oq.Parameter("angle")
    plan = TensorNetworkPlan(oq.Circuit(1).ry(p, 0), (oq.PauliString("Z"),))
    stream = plan.iter_evaluate({"angle": np.array([0.1, 0.2, 0.3])}, batch_chunk_size=1)
    first = next(stream)
    changed = oq.configure_compute(policy=oq.ComputePolicy(max_memory_mb=1e-6))
    tail = list(stream)
    np.testing.assert_allclose(np.concatenate([first, *tail])[:, 0], np.cos([0.1, 0.2, 0.3]))
    assert oq.get_compute_kernel() is changed


def test_dynamic_compiled_shapes_respect_resource_admission():
    torch = pytest.importorskip("torch")
    kernel = oq.ComputeKernel(oq.ComputePolicy(max_memory_mb=0.01))
    fn = kernel.compile(
        lambda x: Ops(x).matmul(x, x.T),
        backend="torch",
        compiler="eager",
        fullgraph=True,
        dynamic=True,
    )
    assert fn(torch.ones((3, 4))).shape == (3, 3)
    assert fn(torch.ones((4, 4))).shape == (4, 4)
    with pytest.raises(oq.ResourceError):
        fn(torch.ones((80, 4)))

    def budgeted(x):
        kernel.admit(x.numel(), 0.001, itemsize=x.dtype.itemsize)
        return x * 2

    fn = kernel.compile(budgeted, backend="torch", compiler="eager", dynamic=True, fullgraph=True)
    assert fn(torch.ones(4)).shape == (4,)
    with pytest.raises(oq.ResourceError):
        fn(torch.ones(1000))


def test_checked_torch_callables_do_not_reuse_another_kernel_policy():
    torch = pytest.importorskip("torch")

    def function(x):
        return Ops(x).matmul(x, x.T)

    def compile_one(budget):
        return oq.ComputeKernel(oq.ComputePolicy(max_memory_mb=budget)).compile(
            function, backend="torch", compiler="eager", fullgraph=True
        )

    small, large = compile_one(0.01), compile_one(1)
    x = torch.ones((80, 4))
    assert large(x).shape == (80, 80)
    with pytest.raises(oq.ResourceError):
        small(x)
    assert large(x).shape == (80, 80)


def test_checked_jax_checkpoint_preserves_gradient_and_error_status():
    jax = pytest.importorskip("jax")
    from openqml.interfaces._jax_network import _checkpoint

    def function(x):
        x = Ops(x).check(x, x > 0, "Checkpoint requires positive inputs.")
        return (x * x).sum()

    gradient = jax.value_and_grad(lambda x: _checkpoint(function, x))
    checked = oq.compile_compute(gradient, backend="jax")
    value, derivative = checked(jax.numpy.array([2.0, 3.0]))
    np.testing.assert_allclose(value, 13)
    np.testing.assert_allclose(derivative, [4, 6])
    with pytest.raises(oq.ValidationError, match="Checkpoint"):
        checked(jax.numpy.array([-2.0, 3.0]))
