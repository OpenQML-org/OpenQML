"""Accelerator staging and provider lifecycle contracts; hardware is never emulated."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import openqml as oq


def test_parameter_preparation_shares_one_buffer_and_preserves_nested_feedback(monkeypatch):
    from openqml.interop._parameter_batches import binding_matrix, parameter_batches

    def circuit(angle):
        return (
            oq.Circuit(2)
            .ry(angle, 0)
            .measure(0, 0)
            .if_else(
                oq.ClassicalCondition((0,), 1),
                oq.Circuit(2).ry(angle + 0.2, 1),
                oq.Circuit(2).rx(-angle, 0).rz(angle / 3, 1),
            )
            .with_global_phase(angle)
        )

    circuits = [circuit(0.2), circuit(-0.4)]
    obs = (oq.PauliString("IZ"),)
    reference = oq.expectations_batch(circuits, obs, backend="numpy")

    def no_repeated_rewrite(*args, **kwargs):
        pytest.fail("Preparing a parameter batch must traverse each block only once.")

    with monkeypatch.context() as scoped:
        scoped.setattr(oq.Circuit, "replace_operation", no_repeated_rewrite)
        rows, prototype, bindings = next(parameter_batches(circuits))
    matrix = binding_matrix(bindings)
    assert rows == (0, 1) and matrix.shape == (2, 4) and matrix.flags.c_contiguous
    assert all(np.shares_memory(matrix, column) for column in bindings.values())
    assert binding_matrix(bindings, tuple(bindings)) is matrix
    names = tuple(reversed(tuple(bindings)))
    np.testing.assert_array_equal(binding_matrix(bindings, names), matrix[:, ::-1])
    prepared = [
        prototype.bind({name: column[i] for name, column in bindings.items()}) for i in rows
    ]
    np.testing.assert_allclose(
        oq.expectations_batch(prepared, obs, backend="numpy"), reference, atol=2e-12
    )


def test_native_parameter_staging_uses_one_packed_conversion(monkeypatch):
    torch = pytest.importorskip("torch")
    from openqml.interop._parameter_batches import parameter_batches
    from openqml.sim._cuda import _native_parameters

    circuits = [oq.Circuit(1).ry(x, 0).rz(2 * x, 0) for x in (0.1, 0.2, 0.3)]
    _, _, bindings = next(parameter_batches(circuits))
    calls, original = [], torch.Tensor.to

    def convert(tensor, *args, **kwargs):
        calls.append(tuple(tensor.shape))
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", convert)
    parameters = _native_parameters(bindings, 1, 3, torch.empty((), dtype=torch.float32))
    assert calls == [(2, 2)]
    columns = tuple(parameters.values())
    assert len({column.untyped_storage().data_ptr() for column in columns}) == 1
    np.testing.assert_allclose(columns[0].numpy(), [0.2, 0.3], atol=1e-7)
    assert _native_parameters({}, 0, 2, torch.empty(())) == {}
    assert calls == [(2, 2)]


@pytest.mark.parametrize(
    "options", [{"precision": []}, {"precision": "half"}, {"device": 0}, {"device": "cpu"}]
)
def test_cuda_options_are_validated_without_initializing_a_device(options):
    with pytest.raises(oq.ValidationError):
        oq.CUDABackend(**options)


def test_async_estimator_retains_receipts_and_reuses_layout_readouts(monkeypatch):
    pytest.importorskip("qiskit")
    from qiskit.primitives import StatevectorEstimator
    from qiskit.quantum_info import SparsePauliOp

    class Job:
        ready = False
        polls = 0

        def __init__(self, native, identifier):
            self.native, self.identifier = native, identifier

        def job_id(self):
            return self.identifier

        def result(self, timeout=None):
            self.polls += 1
            if not self.ready:
                raise TimeoutError("still queued")
            return self.native.result()

    jobs, native, readouts = [], StatevectorEstimator(seed=17), []
    original = SparsePauliOp.from_list

    def operator(*args, **kwargs):
        readouts.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(SparsePauliOp, "from_list", operator)

    def submit(pubs, **options):
        job = Job(native.run(pubs, **options), f"retained-{len(jobs)}")
        jobs.append(job)
        return job

    backend = oq.QiskitEstimatorBackend(
        SimpleNamespace(run=submit), recovery=oq.RecoveryPolicy(max_attempts=1)
    )
    first = backend.submit_expectations(oq.Circuit(1).ry(0.2, 0), (oq.PauliString("Z"),))
    second = backend.submit_expectations(oq.Circuit(1).ry(0.7, 0), (oq.PauliString("Z"),))
    assert len(jobs) == 2 and [job.polls for job in jobs] == [0, 0]
    assert len(readouts) == 1
    receipt = first.to_dict()
    with pytest.raises(oq.BackendJobError) as failure:
        first.result()
    assert failure.value.job_id == receipt["job_id"]
    jobs[0].ready = jobs[1].ready = True
    restored = oq.recover_job(receipt, lambda identifier: jobs[0])
    np.testing.assert_allclose(restored, [[np.cos(0.2)]], atol=2e-12)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda _: second.result(), range(3)))
    assert all(result is results[0] for result in results)
    np.testing.assert_allclose(results[0], [[np.cos(0.7)]], atol=2e-12)
    assert jobs[1].polls == 1 and len(jobs) == 2
    assert backend.with_shots(100)._readouts is backend._readouts
    backend.clear_cache()
    backend.submit_expectations(oq.Circuit(1).ry(0.4, 0), (oq.PauliString("Z"),))
    assert len(readouts) == 2


def test_async_sampler_waits_only_on_result_and_never_replays_unknown_submission():
    pytest.importorskip("qiskit")
    from qiskit.primitives import StatevectorSampler

    calls, polls, native = [], [], StatevectorSampler(seed=17)

    def submit(pubs, **options):
        calls.append(pubs)
        job = native.run(pubs, **options)

        def result():
            polls.append(1)
            return job.result()

        return SimpleNamespace(job_id=lambda: "sampler-retained", result=result)

    pending = oq.QiskitSamplerBackend(SimpleNamespace(run=submit)).submit_samples(
        oq.Circuit(2).x(1), 23
    )
    assert len(calls) == 1 and not polls
    assert pending.to_dict()["job_id"] == "sampler-retained"
    assert pending.result() == [{"01": 23}] and len(polls) == 1
    attempts = []

    def unknown(pubs, **options):
        attempts.append(1)
        raise ConnectionError("submission acknowledgement lost")

    backend = oq.QiskitSamplerBackend(SimpleNamespace(run=unknown))
    with pytest.raises(oq.BackendJobError) as failure:
        backend.submit_samples(oq.Circuit(1), 19)
    assert attempts == [1] and failure.value.stage == "submission"
    assert failure.value.pending is None


def test_provider_submission_rejects_invalid_inputs_before_sdk_execution():
    calls = []
    backend = oq.QiskitEstimatorBackend(SimpleNamespace(run=lambda *a, **k: calls.append(a)))
    obs = (oq.PauliString("Z"),)
    for circuits in (None, [], [object()], oq.Circuit(1).ry(oq.Parameter("x"), 0)):
        with pytest.raises(oq.ValidationError):
            backend.submit_expectations(circuits, obs)
    with pytest.raises(oq.ValidationError):
        backend.submit_expectations(oq.Circuit(1), obs, options={})
    with pytest.raises(oq.ResourceError):
        backend.submit_expectations(
            oq.Circuit(1), obs, options=oq.ExecutionOptions(max_memory_mb=1e-12)
        )
    receipt = {"schema": "openqml.provider_job.v1", "job_id": "id", "backend": "b"}
    for metadata in ({"kind": []}, {"kind": "other"}):
        with pytest.raises(oq.ValidationError):
            oq.recover_job(dict(receipt, metadata=metadata), lambda _: None)
    assert not calls


def test_async_submission_pins_policy_before_caller_primitive_reconfigures(monkeypatch):
    pytest.importorskip("qiskit")
    from importlib import import_module

    kernel_module = import_module("openqml.compute.kernel")
    original = oq.get_compute_kernel()
    monkeypatch.setattr(kernel_module, "_KERNEL", original)

    def submit(*args, **kwargs):
        oq.configure_compute(policy=oq.ComputePolicy(validation_atol=0.1))
        return SimpleNamespace(job_id=lambda: "policy-pinned", result=lambda: [])

    pending = oq.QiskitEstimatorBackend(SimpleNamespace(run=submit)).submit_expectations(
        oq.Circuit(1), (oq.PauliString("Z"),)
    )
    assert oq.get_compute_kernel() is not original
    assert pending.kernel is original


def _physical_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        if os.environ.get("OPENQML_REQUIRE_CUDA") == "1":
            pytest.fail("Hardware acceptance requires a physical CUDA GPU.")
        pytest.skip("A physical CUDA GPU is required")
    return torch


def _report(torch, name, **checks):
    directory = os.environ.get("OPENQML_HARDWARE_REPORT_DIR")
    if directory:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        payload = dict(
            device=properties.name,
            device_uuid=str(getattr(properties, "uuid", "unavailable")),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            **checks,
        )
        (path / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("compile_mode", [None, "default", "reduce-overhead"])
def test_physical_cuda_readouts_sampling_and_resident_gradients(precision, compile_mode):
    torch = _physical_cuda()
    engine = oq.CUDABackend(
        precision=precision,
        device=f"cuda:{torch.cuda.current_device()}",
        compile_mode=compile_mode,
    )
    x = oq.Parameter("x")
    circuit = oq.Circuit(4).h(0).ry(x, 3).cx(3, 1).rz(x / 2, 1)
    observables = (oq.PauliString("IIIZ"), oq.PauliString("ZIII"))
    tolerance = 3e-5 if precision == "float32" else 2e-12
    for offset in (0.0, 0.2, -0.3, 0.5):
        rows = [{"x": float(v)} for v in np.linspace(-1, 1, 35) + offset]
        np.testing.assert_allclose(
            oq.expectations_batch(circuit, observables, rows, backend=engine),
            oq.expectations_batch(circuit, observables, rows, backend="numpy"),
            atol=tolerance,
        )
    results = oq.execute_batch([oq.Circuit(2).x(1)] * 3, backend=engine, shots=31)
    assert [result.counts for result in results] == [{"01": 31}] * 3
    states = [oq.Circuit(1).ry(0.2, 0), oq.Circuit(2).h(1)]
    for result, state in zip(oq.execute_batch(states, backend=engine), states, strict=True):
        np.testing.assert_allclose(result.statevector, oq.statevector(state), atol=tolerance)
    angles = torch.tensor([0.2, -0.4], device=engine.device, dtype=getattr(torch, precision))
    angles.requires_grad_()
    values = oq.expectations(circuit, (observables[0],), {"x": angles}, backend="torch")
    assert values.device == angles.device
    gradient = torch.autograd.grad(values.sum(), angles)[0]
    torch.testing.assert_close(gradient, -angles.sin(), atol=tolerance, rtol=tolerance)
    _report(
        torch,
        f"cuda-{precision}-{compile_mode or 'eager'}",
        precision=precision,
        compile_mode=compile_mode,
        readouts=True,
        gradients=True,
    )


def test_physical_cuda_factor_allocation_and_implicit_gradient():
    torch = _physical_cuda()
    from openqml import finance

    returns = np.random.default_rng(17).normal(size=(4, 32)) * 0.005
    mean = np.random.default_rng(18).normal(size=32) * 0.0001
    plan = finance.AllocationPlan(ridge=0.1, turnover=0, tolerance=1e-5)

    def evaluate(device, dtype):
        samples = torch.tensor(returns, device=device, dtype=dtype)
        mu = torch.tensor(mean, device=device, dtype=dtype, requires_grad=True)
        weights = plan.solve(mu, finance.factor_risk(samples)).weights
        gradient = torch.autograd.grad(weights.square().sum(), mu)[0]
        return weights.detach(), gradient

    actual, gradient = evaluate("cuda", torch.float32)
    expected, expected_gradient = evaluate("cpu", torch.float64)
    assert actual.device.type == gradient.device.type == "cuda"
    np.testing.assert_allclose(actual.cpu(), expected, atol=3e-5)
    np.testing.assert_allclose(gradient.cpu(), expected_gradient, atol=3e-4)
    _report(torch, "cuda-allocation", factor_allocation=True, implicit_gradient=True)


def test_physical_cuda_native_aer_engine():
    if os.environ.get("OPENQML_REQUIRE_AER_GPU") != "1":
        pytest.skip("Native Aer GPU validation is explicitly enabled on the hardware runner")
    torch = _physical_cuda()
    engine = oq.AerBackend(device="GPU", runtime_parameter_binding=True)
    assert engine.is_available(), "The hardware runner must install qiskit-aer-gpu."
    x = oq.Parameter("x")
    circuit = oq.Circuit(5).h(0).ry(x, 4).cx(4, 2).rz(x / 3, 2)
    obs, rows = (oq.PauliString("IIIIZ"),), [{"x": 0.2}, {"x": -0.7}]
    np.testing.assert_allclose(
        oq.expectations_batch(circuit, obs, rows, backend=engine),
        oq.expectations_batch(circuit, obs, rows, backend="numpy"),
        atol=2e-12,
    )
    from openqml.interop.adapters.qiskit import to_qiskit

    native = to_qiskit(circuit.bind(rows[0]))
    engine._add_readouts(native, obs)
    result = engine._run([native], oq.ExecutionOptions())
    assert result.results[0].metadata["device"] == "GPU"
    _report(torch, "aer-native-gpu", native_metadata=result.results[0].metadata)
