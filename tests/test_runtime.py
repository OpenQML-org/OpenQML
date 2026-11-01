"""Dispatch contracts and provider parameter broadcasting, without cloud jobs."""

from dataclasses import replace

import numpy as np
import pytest

import openqml as oq
from openqml import backends


class Device:
    def __init__(
        self, name="configured-qpu", *, kind="QPU", available=True, fail=False, width=None
    ):
        self.name, self.available, self.fail, self.calls = name, available, fail, []
        self.capabilities = oq.BackendCapabilities(
            device_type=kind,
            native_expectations=True,
            native_sampling=True,
            max_qubits=width,
            deterministic_expectations=kind != "QPU",
            exact_expectations=kind != "QPU",
        )

    def is_available(self):
        return self.available

    def expectations_batch(self, circuits, observables, *, options):
        self.calls.append((len(circuits), "expectations"))
        if self.fail:
            raise oq.BackendError("Accepted job failed; do not resubmit.")
        return oq.expectations_batch(circuits, observables, backend="numpy", **options.__dict__)

    def sample_batch(self, circuits, shots, *, options):
        self.calls.append((len(circuits), "sampling"))
        return [
            r.counts
            for r in oq.execute_batch(circuits, shots=shots, backend="numpy", **options.__dict__)
        ]


@pytest.fixture
def hardware():
    device = Device()
    oq.register_qpu_backend(device)
    try:
        yield device
    finally:
        oq.unregister_backend(device.name)


def test_default_runtime_prefers_qpu_and_keeps_explicit_cpu_pin(hardware, monkeypatch):
    gpu = Device("gpu", kind="GPU")
    monkeypatch.setitem(backends._REGISTRY, "gpu", gpu)
    circuit, observable = oq.Circuit(1).ry(0.4, 0), oq.PauliString("Z")
    assert oq.expectation(circuit, observable) == pytest.approx(np.cos(0.4))
    assert hardware.calls == [(1, "expectations")] and gpu.calls == []
    assert oq.sample(oq.Circuit(1).x(0), 11) == {"1": 11}
    assert hardware.calls[-1] == (1, "sampling")
    assert oq.expectation(circuit, observable, backend="numpy") == pytest.approx(np.cos(0.4))
    assert len(hardware.calls) == 2
    route = oq.get_backend("auto").plan([circuit], observables=(observable,))
    assert route.device == "QPU" and route.to_dict()["backend"] == hardware.name


def test_unavailable_and_incompatible_qpu_falls_to_gpu_then_cpu(hardware, monkeypatch):
    gpu = Device("gpu", kind="GPU")
    monkeypatch.setitem(backends._REGISTRY, "gpu", gpu)
    hardware.available = False
    # The wrapper's caller-owned availability callback is explicit.
    oq.unregister_backend(hardware.name)
    oq.register_qpu_backend(hardware, available=hardware.is_available)
    circuit, obs = oq.Circuit(2).ry(0.6, 0), (oq.PauliString("ZI"),)
    route = oq.get_backend("auto").plan([circuit], observables=obs)
    assert route.device == "GPU" and route.candidates[0].eligible is False
    np.testing.assert_allclose(oq.expectations(circuit, obs), [np.cos(0.6)])
    assert gpu.calls == [(1, "expectations")] and hardware.calls == []
    gpu.available = False
    route = oq.get_backend("auto").plan([circuit], observables=obs)
    assert route.device == "CPU"
    np.testing.assert_allclose(oq.expectations(circuit, obs), [np.cos(0.6)])
    hardware.available = True
    hardware.capabilities = replace(hardware.capabilities, max_qubits=1)
    assert oq.get_backend("auto").plan([circuit], observables=obs).device == "CPU"
    # Explicit software caps are rejected before any provider submission.
    with pytest.raises(oq.ResourceError):
        oq.expectations(circuit, obs, max_qubits=1)
    assert hardware.calls == []


def test_provider_failure_never_retries_on_gpu_or_cpu(hardware, monkeypatch):
    gpu = Device("gpu", kind="GPU")
    monkeypatch.setitem(backends._REGISTRY, "gpu", gpu)
    hardware.fail = True
    with pytest.raises(oq.BackendError, match="do not resubmit"):
        oq.expectation(oq.Circuit(1), oq.PauliString("Z"))
    assert hardware.calls == [(1, "expectations")] and gpu.calls == []


def test_qpu_is_never_selected_for_unsupported_state_or_exactness(hardware, monkeypatch):
    gpu = Device("gpu", kind="GPU", available=False)
    monkeypatch.setitem(backends._REGISTRY, "gpu", gpu)
    circuit = oq.Circuit(1).h(0)
    plan = oq.PriorityBackend(require_exact=True).plan(
        [circuit], observables=(oq.PauliString("X"),)
    )
    assert plan.device == "CPU" and "Exact" in plan.candidates[0].reason
    np.testing.assert_allclose(oq.statevector(circuit), [2**-0.5, 2**-0.5])
    assert hardware.calls == []


def test_auto_preserves_numpy_broadcasted_bindings_and_density(hardware, monkeypatch):
    monkeypatch.setitem(backends._REGISTRY, "gpu", Device("gpu", kind="GPU", available=False))
    x = np.array([[0.1], [0.2]])
    y = np.array([[0.3, 0.4, 0.5]])
    circuit = oq.Circuit(1).ry(oq.Parameter("x") + oq.Parameter("y"), 0)
    values = oq.expectations(circuit, (oq.PauliString("Z"),), {"x": x, "y": y})
    np.testing.assert_allclose(values[..., 0], np.cos(x + y))
    assert hardware.calls == [(6, "expectations")]
    reference = oq.statevector(circuit, {"x": x, "y": y}, backend="numpy")
    np.testing.assert_allclose(oq.statevector(circuit, {"x": x, "y": y}), reference)
    np.testing.assert_allclose(
        oq.density_matrix(circuit, {"x": x, "y": y}),
        np.einsum("...i,...j->...ij", reference, reference.conj()),
    )


def test_shift_engine_pins_qpu_for_primal_and_all_shift_points(hardware, monkeypatch):
    monkeypatch.setitem(backends._REGISTRY, "gpu", Device("gpu", kind="GPU", available=False))
    circuit = oq.Circuit(1).ry(oq.Parameter("x"), 0)
    plan = oq.ShiftPlan(circuit, (oq.PauliString("Z"),))
    assert plan.backend.name == hardware.name and plan.coalesce is False
    values = plan.jacobian([{"x": 0.2}, {"x": 0.2}], chunk_size=16)
    np.testing.assert_allclose(values[:, 0, 0], -np.sin(0.2), atol=1e-14)
    assert sum(n for n, task in hardware.calls) == 4


def test_quantum_rl_policy_default_routes_forward_and_backward_to_qpu(hardware, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setitem(backends._REGISTRY, "gpu", Device("gpu", kind="GPU", available=False))
    model = oq.torch.QuantumActorCritic(2, 2, num_qubits=2, depth=1, device="cpu")
    x = torch.randn(3, 2, requires_grad=True)
    output = model(x)
    (output.policy.square().sum() + output.value.square().sum()).backward()
    assert len(hardware.calls) >= 2 and all(task == "expectations" for _, task in hardware.calls)
    assert x.grad is not None and model.quantum.weights.grad is not None


def test_cpu_topology_cache_is_shared_across_parameter_shift_rows(monkeypatch):
    import openqml.adaptive as adaptive
    from openqml.adaptive import AdaptiveBackend

    calls, original = [], adaptive.plan_execution

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(adaptive, "plan_execution", counted)
    cpu = AdaptiveBackend(name="cpu")
    circuit = oq.Circuit(2).ry(oq.Parameter("x"), 0).rzz(oq.Parameter("y"), 0, 1)
    obs = (oq.PauliString("XI"), oq.PauliString("ZI"))
    rows = [{"x": x, "y": x * 0.3} for x in np.linspace(-1, 1, 31)]
    for _ in range(2):
        result = oq.expectations_batch(circuit, obs, rows, backend=cpu)
        np.testing.assert_allclose(
            result, oq.expectations_batch(circuit, obs, rows, backend="numpy"), atol=2e-14
        )
    assert len(calls) == 1


def test_estimator_parameter_pubs_transpiles_once_and_caches_no_results():
    pytest.importorskip("qiskit")
    from qiskit.primitives import StatevectorEstimator

    class Estimator:
        def __init__(self):
            self.engine, self.calls = StatevectorEstimator(), []

        def run(self, pubs, **kwargs):
            self.calls.append(pubs)
            return self.engine.run(pubs, **kwargs)

    class Manager:
        calls = 0

        def run(self, circuit):
            self.calls += 1
            return circuit

    engine, manager = Estimator(), Manager()
    provider = oq.QiskitEstimatorBackend(engine, precision=0, pass_manager=manager, cache_size=2)
    x = oq.Parameter("x")
    circuit = oq.Circuit(2).ry(x, 0).rzz(x * 0.3, 0, 1)
    rows = [{"x": 0.2}, {"x": -0.4}, {"x": 0.2}]
    obs = (oq.PauliString("XI"), oq.PauliString("ZI"))
    for _ in range(2):
        np.testing.assert_allclose(
            oq.expectations_batch(circuit, obs, rows, backend=provider),
            oq.expectations_batch(circuit, obs, rows, backend="numpy"),
            atol=1e-14,
        )
    assert manager.calls == 1 and len(engine.calls) == 2
    assert all(len(pubs) == 1 and pubs[0][2].shape == (3, 2) for pubs in engine.calls)
    provider.clear_cache()
    oq.expectations_batch(circuit, obs, rows, backend=provider)
    assert manager.calls == 2
    constant = oq.expectations_batch(
        [oq.Circuit(1).x(0)] * 3, (oq.PauliString("Z"),), backend=provider
    )
    np.testing.assert_allclose(constant, -1)


def test_sampler_vectorized_rows_keep_bit_order_and_separate_counts():
    pytest.importorskip("qiskit")
    from qiskit.primitives import StatevectorSampler

    class Sampler:
        def __init__(self):
            self.engine, self.pubs = StatevectorSampler(seed=42), None

        def run(self, pubs, **kwargs):
            self.pubs = pubs
            return self.engine.run(pubs, **kwargs)

    engine = Sampler()
    provider = oq.QiskitSamplerBackend(engine)
    circuit = oq.Circuit(2).ry(oq.Parameter("x"), 0).measure(0, 2).measure(1, 0)
    rows = [{"x": 0.0}, {"x": np.pi}, {"x": 0.0}]
    results = oq.execute_batch(circuit, rows, backend=provider, shots=19)
    assert [r.counts for r in results] == [{"000": 19}, {"001": 19}, {"000": 19}]
    assert len(engine.pubs) == 1 and engine.pubs[0][1].shape == (3, 1)
    results = oq.execute_batch([oq.Circuit(1).x(0)] * 3, backend=provider, shots=7)
    assert [r.counts for r in results] == [{"1": 7}] * 3


def test_actual_cuda_native_readouts_and_sampling_when_hardware_is_available():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware is unavailable")
    circuit = oq.Circuit(3).ry(oq.Parameter("x"), 0).rzz(0.2, 0, 1).cx(1, 2)
    rows = [{"x": float(x)} for x in np.linspace(-1, 1, 17)]
    obs = (oq.PauliString("XII"), oq.PauliString("ZII"))
    np.testing.assert_allclose(
        oq.expectations_batch(circuit, obs, rows, backend="gpu"),
        oq.expectations_batch(circuit, obs, rows, backend="numpy"),
        atol=1e-13,
    )
    assert oq.sample(oq.Circuit(2).x(0), 23, backend="gpu") == {"10": 23}
    measured = oq.Circuit(1).ry(oq.Parameter("x"), 0)
    counts = oq.execute_batch(
        measured, [{"x": 0.0}, {"x": np.pi}, {"x": 0.0}], backend="gpu", shots=23
    )
    assert [r.counts for r in counts] == [{"0": 23}, {"1": 23}, {"0": 23}]
    states = [oq.Circuit(1).ry(x, 0).with_global_phase(x * 0.4) for x in (0.2, 0.3, 0.4)]
    actual = oq.execute_batch(states, backend="gpu")
    for result, circuit in zip(actual, states, strict=True):
        np.testing.assert_allclose(
            result.statevector, oq.statevector(circuit, backend="numpy"), atol=1e-14
        )


def test_accelerator_batch_template_preserves_full_complex_phase_on_native_tensors():
    torch = pytest.importorskip("torch")
    from openqml.batch import _vectorized_states

    circuits = [
        oq.Circuit(2).ry(x, 0).rx(-x * 0.7, 1).cx(0, 1).with_global_phase(0.3 * x)
        for x in (0.1, 0.5, -0.8)
    ]
    values = _vectorized_states(
        circuits, oq.ExecutionOptions(), backend="torch", like=torch.empty((), dtype=torch.float64)
    )
    reference = np.stack([oq.statevector(c, backend="numpy") for c in circuits])
    assert values.device.type == "cpu" and values.shape == (3, 4)
    np.testing.assert_allclose(values.numpy(), reference, atol=1e-14)
