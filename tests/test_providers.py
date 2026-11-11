from types import SimpleNamespace

import numpy as np
import pytest

from openqml import (
    Circuit,
    CirqDensityBackend,
    Hamiltonian,
    Parameter,
    PauliString,
    QiskitEstimatorBackend,
    QiskitSamplerBackend,
    ResourceError,
    estimate,
    expectation,
    expectations_batch,
    parameter_shift_jacobian,
    sample,
)


@pytest.mark.parametrize("value", [1.02, np.nan, np.inf, 1j])
def test_mitigated_estimator_contract_only_relaxes_spectral_bounds(value):
    pytest.importorskip("qiskit")
    from openqml import BackendError, expectations

    class Estimator:
        def run(self, pubs, **options):
            result = [SimpleNamespace(data=SimpleNamespace(evs=np.array([[value]])))]
            return SimpleNamespace(result=lambda: result, job_id=lambda: "mitigated-test")

    circuit, observables = Circuit(1), (PauliString("Z"),)
    with pytest.raises(BackendError):
        expectations_batch(circuit, observables, backend=QiskitEstimatorBackend(Estimator()))
    backend = QiskitEstimatorBackend(Estimator(), mitigated=True)
    if value == 1.02:
        np.testing.assert_allclose(
            expectations_batch(circuit, observables, backend=backend), [[value]]
        )
        np.testing.assert_allclose(expectations(circuit, observables, backend=backend), [value])
        assert not backend.with_shots(100).capabilities.bounded_expectations
    else:
        with pytest.raises(BackendError):
            expectations_batch(circuit, observables, backend=backend)


def test_mitigated_receipt_recovery_preserves_estimate_contract_without_resubmission():
    pytest.importorskip("qiskit")
    from openqml import BackendJobError, RecoveryPolicy, recover_job

    class Job:
        ready = False

        def job_id(self):
            return "retained-mitigated-job"

        def result(self):
            if not self.ready:
                raise TimeoutError("pending")
            return [SimpleNamespace(data=SimpleNamespace(evs=np.array([[1.03]])))]

    job = Job()
    submissions = []

    def submit(pubs, **options):
        submissions.append(pubs)
        return job

    backend = QiskitEstimatorBackend(
        SimpleNamespace(run=submit), mitigated=True, recovery=RecoveryPolicy(max_attempts=1)
    )
    with pytest.raises(BackendJobError) as failure:
        expectations_batch(Circuit(1), (PauliString("Z"),), backend=backend)
    receipt = failure.value.pending.to_dict()
    job.ready = True
    np.testing.assert_allclose(recover_job(receipt, lambda identifier: job), [[1.03]])
    assert len(submissions) == 1


def test_qiskit_v2_estimator_multiple_observables_layout_and_shift():
    pytest.importorskip("qiskit")
    from qiskit.primitives import StatevectorEstimator
    from qiskit.transpiler import CouplingMap
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    circuit = Circuit(2).ry(Parameter("x").sin(), 0).rx(-0.37, 1).cx(0, 1)
    rows = [{"x": 0.41}, {"x": -0.63}]
    obs = (PauliString("YZ"), Hamiltonian((PauliString("ZI", 0.7), PauliString("YY", -0.3))))
    manager = generate_preset_pass_manager(
        optimization_level=0,
        coupling_map=CouplingMap.from_line(3),
        initial_layout=[2, 0],
        basis_gates=["rz", "sx", "x", "cx"],
    )
    backend = QiskitEstimatorBackend(
        StatevectorEstimator(seed=13), precision=0, pass_manager=manager
    )
    np.testing.assert_allclose(
        expectations_batch(circuit, obs, rows, backend=backend),
        expectations_batch(circuit, obs, rows),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        parameter_shift_jacobian(circuit, obs, rows, backend=backend),
        parameter_shift_jacobian(circuit, obs, rows),
        atol=1e-12,
    )


def test_qiskit_v2_sampler_bit_order_idle_bits_and_measurement_estimator():
    pytest.importorskip("qiskit")
    from qiskit.primitives import StatevectorSampler

    backend = QiskitSamplerBackend(StatevectorSampler(seed=19))
    circuit = Circuit(2).x(0).measure(0, 2).measure(1, 0)
    assert sample(circuit, 17, backend=backend) == {"001": 17}
    bell = Circuit(2).h(0).cx(0, 1)
    result = estimate(
        bell,
        (PauliString("XX"), PauliString("YY"), PauliString("ZZ")),
        backend=backend,
        shots=301,
        seed=17,
    )
    np.testing.assert_allclose(result.values, [[1, -1, 1]], atol=1e-12)
    np.testing.assert_allclose(result.standard_errors, 0, atol=1e-12)


def test_cirq_density_noise_and_exact_gradients_match_channel_formula():
    cirq = pytest.importorskip("cirq")
    gamma = 0.17
    backend = CirqDensityBackend(noise=cirq.amplitude_damp(gamma))
    circuit = Circuit(1).ry(Parameter("theta"), 0)
    obs = (PauliString("X"), PauliString("Y"), PauliString("Z"))
    angle = 0.63
    value = expectations_batch(circuit, obs, [{"theta": angle}], backend=backend)[0]
    shifted_phase = expectations_batch(
        circuit.with_global_phase(1.3), obs, [{"theta": angle}], backend=backend
    )[0]
    np.testing.assert_allclose(value, shifted_phase, atol=1e-12)
    np.testing.assert_allclose(
        value,
        [np.sqrt(1 - gamma) * np.sin(angle), 0, (1 - gamma) * np.cos(angle) + gamma],
        atol=1e-12,
    )
    jac = parameter_shift_jacobian(circuit, obs, [{"theta": angle}], backend=backend)[0, :, 0]
    np.testing.assert_allclose(
        jac, [np.sqrt(1 - gamma) * np.cos(angle), 0, -(1 - gamma) * np.sin(angle)], atol=1e-12
    )
    # Imaginary Pauli Y tests rho/P index and conjugation conventions.
    ry = expectation(Circuit(1).rx(angle, 0), PauliString("Y"), backend=backend)
    assert ry == pytest.approx(-np.sqrt(1 - gamma) * np.sin(angle), abs=1e-12)
    with pytest.raises(ResourceError, match="budget"):
        expectation(Circuit(20), PauliString("Z" * 20), backend=backend, max_memory_mb=1)


def test_cirq_native_sampling_replays_mid_circuit_classical_writes():
    pytest.importorskip("cirq")
    circuit = Circuit(2).x(0).measure(0, 1).x(0).measure(0, 1).x(1).measure(1, 0)
    assert sample(circuit, 17, backend="cirq-density", seed=3) == {"10": 17}
    bell = Circuit(2).h(0).measure(0, 0).cx(0, 1).measure(1, 1)
    assert set(sample(bell, 128, backend="cirq-density", seed=13)) <= {"00", "11"}


def test_cirq_noisy_sampling_does_not_insert_idle_gates_per_operation():
    cirq = pytest.importorskip("cirq")
    backend = CirqDensityBackend(noise=cirq.amplitude_damp(0.3))
    circuit = Circuit(2).x(0).x(1).rx(0.17, 0)
    from openqml.adapters.cirq import to_cirq

    native = to_cirq(circuit)
    native.append(cirq.measure(cirq.LineQubit(0), key="c0"), strategy=cirq.InsertStrategy.NEW)
    native.append(cirq.measure(cirq.LineQubit(1), key="c1"), strategy=cirq.InsertStrategy.NEW)
    result = cirq.DensityMatrixSimulator(dtype=np.complex128, noise=backend.noise, seed=19).run(
        native, repetitions=301
    )
    counts = {}
    for a, b in zip(result.measurements["c0"][:, 0], result.measurements["c1"][:, 0], strict=True):
        key = f"{a}{b}"
        counts[key] = counts.get(key, 0) + 1
    assert sample(circuit, 301, backend=backend, seed=19) == counts


def test_noisy_aer_density_quantum_layer_gradcheck():
    torch = pytest.importorskip("torch")
    pytest.importorskip("qiskit_aer")
    from qiskit_aer.noise import NoiseModel, amplitude_damping_error

    from openqml import AerBackend
    from openqml.interfaces.torch import QuantumLayer

    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(amplitude_damping_error(0.17), ["ry"])
    backend = AerBackend(name="noisy-aer", method="density_matrix", noise_model=noise)
    circuit = Circuit(1).ry(Parameter("x") * Parameter("w"), 0)
    layer = QuantumLayer(circuit, ("x",), ("w",), backend=backend, dtype=torch.float64)
    x = torch.tensor([[0.41], [-0.63]], dtype=torch.float64, requires_grad=True)
    w = torch.tensor([0.73], dtype=torch.float64, requires_grad=True)

    def f(x, w):
        return torch.func.functional_call(layer, {"weights": w}, (x,))

    assert torch.autograd.gradcheck(f, (x, w), atol=1e-6)
    torch.testing.assert_close(f(x, w), 0.83 * torch.cos(x * w) + 0.17, atol=1e-12, rtol=1e-12)


def test_aer_compilation_cache_reuses_topology_bounds_and_evicts():
    pytest.importorskip("qiskit_aer")
    from openqml import AerBackend

    engine = AerBackend(compile_cache_size=2)
    circuit = Circuit(2).ry(Parameter("x"), 1).cx(1, 0)
    obs = (PauliString("ZZ"),)
    rows = [{"x": x} for x in [0.17, 0.41, -0.63]]
    np.testing.assert_allclose(
        expectations_batch(circuit, obs, rows, backend=engine),
        expectations_batch(circuit, obs, rows),
        atol=1e-12,
    )
    info = engine.cache_info()
    assert info == {"hits": 0, "misses": 1, "size": 1, "maxsize": 2}
    other_rows = [{"x": x} for x in [0.13, -0.77]]
    np.testing.assert_allclose(
        expectations_batch(circuit, obs, other_rows, backend=engine),
        expectations_batch(circuit, obs, other_rows),
        atol=1e-12,
    )
    assert engine.cache_info()["hits"] == 1
    for word in ["XX", "YY", "IZ"]:
        expectations_batch(circuit, (PauliString(word),), rows, backend=engine)
    assert engine.cache_info()["size"] == 2
    engine.clear_cache()
    assert (
        engine.cache_info()["size"]
        == engine.cache_info()["hits"]
        == engine.cache_info()["misses"]
        == 0
    )


@pytest.mark.parametrize("gate", ["cp", "crx", "cry", "crz", "u"])
def test_aer_cached_parameter_families_match_bound_reference(gate):
    pytest.importorskip("qiskit_aer")
    from openqml import AerBackend

    circuit = Circuit(2).h(0).ry(0.43, 1)
    if gate == "u":
        circuit = circuit.u(Parameter("a"), Parameter("b"), Parameter("a").sin(), 1)
    else:
        circuit = getattr(circuit, gate)(Parameter("a").sin() * Parameter("b"), 0, 1)
    rows = [{"a": 0.41, "b": -0.67}, {"a": -0.73, "b": 0.31}]
    obs = (PauliString("XI"), PauliString("YZ"), PauliString("ZZ"))
    cached = AerBackend()
    bound = AerBackend(parameter_batching=False)
    np.testing.assert_allclose(
        expectations_batch(circuit, obs, rows, backend=cached),
        expectations_batch(circuit, obs, rows, backend=bound),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        parameter_shift_jacobian(circuit, obs, rows, backend=cached),
        parameter_shift_jacobian(circuit, obs, rows),
        atol=1e-12,
    )


def test_noisy_native_controlled_gate_and_cirq_lowered_noise_gradients():
    pytest.importorskip("qiskit_aer")
    cirq = pytest.importorskip("cirq")
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    from openqml import AerBackend

    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(depolarizing_error(0.19, 2), ["cp"])
    circuit = Circuit(2).h(0).ry(0.43, 1).cp(Parameter("theta"), 0, 1)
    obs = (PauliString("XI"), PauliString("YY"), PauliString("YZ"))
    engines = (
        AerBackend(method="density_matrix", noise_model=noise),
        CirqDensityBackend(noise=cirq.depolarize(0.13)),
    )
    for engine in engines:
        angle = 0.63
        gradient = parameter_shift_jacobian(circuit, obs, [{"theta": angle}], backend=engine)
        plus = expectations_batch(circuit, obs, [{"theta": angle + 1e-6}], backend=engine)
        minus = expectations_batch(circuit, obs, [{"theta": angle - 1e-6}], backend=engine)
        np.testing.assert_allclose(gradient[:, :, 0], (plus - minus) / 2e-6, atol=1e-8)
