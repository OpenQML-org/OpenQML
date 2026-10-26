"""Physical maps and retained classical semantics across real SDK adapters."""

import numpy as np
import pytest

from openqml import (
    AerBackend,
    Circuit,
    ClassicalCondition,
    ConversionError,
    Parameter,
    PauliString,
    QuantumProgram,
    ShotBackend,
    ValidationError,
    convert,
    density_matrix,
    diagnose,
    estimate,
    expectation,
    expectations,
    expectations_batch,
    parameter_shift,
    sample,
)


def noisy_circuit():
    rng = np.random.default_rng(72)
    isometry, _ = np.linalg.qr(rng.normal(size=(8, 4)) + 1j * rng.normal(size=(8, 4)))
    return (
        Circuit(3)
        .h(0)
        .ry(0.52, 1)
        .cx(0, 2)
        .p(0.31, 2)
        .amplitude_damping(0.19, 2)
        .phase_damping(0.27, 0)
        .depolarizing(0.17, 1)
        .bit_flip(0.23, 0)
        .phase_flip(0.12, 2)
        .channel(isometry.reshape(2, 4, 4), 2, 0)
        .reset(1)
        .rxx(0.37, 0, 2)
    )


@pytest.mark.parametrize("source", ["qiskit", "pennylane", "cirq"])
@pytest.mark.parametrize("target", ["qiskit", "pennylane", "cirq"])
def test_channel_and_reset_cross_sdk_pairs_preserve_density(source, target):
    pytest.importorskip(source)
    pytest.importorskip(target)
    c = noisy_circuit()
    native_source = convert(c, target=source).value
    ir = convert(native_source, source=source, target="ir").value
    native_target = convert(ir, target=target).value
    restored = convert(native_target, source=target, target="ir").value
    np.testing.assert_allclose(density_matrix(restored), density_matrix(c), atol=2e-12)
    np.testing.assert_allclose(
        density_matrix(restored, backend=target), density_matrix(c), atol=2e-12
    )


@pytest.mark.parametrize("framework", ["qiskit", "pennylane", "cirq"])
@pytest.mark.parametrize(
    "name",
    [
        "amplitude_damping",
        "phase_damping",
        "depolarizing",
        "bit_flip",
        "phase_flip",
    ],
)
def test_native_named_noise_import_has_same_probability_convention(framework, name):
    sdk = pytest.importorskip(framework)
    c = Circuit(1).ry(0.43, 0).rz(0.3, 0)
    native = convert(c, target=framework).value
    if framework == "qiskit":
        noise = pytest.importorskip("qiskit_aer.noise")
        if name == "depolarizing":
            # Aer uses E(rho)=(1-lambda)rho+lambda I/2, while the IR p
            # is the total probability of a nonidentity Pauli error.
            error = noise.depolarizing_error(4 * 0.23 / 3, 1)
        elif name in {"bit_flip", "phase_flip"}:
            error = noise.pauli_error([("X" if name == "bit_flip" else "Z", 0.23), ("I", 0.77)])
        else:
            error = getattr(noise, name + "_error")(0.23)
        native.append(error.to_instruction(), [0])
    elif framework == "pennylane":
        classes = {
            "amplitude_damping": "AmplitudeDamping",
            "phase_damping": "PhaseDamping",
            "depolarizing": "DepolarizingChannel",
            "bit_flip": "BitFlip",
            "phase_flip": "PhaseFlip",
        }
        native = sdk.tape.QuantumScript(
            list(native.operations) + [getattr(sdk, classes[name])(0.23, wires=0)],
            [],
        )
    else:
        classes = {
            "amplitude_damping": "amplitude_damp",
            "phase_damping": "phase_damp",
            "depolarizing": "depolarize",
            "bit_flip": "bit_flip",
            "phase_flip": "phase_flip",
        }
        native.append(getattr(sdk, classes[name])(0.23).on(sdk.LineQubit(0)))
    ir = convert(native, source=framework, target="ir").value
    np.testing.assert_allclose(
        density_matrix(ir),
        density_matrix(getattr(c, name)(0.23, 0)),
        atol=2e-12,
    )


def test_cirq_multi_qubit_channel_falls_back_to_kraus_and_recording_diagnoses():
    cirq = pytest.importorskip("cirq")
    q = cirq.LineQubit.range(2)
    native = cirq.Circuit(cirq.H(q[0]), cirq.CNOT(*q), cirq.depolarize(0.2, n_qubits=2).on(*q))
    ir = convert(native, source="cirq", target="ir").value
    assert ir.operations[-1].name == "kraus"
    expected = (
        cirq.DensityMatrixSimulator(dtype=np.complex128).simulate(native).final_density_matrix
    )
    np.testing.assert_allclose(density_matrix(ir), expected, atol=1e-13)
    native = cirq.Circuit(cirq.KrausChannel([np.eye(2)], key="which_kraus").on(q[0]))
    with pytest.raises(ConversionError, match="CHANNEL_RECORD_UNSUPPORTED"):
        convert(native, source="cirq", target="ir")


@pytest.mark.parametrize("value", range(4))
def test_qiskit_and_qasm3_multibit_feedback_is_atomic(value):
    pytest.importorskip("qiskit_qasm3_import")
    c = (
        Circuit(3)
        .h(0)
        .h(1)
        .measure(0, 2)
        .measure(1, 0)
        .if_else(
            ClassicalCondition((2, 0), value),
            Circuit(3).x(0).measure(0, 2).x(2),
            Circuit(3).ry(0.4, 2),
        )
        .measure(2, 1)
    )
    for target in ("qiskit", "openqasm3"):
        exported = convert(c, target=target).value
        ir = convert(exported, source=target, target="ir").value
        np.testing.assert_allclose(density_matrix(ir), density_matrix(c), atol=2e-13)
        # Compare the joint distribution of classical storage, not just rho.
        a, b = sample(c, 30_000, seed=4), sample(ir, 30_000, seed=4)
        for bits in set(a) | set(b):
            assert a.get(bits, 0) / 30_000 == pytest.approx(b.get(bits, 0) / 30_000, abs=0.02)


def test_symbol_identity_nested_global_phases_and_qiskit_control_budget():
    pytest.importorskip("qiskit")
    x = Parameter("x")
    c = (
        Circuit(2)
        .ry(x, 0)
        .measure(0, 0)
        .if_else(
            ClassicalCondition((0,), 1),
            Circuit(2).ry(x * x, 1).with_global_phase(x.sin()),
            Circuit(2).rx(x, 1),
        )
    )
    native = convert(c, target="qiskit").value
    assert len(native.parameters) == 1
    ir = convert(native, source="qiskit", target="ir").value
    np.testing.assert_allclose(density_matrix(c, {"x": 0.3}), density_matrix(ir, {"x": 0.3}))
    for count in (1, 2, 3):
        with pytest.raises(ConversionError, match="EXPANSION_LIMIT"):
            convert(native, source="qiskit", target="ir", source_options={"max_operations": count})


@pytest.mark.parametrize("backend", ["qiskit", "pennylane", "cirq", "aer-density", "cirq-density"])
def test_exact_postmeasurement_density_is_ensemble_not_single_trajectory(backend):
    pytest.importorskip({"aer-density": "qiskit_aer", "cirq-density": "cirq"}.get(backend, backend))
    c = Circuit(1).h(0).measure(0, 0).h(0).amplitude_damping(0.2, 0)
    expected = np.diag([0.6, 0.4])
    np.testing.assert_allclose(density_matrix(c, backend=backend), expected, atol=1e-12)
    np.testing.assert_allclose(
        expectations(c, [PauliString("X"), PauliString("Z")], backend=backend), [0, 0.2], atol=1e-12
    )
    if backend in {"aer-density", "cirq-density"}:
        controlled = c.if_else(ClassicalCondition((0,), 1), Circuit(1).x(0))
        with pytest.raises(ValidationError, match="feedback|classical"):
            expectation(controlled, PauliString("Z"), backend=backend)


def test_aer_density_readouts_reuse_state_and_channel_gradients(monkeypatch):
    pytest.importorskip("qiskit_aer")
    engine, original = AerBackend(method="density_matrix"), AerBackend.run_density
    calls = []

    def counted(self, circuit, *, options):
        calls.append(circuit)
        return original(self, circuit, options=options)

    monkeypatch.setattr(AerBackend, "run_density", counted)
    c = noisy_circuit()
    obs = [PauliString("XIZ"), PauliString("YIX"), PauliString("ZZZ")]
    np.testing.assert_allclose(
        expectations_batch([c, c], obs, backend=engine), expectations_batch([c, c], obs), atol=1e-12
    )
    assert len(calls) == 2
    p = Parameter("p")
    c = Circuit(1).ry(0.7, 0).amplitude_damping(p, 0)
    actual = parameter_shift(c, PauliString("X"), {"p": 0.23}, backend=engine)
    assert actual["p"] == pytest.approx(-np.sin(0.7) / (2 * np.sqrt(0.77)), abs=1e-11)


@pytest.mark.parametrize("backend", ["numpy", "aer", "aer-density"])
def test_feedback_shots_and_observable_estimator_keep_classical_storage(backend):
    if backend != "numpy":
        pytest.importorskip("qiskit_aer")
    c = (
        Circuit(2)
        .h(0)
        .measure(0, 3)
        .if_else(
            ClassicalCondition((3,), 1),
            Circuit(2).x(0).measure(0, 3).x(1),
        )
        .amplitude_damping(0.2, 1)
    )
    obs = [PauliString("ZI"), PauliString("IZ"), PauliString("ZZ")]
    result = estimate(c, obs, backend=backend, shots=10_000, seed=7)
    np.testing.assert_allclose(result.values[0], [1, 0.2, 0.2], atol=0.04)
    assert result.shots_per_circuit == 10_000
    np.testing.assert_allclose(
        expectations(c, obs, backend=ShotBackend(backend, 10_000, seed=7)), [1, 0.2, 0.2], atol=0.04
    )


def test_qasm_noise_and_control_diagnostics_have_recursive_locations():
    c = Circuit(1).if_else(
        ClassicalCondition((0,), 1),
        Circuit(1).amplitude_damping(Parameter("p"), 0),
    )
    for target in ("openqasm2", "openqasm3"):
        diagnostics = diagnose(c, target)
        channel = next(d for d in diagnostics if d.code == "CHANNEL_UNSUPPORTED")
        assert "blocks" in channel.location
        with pytest.raises(ConversionError, match="CHANNEL_UNSUPPORTED"):
            convert(c, target=target)
    for target in ("qiskit", "pennylane", "cirq"):
        assert "UNBOUND_CHANNEL" in {d.code for d in diagnose(c, target)}
    for target in ("pennylane",):
        assert "CONTROL_FLOW_UNSUPPORTED" in {d.code for d in diagnose(c, target)}
    assert "CONTROL_FLOW_UNSUPPORTED" not in {d.code for d in diagnose(c, "cirq")}
    with pytest.raises(ConversionError):
        convert("OPENQASM 3.0; qubit q; amplitude_damping(0.2) q;", source="openqasm3", target="ir")


def test_qasm3_feedback_still_retains_unsupported_loop_diagnostic():
    pytest.importorskip("qiskit_qasm3_import")
    source = (
        'OPENQASM 3.0; include "stdgates.inc"; qubit q; bit c; '
        "c = measure q; while(c) { x q; c = measure q; }"
    )
    with pytest.raises(ConversionError, match="CLASSICAL_CONTROL"):
        QuantumProgram.from_qasm(source).to_ir()


@pytest.mark.parametrize("framework", ["qiskit", "cirq"])
def test_timing_operations_cannot_silently_fall_back_to_identity_matrix(framework):
    sdk = pytest.importorskip(framework)
    if framework == "qiskit":
        native = sdk.QuantumCircuit(1)
        native.delay(17, 0, unit="ns")
    else:
        native = sdk.Circuit(sdk.wait(sdk.LineQubit(0), nanos=17))
    with pytest.raises(ConversionError, match="TIMING_UNSUPPORTED"):
        convert(native, source=framework, target="ir")


def test_qiskit_custom_gate_cannot_impersonate_kraus_instruction():
    sdk = pytest.importorskip("qiskit")
    gate = sdk.circuit.Gate("kraus", 1, [])
    gate.definition = sdk.QuantumCircuit(1)
    gate.definition.h(0)
    native = sdk.QuantumCircuit(1)
    native.append(gate, [0])
    restored = convert(native, source="qiskit", target="ir").value
    np.testing.assert_allclose(density_matrix(restored), density_matrix(Circuit(1).h(0)))


@pytest.mark.parametrize("kind", ["aer", "cirq"])
def test_measurement_rewrite_cannot_change_provider_native_noise_placement(kind):
    if kind == "aer":
        noise = pytest.importorskip("qiskit_aer.noise")
        model = noise.NoiseModel()
        model.add_all_qubit_quantum_error(noise.amplitude_damping_error(0.2), ["measure"])
        backend = AerBackend(method="density_matrix", noise_model=model)
    else:
        cirq = pytest.importorskip("cirq")
        from openqml import CirqDensityBackend

        backend = CirqDensityBackend(noise=cirq.depolarize(0.1))
    with pytest.raises(ValidationError, match="noise placement"):
        density_matrix(Circuit(1).h(0).measure(0, 0), backend=backend)
