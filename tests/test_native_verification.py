"""Execute actual exported artifacts independently of the IR importers."""

import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from openqml import (
    Circuit,
    ClassicalCondition,
    ConversionError,
    ConversionPlan,
    Parameter,
    ResourceError,
    ValidationError,
    convert,
    plan_conversion,
)


def require_target(target):
    pytest.importorskip("qiskit" if target.startswith("openqasm") else target)
    if target == "openqasm3":
        pytest.importorskip("qiskit_qasm3_import")


@pytest.mark.parametrize("target", ["qiskit", "cirq", "pennylane", "openqasm2", "openqasm3"])
@pytest.mark.parametrize("chunk", [1, 3])
def test_native_unitary_checks_preserve_phase_wire_order_and_complete_input_basis(target, chunk):
    require_target(target)
    c = Circuit(3).u(0.23, -0.31, 0.4, 2).cx(2, 0).cry(0.37, 1, 0).rx(-0.19, 1)
    if target != "openqasm2":
        c = c.with_global_phase(-0.27)
    plan = plan_conversion(c, target=target)
    before = plan.to_json()
    check = plan.verify(mode="native", basis_chunk_size=chunk)
    assert check.matched and check.checked_inputs == 8 and check.checked_entries == 64
    assert check.mode == "native" and check.max_absolute_error < 3e-12
    assert check.runtime_backend == ("qiskit" if target.startswith("openqasm") else target)
    assert check.to_dict()["scope"] == "native_runtime_at_bindings"
    assert check.to_dict()["basis_indexing"] == "basis_states"
    check.raise_for_errors()
    assert before == plan.to_json()
    with pytest.raises(FrozenInstanceError):
        check.runtime_backend = None


@pytest.mark.parametrize("target", ["qiskit", "cirq", "pennylane"])
def test_native_channel_checks_cover_kraus_reset_coherences_and_reversed_wires(target):
    require_target(target)
    rng = np.random.default_rng(41)
    isometry, _ = np.linalg.qr(rng.normal(size=(8, 4)) + 1j * rng.normal(size=(8, 4)))
    c = (
        Circuit(3)
        .ry(0.32, 2)
        .cx(2, 0)
        .channel(isometry.reshape(2, 4, 4), 2, 0)
        .amplitude_damping(0.2, 1)
        .depolarizing(0.13, 0)
        .phase_damping(0.12, 2)
        .reset(1)
        .with_global_phase(0.4)
    )
    check = plan_conversion(c, target=target).verify(
        mode="native", basis_chunk_size=7, max_work=50_000_000
    )
    assert check.matched and check.checked_inputs == 64 and check.checked_entries == 4096
    data = json.loads(check.to_json())
    assert data["quantum_inputs"] == "spanning_pure_states"
    assert data["basis_indexing"] == "basis_states_then_pair_real_imaginary"
    assert check.semantics == "quantum_channel" and check.max_absolute_error < 3e-12


@pytest.mark.parametrize("target", ["qiskit", "cirq", "pennylane"])
def test_symbolic_native_checks_are_pointwise_and_do_not_retain_bindings(target):
    require_target(target)
    c = (
        Circuit(2)
        .ry(Parameter("x").sin() * Parameter("w"), 1)
        .cx(1, 0)
        .with_global_phase(Parameter("phase"))
    )
    plan = plan_conversion(c, target=target)
    before = plan.to_json()
    for x in (-0.3, 0.2):
        check = plan.verify(dict(x=x, w=0.7, phase=-0.2), mode="native")
        assert check.matched and check.mode == "native"
        assert set(check.to_dict()).isdisjoint({"bindings", "x", "w", "phase"})
    assert plan.to_json() == before


@pytest.mark.parametrize("target", ["qiskit", "cirq", "pennylane"])
@pytest.mark.parametrize("fault", ["unitary", "phase", "channel"])
def test_native_checks_catch_compensating_export_import_errors(target, fault, monkeypatch):
    require_target(target)
    from importlib import import_module

    adapter = import_module(f"openqml.adapters.{target}")
    if fault == "channel":
        source, wrong = Circuit(1).phase_damping(0.2, 0), Circuit(1)
    elif fault == "phase":
        source, wrong = Circuit(1).h(0).with_global_phase(0.3), Circuit(1).h(0)
    else:
        source, wrong = Circuit(1), Circuit(1).z(0)
    native = convert(wrong, target=target).value
    monkeypatch.setattr(adapter, f"to_{target}", lambda *a, **kw: native)
    monkeypatch.setattr(adapter, f"from_{target}", lambda *a, **kw: source)
    plan = plan_conversion(source, target=target)
    assert plan.verify().matched  # A compensating importer masks the bad export.
    monkeypatch.setattr(
        adapter, f"from_{target}", lambda *a, **kw: pytest.fail("Native mode reimported IR")
    )
    check = plan.verify(mode="native")
    assert not check.matched and check.mismatched_entries > 0
    assert check.diagnostics[0].code == "CONVERSION_NATIVE_MISMATCH"
    assert check.worst_input is not None and check.worst_record == ()
    if fault == "unitary":
        assert (check.worst_input, check.worst_output) == (1, 1)
    elif fault == "channel":
        assert check.worst_input >= 2  # Computational-basis density inputs agree.
    with pytest.raises(ConversionError, match="CONVERSION_NATIVE_MISMATCH"):
        check.raise_for_errors()


@pytest.mark.parametrize("target", ["openqasm2", "openqasm3"])
def test_native_qasm_verification_does_not_use_the_ir_qasm_importer(target, monkeypatch):
    require_target(target)
    import openqml.adapters.openqasm as qasm

    monkeypatch.setattr(
        qasm, "from_openqasm", lambda *a, **kw: pytest.fail("Reimported QASM through IR")
    )
    c = Circuit(2).ry(0.3, 1).cx(1, 0)
    assert plan_conversion(c, target=target).verify(mode="native").matched


@pytest.mark.parametrize("width", [1, 2, 3])
def test_physical_channel_probes_span_the_complete_operator_space(width):
    from openqml._native_verification import _probes

    vectors = np.stack(tuple(_probes(width, True)))
    dimension = 1 << width
    assert vectors.shape == (dimension * dimension, dimension)
    np.testing.assert_allclose(np.sum(np.abs(vectors) ** 2, axis=-1), 1, atol=3e-15)
    matrices = (vectors[:, :, None] * vectors[:, None, :].conj()).reshape(dimension * dimension, -1)
    assert np.linalg.matrix_rank(matrices) == dimension * dimension


@pytest.mark.parametrize("mode", [True, 0, None, "runtime", "auto"])
def test_unknown_verification_modes_are_explicit(mode):
    with pytest.raises(ValidationError, match="mode"):
        plan_conversion(Circuit(1), target="ir").verify(mode=mode)


@pytest.mark.parametrize(
    "circuit",
    [
        Circuit(1).measure(0, 0),
        Circuit(1).if_else(ClassicalCondition((0,), 0), Circuit(1).x(0)),
    ],
)
def test_native_measurement_and_feedback_scope_is_rejected_before_export(circuit, monkeypatch):
    plan = plan_conversion(circuit, target="qiskit")
    monkeypatch.setattr(ConversionPlan, "export", lambda *a: pytest.fail("Unexpected export"))
    with pytest.raises(ConversionError, match="NATIVE_VERIFICATION_UNSUPPORTED"):
        plan.verify(mode="native")


def test_native_mode_never_silently_ignores_reimport_options_or_unknown_targets():
    with pytest.raises(ConversionError, match="NATIVE_VERIFICATION_UNSUPPORTED"):
        plan_conversion(Circuit(1), target="json").verify(mode="native")
    with pytest.raises(ValidationError, match="source_options"):
        plan_conversion(Circuit(1), target="qiskit").verify(
            mode="native", source_options={"custom": 1}
        )


@pytest.mark.parametrize("target", ["qiskit", "cirq", "pennylane"])
def test_nonunitary_native_artifacts_cannot_pass_by_stochastic_statevector_sampling(
    target, monkeypatch
):
    require_target(target)
    from importlib import import_module

    native = convert(Circuit(1).bit_flip(0.001, 0), target=target).value
    adapter = import_module(f"openqml.adapters.{target}")
    monkeypatch.setattr(adapter, f"to_{target}", lambda *a, **kw: native)
    with pytest.raises(ConversionError, match="NATIVE_VERIFICATION_UNSUPPORTED"):
        plan_conversion(Circuit(1), target=target).verify(mode="native")


@pytest.mark.parametrize("budget", ["operations", "work"])
def test_native_budget_failures_precede_sdk_execution(budget, monkeypatch):
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    import openqml.adapters.qiskit as adapter

    wrong = QuantumCircuit(1)
    for _ in range(20):
        wrong.x(0)
    plan = plan_conversion(
        Circuit(1), target="qiskit", max_operations=2 if budget == "operations" else 100
    )
    monkeypatch.setattr(adapter, "to_qiskit", lambda *a: wrong)
    monkeypatch.setattr(
        Statevector, "evolve", lambda *a: pytest.fail("SDK executed before preflight")
    )
    with pytest.raises(
        ResourceError, match="max_operations" if budget == "operations" else "max_work"
    ):
        plan.verify(mode="native", max_work=100)


@pytest.mark.parametrize("fault", ["shape", "nan", "exception"])
def test_native_runtime_failures_and_invalid_arrays_are_reported(fault, monkeypatch):
    pytest.importorskip("qiskit")
    import openqml._native_verification as native

    def run(probes):
        if fault == "exception":
            raise RuntimeError("Injected runtime failure")
        return np.zeros((len(probes), 3)) if fault == "shape" else np.full((len(probes), 2), np.nan)

    monkeypatch.setattr(native, "_qiskit", lambda *a: (run, 1))
    error, message = (
        (ConversionError, "NATIVE_VERIFICATION_EXECUTION")
        if fault == "exception"
        else (ValidationError, "shape" if fault == "shape" else "nonfinite")
    )
    with pytest.raises(error, match=message):
        plan_conversion(Circuit(1), target="qiskit").verify(mode="native")


def test_native_memory_budget_reduces_probe_chunks_and_covers_every_input(monkeypatch):
    pytest.importorskip("qiskit")
    import openqml._native_verification as native

    seen, original = [], native._qiskit

    def prepare(*args):
        run, work = original(*args)

        def track(probes):
            seen.append(len(probes))
            return run(probes)

        return track, work

    monkeypatch.setattr(native, "_qiskit", prepare)
    check = plan_conversion(Circuit(1).phase_damping(0.2, 0), target="qiskit").verify(
        mode="native", max_memory_mb=0.0015
    )
    assert check.matched and check.checked_inputs == 4
    assert seen == [1] * 4


def test_invalid_native_artifacts_report_preparation_failures(monkeypatch):
    pytest.importorskip("qiskit")
    import openqml.adapters.qiskit as adapter

    monkeypatch.setattr(adapter, "to_qiskit", lambda *a: object())
    with pytest.raises(ConversionError, match="NATIVE_VERIFICATION_PREPARATION"):
        plan_conversion(Circuit(1), target="qiskit").verify(mode="native")
