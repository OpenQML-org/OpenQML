"""Full-input numerical round-trip checks, including phase and classical records."""

import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from openqml import (
    Circuit,
    ClassicalCondition,
    ConversionError,
    ConversionPlan,
    FrameworkAdapter,
    Parameter,
    ResourceError,
    ValidationError,
    density_matrix,
    plan_conversion,
    register_adapter,
    statevector,
    unregister_adapter,
)


@pytest.fixture
def fault_adapter():
    def install(exporter, importer=None):
        register_adapter(
            FrameworkAdapter("verification-test", importer or (lambda x, **kw: x), exporter)
        )
        return "verification-test"

    yield install
    try:
        unregister_adapter("verification-test")
    except ValidationError:
        pass


@pytest.mark.parametrize(
    "target", ["ir", "json", "openqasm2", "openqasm3", "qiskit", "cirq", "pennylane"]
)
def test_full_unitary_input_basis_round_trip(target):
    if target in {"qiskit", "cirq", "pennylane"}:
        pytest.importorskip(target)
    c = Circuit(3).u(0.2, -0.3, 0.7, 2).cx(2, 0).ry(0.3, 1).crx(-0.4, 1, 2)
    if target != "openqasm2":
        c = c.with_global_phase(0.32)
    plan = plan_conversion(c, target=target)
    before = plan.to_json()
    check = plan.verify(basis_chunk_size=3)
    assert check.matched and check.checked_inputs == 8
    assert check.checked_entries == 64 and check.mismatched_entries == 0
    assert check.semantics == "unitary_including_global_phase"
    assert check.max_absolute_error < 2e-12
    assert plan.to_json() == before
    data = json.loads(check.to_json())
    assert data["scope"] == "adapter_roundtrip_at_bindings"
    assert data["classical_input"] == "all_zero"
    assert data["quantum_inputs"] == "complete_basis"
    with pytest.raises(FrozenInstanceError):
        check.matched = False


@pytest.mark.parametrize("target", ["json", "openqasm3", "qiskit", "cirq", "pennylane"])
def test_binding_verification_does_not_mutate_or_retain_the_symbolic_plan(target):
    if target in {"qiskit", "cirq", "pennylane"}:
        pytest.importorskip(target)
    c = Circuit(2).ry(Parameter("theta").sin(), 0).cx(0, 1).with_global_phase(Parameter("phase"))
    plan = plan_conversion(c, target=target)
    original = plan.to_json()
    for value in (-0.3, 0.7):
        check = plan.verify({"theta": value, "phase": -0.2})
        assert check.matched and check.checked_inputs == 4
        assert "theta" not in check.to_json() and "phase" not in check.to_dict()
    assert original == plan.to_json()
    with pytest.raises(ValidationError, match="Bind all parameters"):
        plan.verify()


@pytest.mark.parametrize("target", ["ir", "json", "qiskit", "cirq", "pennylane"])
def test_channel_maps_cover_off_diagonal_matrix_units_and_reversed_operands(target):
    if target in {"qiskit", "cirq", "pennylane"}:
        pytest.importorskip(target)
    rng = np.random.default_rng(92)
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
    )
    check = plan_conversion(c, target=target).verify(basis_chunk_size=7)
    assert check.matched and check.checked_inputs == 64
    assert check.checked_entries == 4096
    assert check.semantics == "quantum_instrument" and check.num_clbits == 0
    assert check.max_absolute_error < 3e-12


@pytest.mark.parametrize("target", ["ir", "json", "openqasm3", "qiskit", "cirq"])
def test_measurements_and_feedback_preserve_maps_for_each_final_record(target):
    if target in {"qiskit", "cirq"}:
        pytest.importorskip(target)
    c = (
        Circuit(2)
        .h(0)
        .ry(0.4, 1)
        .measure(0, 0)
        .if_else(ClassicalCondition((0,), 1), Circuit(2).x(1), Circuit(2).rz(0.3, 1))
        .reset(0)
    )
    check = plan_conversion(c, target=target).verify(basis_chunk_size=3)
    assert check.matched and check.checked_inputs == 16
    assert check.checked_entries == 2 * 256
    assert check.num_clbits == 1 and check.max_absolute_error < 3e-12


@pytest.mark.parametrize("target", ["json", "openqasm3", "qiskit"])
def test_nested_feedback_overwrites_and_initially_zero_storage_are_checked(target):
    if target == "qiskit":
        pytest.importorskip(target)
    body = (
        Circuit(2)
        .measure(1, 0)
        .if_else(ClassicalCondition((0,), 1), Circuit(2).x(0), Circuit(2).z(1))
    )
    c = Circuit(2).if_else(ClassicalCondition((1,), 0), body, Circuit(2).y(0)).measure(0, 1)
    check = plan_conversion(c, target=target).verify()
    assert check.matched and check.checked_entries == 4 * 256


def test_unitary_comparison_detects_errors_invisible_from_zero_input(fault_adapter):
    c, wrong = Circuit(1).z(0), Circuit(1)
    np.testing.assert_allclose(statevector(c), statevector(wrong))
    check = plan_conversion(c, target=fault_adapter(lambda c: wrong)).verify()
    assert not check.matched and check.mismatched_entries == 1
    assert check.max_absolute_error == 2
    assert (check.worst_input, check.worst_output, check.worst_record) == (1, 1, ())
    assert check.diagnostics[0].code == "CONVERSION_ROUNDTRIP_MISMATCH"
    with pytest.raises(ConversionError, match="CONVERSION_ROUNDTRIP_MISMATCH"):
        check.raise_for_errors()


def test_global_phase_loss_is_reported_without_phase_alignment(fault_adapter):
    c = Circuit(1).h(0).with_global_phase(0.3)
    check = plan_conversion(c, target=fault_adapter(lambda c: Circuit(1).h(0))).verify()
    assert not check.matched and check.mismatched_entries == 4
    assert check.semantics == "unitary_including_global_phase"


def test_channel_comparison_detects_coherences_invisible_from_diagonal_inputs(fault_adapter):
    c, wrong = Circuit(1).phase_damping(0.2, 0), Circuit(1)
    np.testing.assert_allclose(density_matrix(c), density_matrix(wrong))
    check = plan_conversion(c, target=fault_adapter(lambda c: wrong)).verify()
    assert not check.matched and check.mismatched_entries == 2
    assert check.worst_input in (1, 2) and check.worst_record == ()
    assert check.max_absolute_error == pytest.approx(1 - np.sqrt(0.8))


def test_instrument_comparison_detects_record_loss_hidden_by_reset(fault_adapter):
    c = Circuit(1).measure(0, 0).reset(0)
    wrong = Circuit(1).x(0).measure(0, 0).reset(0)
    np.testing.assert_allclose(density_matrix(c), density_matrix(wrong))
    check = plan_conversion(c, target=fault_adapter(lambda c: wrong)).verify()
    assert not check.matched and check.mismatched_entries == 4
    assert check.worst_record in ((), (0,)) and check.max_absolute_error == 1


def test_instrument_semantics_ignore_unobservable_global_phase(fault_adapter):
    c = Circuit(1).phase_damping(0.3, 0).with_global_phase(0.4)
    wrong = Circuit(1).phase_damping(0.3, 0)
    assert plan_conversion(c, target=fault_adapter(lambda c: wrong)).verify().matched


def test_linear_basis_engine_matches_analytic_measure_and_correct_instrument():
    from openqml.planning import _record_plan
    from openqml.verification import _evolve_basis

    c = Circuit(1).measure(0, 0).if_else(ClassicalCondition((0,), 1), Circuit(1).x(0))
    basis = np.eye(4, dtype=complex)
    result = _evolve_basis(c, basis, True, _record_plan(c, True))
    kraus = {frozenset(): np.array([[1, 0], [0, 0]]), frozenset({0}): np.array([[0, 1], [0, 0]])}
    for key, matrix in kraus.items():
        expected = np.stack(
            [(matrix @ e.reshape(2, 2) @ matrix.conj().T).reshape(4) for e in basis]
        )
        np.testing.assert_allclose(result[key], expected, atol=2e-15)


@pytest.mark.parametrize(
    "option",
    [
        {"max_qubits": 1},
        {"max_work": 3},
        {"max_memory_mb": 1e-9},
        {"max_branches": 1},
    ],
)
def test_source_resource_failures_precede_export(option, monkeypatch):
    c = Circuit(2).h(0).measure(0, 0)
    plan = plan_conversion(c, target="json")
    monkeypatch.setattr(
        ConversionPlan, "export", lambda *args: pytest.fail("Export preceded bounds")
    )
    with pytest.raises(ResourceError):
        plan.verify(**option)


@pytest.mark.parametrize(
    "option",
    [
        {"max_qubits": 0},
        {"max_work": True},
        {"max_memory_mb": 0},
        {"max_branches": 0},
        {"basis_chunk_size": 0},
        {"basis_chunk_size": 1.5},
        {"atol": -1},
        {"rtol": -1},
        {"atol": float("nan")},
        {"max_memory_mb": float("inf")},
    ],
)
def test_invalid_verification_settings(option):
    with pytest.raises(ValidationError):
        plan_conversion(Circuit(1), target="ir").verify(**option)


@pytest.mark.parametrize("bindings", [{}, {"x": 0.1, "other": 0.2}, {"x": [0.1]}, {"x": 1j}, [0.1]])
def test_verification_requires_complete_finite_scalar_bindings(bindings):
    with pytest.raises(ValidationError):
        plan_conversion(Circuit(1).ry(Parameter("x"), 0), target="json").verify(bindings)


def test_oversized_source_and_shared_expressions_fail_before_binding_or_export(monkeypatch):
    huge = plan_conversion(Circuit(1_000_000), target="ir")
    monkeypatch.setattr(ConversionPlan, "export", lambda *args: pytest.fail("Unexpected export"))
    with pytest.raises(ResourceError, match="max_work"):
        huge.verify(max_qubits=1_000_000)
    angle = Parameter("x")
    for _ in range(25):
        angle = angle.sin() * angle
    plan = plan_conversion(Circuit(1).ry(angle, 0), target="ir")
    with pytest.raises(ResourceError, match="expressions"):
        plan.verify({"x": 0.1}, max_work=10000)


def test_memory_budget_adapts_basis_chunk_size_without_skipping_entries(monkeypatch):
    import openqml.verification as verifier

    sizes, original = [], verifier._evolve_basis

    def track(circuit, basis, *args):
        sizes.append(len(basis))
        return original(circuit, basis, *args)

    monkeypatch.setattr(verifier, "_evolve_basis", track)
    c = Circuit(1).measure(0, 0).reset(0)
    check = plan_conversion(c, target="json").verify(max_memory_mb=0.003)
    assert check.matched and check.checked_inputs == 4 and check.checked_entries == 32
    assert sizes == [1] * 8


@pytest.mark.parametrize("wrong", [Circuit(2), Circuit(1).measure(0, 0)])
def test_structural_round_trip_failures_are_explicit(wrong, fault_adapter):
    with pytest.raises(ValidationError, match="register width"):
        plan_conversion(Circuit(1), target=fault_adapter(lambda c: wrong)).verify()


def test_incompatible_plans_are_not_verified_as_success():
    plan = plan_conversion(Circuit(1).with_global_phase(0.3), target="openqasm2")
    with pytest.raises(ConversionError, match="GLOBAL_PHASE_UNSUPPORTED"):
        plan.verify()


def test_reimport_options_are_forwarded_to_custom_adapters(fault_adapter):
    received = []

    def importer(value, **options):
        received.append(options)
        return value

    target = fault_adapter(lambda c: c, importer)
    assert plan_conversion(Circuit(1), target=target).verify(source_options={"custom": 7}).matched
    assert received == [{"custom": 7}]


def test_record_search_is_bounded_by_work_before_expanding_assignments(monkeypatch):
    c = Circuit(1)
    for bit in range(100):
        c = c.measure(0, bit)
    plan = plan_conversion(c, target="json")
    monkeypatch.setattr(ConversionPlan, "export", lambda *args: pytest.fail("Unexpected export"))
    with pytest.raises(ResourceError, match="max_work"):
        plan.verify(max_branches=1 << 100, max_work=10000)


def test_large_finite_memory_budget_is_accepted_without_float_overflow():
    check = plan_conversion(Circuit(1).h(0), target="json").verify(max_memory_mb=1e308)
    assert check.matched and check.checked_entries == 4
    check.raise_for_errors()


def test_reimported_parameters_and_new_effects_are_reported(fault_adapter):
    target = fault_adapter(lambda c: Circuit(1).ry(Parameter("unexpected"), 0))
    with pytest.raises(ValidationError, match="unbound parameters"):
        plan_conversion(Circuit(1), target=target).verify()
    unregister_adapter(target)
    target = fault_adapter(lambda c: Circuit(1).reset(0))
    with pytest.raises(ValidationError, match="introduced effects"):
        plan_conversion(Circuit(1), target=target).verify()
