"""Preflight must expose target costs and phase loss before native export."""

import builtins
import json

import numpy as np
import pytest

from openqml import (
    Circuit,
    ClassicalCondition,
    ConversionError,
    Parameter,
    ValidationError,
    decompose,
    density_matrix,
    diagnose,
    plan_conversion,
    statevector,
)
from openqml.adapters.openqasm import from_openqasm, to_openqasm
from openqml.cli import main


def reject_sdks(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"qiskit", "pennylane", "cirq"}:
            raise AssertionError(f"Unexpected SDK import: {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


def test_prepared_resources_and_nested_expansion_failure_are_sdk_free(monkeypatch):
    reject_sdks(monkeypatch)
    circuit = Circuit(2).if_else(
        ClassicalCondition((0,), 0), Circuit(2).crx(0.4, 0, 1), Circuit(2).crx(0.7, 1, 0)
    )
    plan = plan_conversion(circuit, target="openqasm3")
    data = plan.to_dict()
    assert data["resources"]["gates"] == 2
    assert data["prepared_resources"]["gates"] == 12
    assert data["preparation_complete"] and data["required_extras"] == []
    assert plan.circuit is circuit
    limited = plan_conversion(circuit, target="openqasm3", max_operations=10)
    assert not limited.compatible and limited.to_dict()["prepared_resources"] is None
    error = next(d for d in limited.diagnostics if d.code == "DECOMPOSITION_LIMIT")
    assert error.location == "operations[0].blocks[1].operations[0]"
    with pytest.raises(ConversionError, match="DECOMPOSITION_LIMIT"):
        limited.export()
    with pytest.raises(ConversionError, match="DECOMPOSITION_LIMIT"):
        decompose(circuit, basis={"h", "rz", "cx"}, max_operations=10)


def test_single_qubit_matrix_synthesis_preserves_every_column_and_phase_without_sdks(monkeypatch):
    reject_sdks(monkeypatch)
    rng = np.random.default_rng(7)
    matrices = [
        np.eye(2, dtype=complex),
        np.diag([1j, -1j]),
        np.array([[0, -1j], [1j, 0]]),
        np.array([[0, 1j], [1j, 0]]),
    ]
    for theta in (0.0, 1e-16, 1e-12, np.pi - 1e-16, np.pi - 1e-12):
        cosine, sine = np.cos(theta / 2), np.sin(theta / 2)
        matrices.append(
            np.exp(0.31j)
            * np.array(
                [
                    [cosine, -np.exp(-0.47j) * sine],
                    [np.exp(0.62j) * sine, np.exp(0.15j) * cosine],
                ]
            )
        )
    for _ in range(12):
        matrix, _ = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
        matrices.append(matrix)
    for matrix in matrices:
        circuit = Circuit(2).h(0).cx(0, 1).unitary(matrix, 1).with_global_phase(0.23)
        plan = plan_conversion(circuit, target="openqasm3")
        assert plan.preparation_complete and not plan.required_extras
        assert all(op.name != "unitary" for _, op in plan.prepared_circuit.walk_operations())
        restored = from_openqasm(plan.export().value)
        for basis in (Circuit(2), Circuit(2).x(0), Circuit(2).x(1), Circuit(2).x(0).x(1)):
            np.testing.assert_allclose(
                statevector(basis.compose(restored)),
                statevector(basis.compose(circuit)),
                atol=2e-12,
            )
        lowered = decompose(Circuit(1).unitary(matrix, 0), basis={"p", "ry"})
        for basis in (Circuit(1), Circuit(1).x(0)):
            np.testing.assert_allclose(
                statevector(basis.compose(lowered)), matrix[:, len(basis.operations)], atol=2e-12
            )


def test_qasm2_preflight_rejects_phase_revealed_by_matrix_synthesis(monkeypatch):
    reject_sdks(monkeypatch)
    circuit = Circuit(1).unitary(1j * np.eye(2), 0)
    plan = plan_conversion(circuit, target="openqasm2")
    assert not plan.compatible
    assert any(d.code == "GLOBAL_PHASE_UNSUPPORTED" for d in plan.diagnostics)
    with pytest.raises(ConversionError, match="GLOBAL_PHASE_UNSUPPORTED"):
        to_openqasm(circuit, version=2)
    # Anti-diagonal matrices have free Euler angles. Their phase can be
    # preserved directly without introducing an unrepresentable phase field.
    for matrix in (np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]])):
        circuit = Circuit(1).unitary(matrix, 0)
        assert plan_conversion(circuit, target="openqasm2").compatible
        restored = from_openqasm(to_openqasm(circuit, version=2))
        for basis in (Circuit(1), Circuit(1).x(0)):
            np.testing.assert_allclose(
                statevector(basis.compose(restored)), matrix[:, len(basis.operations)], atol=1e-12
            )


def test_multi_qubit_synthesis_is_declared_instead_of_claiming_final_resources(monkeypatch):
    reject_sdks(monkeypatch)
    plan = plan_conversion(Circuit(2).unitary(np.eye(4), 0, 1), target="openqasm3")
    assert plan.compatible and not plan.preparation_complete
    assert plan.required_extras == ("qiskit",)
    assert plan.to_dict()["prepared_resources"] is None
    assert [d.code for d in plan.diagnostics] == ["MATRIX_SYNTHESIS_REQUIRED"]
    assert plan.diagnostics[0].location == "operations[0]"
    assert plan_conversion(Circuit(1).h(0), target="cirq").required_extras == ("cirq",)


def test_sdk_synthesis_result_is_rechecked_for_expansion_and_phase():
    pytest.importorskip("qiskit")
    matrix = np.diag([1, 1, 1, np.exp(0.37j)])
    circuit = Circuit(2).h(0).h(1).unitary(matrix, 1, 0)
    plan = plan_conversion(circuit, target="openqasm3")
    restored = from_openqasm(plan.export().value)
    np.testing.assert_allclose(statevector(restored), statevector(circuit), atol=2e-12)
    limited = plan_conversion(circuit, target="openqasm3", max_operations=4)
    assert limited.compatible and not limited.preparation_complete
    with pytest.raises(ConversionError, match="LIMIT"):
        limited.export()
    phase_plan = plan_conversion(
        Circuit(2).unitary(np.exp(0.3j) * np.eye(4), 0, 1), target="openqasm2"
    )
    assert phase_plan.compatible and not phase_plan.preparation_complete
    with pytest.raises(ConversionError, match="GLOBAL_PHASE_UNSUPPORTED"):
        phase_plan.export()


def test_qasm_statement_budget_predicts_repeated_false_branches_before_emission(monkeypatch):
    reject_sdks(monkeypatch)
    leaf = Circuit(1).ry(Parameter("theta"), 0).with_global_phase(0.2)
    nested = Circuit(1).if_else(ClassicalCondition((0, 1, 2), 5), Circuit(1).x(0), leaf)
    circuit = Circuit(1).if_else(ClassicalCondition((0, 1), 1), nested, nested)
    text = to_openqasm(circuit)
    lines = len(text.splitlines())
    assert lines > sum(1 for _ in circuit.walk_operations())
    assert plan_conversion(circuit, target="openqasm3", max_output_statements=lines).compatible
    limited = plan_conversion(circuit, target="openqasm3", max_output_statements=lines - 1)
    assert not limited.compatible
    assert any(d.code == "QASM_EXPORT_LIMIT" for d in limited.diagnostics)
    with pytest.raises(ConversionError, match="QASM_EXPORT_LIMIT"):
        to_openqasm(circuit, max_output_statements=lines - 1)
    np.testing.assert_allclose(
        density_matrix(from_openqasm(text), {"theta": 0.4}),
        density_matrix(circuit, {"theta": 0.4}),
    )


def test_guard_width_and_depth_are_diagnosed_at_the_original_operation():
    circuit = Circuit(1).if_else(ClassicalCondition(tuple(range(65)), 0), Circuit(1).x(0))
    plan = plan_conversion(circuit, target="openqasm3")
    diagnostic = next(d for d in plan.diagnostics if d.code == "QASM_EXPORT_LIMIT")
    assert diagnostic.location == "operations[0]"
    deep = Circuit(1).if_else(ClassicalCondition((0,), 0), Circuit(1).x(0))
    assert diagnose(deep, "cirq", max_depth=0)[0].code == "CONTROL_DEPTH_LIMIT"
    with pytest.raises(ConversionError, match="CONTROL_DEPTH_LIMIT"):
        plan_conversion(deep, target="openqasm3", max_depth=0)
    with pytest.raises(ConversionError, match="CONTROL_DEPTH_LIMIT"):
        decompose(deep, max_depth=0)


@pytest.mark.parametrize("option", ["max_operations", "max_depth", "max_output_statements"])
@pytest.mark.parametrize("value", [True, -1, 1.5])
def test_invalid_conversion_budgets(option, value):
    with pytest.raises(ValidationError):
        plan_conversion(Circuit(1), target="ir", **{option: value})


def test_cli_preflight_reports_target_expansion_with_configured_limits(tmp_path, capsys):
    source = tmp_path / "circuit.json"
    source.write_text(Circuit(2).crx(0.2, 0, 1).to_json())
    args = ["convert", str(source), "--from", "json", "--to", "openqasm3", "--max-operations", "3"]
    assert main(args + ["--check"]) == 2
    plan = json.loads(capsys.readouterr().out)
    assert not plan["compatible"] and plan["limits"]["max_operations"] == 3
    assert main(args) == 2
    assert "DECOMPOSITION_LIMIT" in capsys.readouterr().err


@pytest.mark.parametrize("framework", ["cirq", "pennylane"])
def test_native_exports_respect_explicit_operation_limits(framework):
    pytest.importorskip(framework)
    from openqml.frontends import get_adapter

    circuit = Circuit(2).h(0).cx(0, 1)
    native = plan_conversion(circuit, target=framework, max_operations=2).export().value
    restored = get_adapter(framework).importer(native)
    np.testing.assert_allclose(statevector(restored), statevector(circuit), atol=2e-12)
    with pytest.raises(ConversionError, match="DECOMPOSITION_LIMIT"):
        get_adapter(framework).exporter(circuit, max_operations=1)
