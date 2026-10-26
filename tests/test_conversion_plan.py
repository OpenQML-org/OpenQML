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
    convert,
    diagnose,
    plan_conversion,
    statevector,
)
from openqml.cli import main


def test_planning_never_imports_target_frameworks(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"qiskit", "pennylane", "cirq"}:
            raise AssertionError(f"Planning imported {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    circuit = Circuit(2).h(0).cx(0, 1)
    for target in ("qiskit", "pennylane", "cirq", "openqasm2", "openqasm3", "json", "ir"):
        plan = plan_conversion(circuit, target=target)
        assert plan.compatible
        payload = json.loads(plan.to_json())
        assert payload["schema"] == "openqml.conversion_plan"
        assert payload["resources"]["num_qubits"] == 2


def test_plan_collects_incompatibilities_and_export_refuses_them():
    circuit = (
        Circuit(1)
        .ry(Parameter("theta"), 0)
        .amplitude_damping(0.2, 0)
        .measure(0, 0)
        .if_else(ClassicalCondition((0,), 1), Circuit(1).x(0))
    )
    plan = plan_conversion(circuit, target="openqasm2")
    assert not plan.compatible
    assert {d.code for d in plan.diagnostics} >= {
        "UNBOUND_PARAMETER",
        "CHANNEL_UNSUPPORTED",
        "CONTROL_FLOW_UNSUPPORTED",
    }
    with pytest.raises(ConversionError) as caught:
        plan.export()
    assert caught.value.diagnostics == plan.diagnostics


def test_plan_exports_prepared_identifiers_and_preserves_original_ownership():
    circuit = Circuit(1).ry(Parameter("input"), 0).with_global_phase(0.31)
    plan = plan_conversion(circuit, target="qasm3")
    assert plan.compatible
    assert plan.parameter_map == {"input": "parameter_0"}
    assert plan.circuit is circuit
    result = plan.export()
    assert convert(circuit, target="qasm3") == result
    restored = convert(result.value, source="openqasm3", target="ir").value
    np.testing.assert_allclose(
        statevector(restored, {"parameter_0": 0.41}), statevector(circuit, {"input": 0.41})
    )


def test_nested_decomposition_notes_have_exact_ir_locations():
    circuit = Circuit(2).if_else(ClassicalCondition((0,), 0), Circuit(2).crx(0.4, 0, 1))
    plan = plan_conversion(circuit, target="openqasm3")
    notes = [d for d in plan.diagnostics if d.code == "EXACT_GATE_DECOMPOSITION"]
    assert len(notes) == 1
    assert notes[0].location == "operations[0].blocks[0].operations[0]"
    assert plan.export().value.startswith("OPENQASM 3.0;")


def test_plan_binding_optimization_and_source_failures():
    circuit = Circuit(1).ry(Parameter("theta"), 0).x(0).x(0)
    plan = plan_conversion(circuit, target="openqasm2", bindings={"theta": 0.27}, optimize=True)
    assert plan.compatible and len(plan.circuit.operations) == 1
    assert plan.optimization is not None
    assert "0.27" in plan.export().value
    with pytest.raises(ConversionError):
        plan_conversion("invalid qasm", source="openqasm3", target="json")
    with pytest.raises(ValidationError):
        diagnose("not a circuit", "json")


def test_cli_preflight_json_exit_codes_and_no_output_file(tmp_path, capsys):
    source = tmp_path / "circuit.json"
    source.write_text(Circuit(1).ry(Parameter("theta"), 0).to_json())
    args = ["convert", str(source), "--from", "json", "--to", "openqasm2", "--check"]
    assert main(args) == 2
    assert not json.loads(capsys.readouterr().out)["compatible"]
    args[args.index("openqasm2")] = "openqasm3"
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["compatible"]
    output = tmp_path / "output.qasm"
    assert main(args + ["-o", str(output)]) == 2
    assert not output.exists()
