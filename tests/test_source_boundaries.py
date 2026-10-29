"""Program text is inspected without execution or a whole-program equivalence claim."""

import ast
import json

import pytest

import openqml as oq
from openqml.cli import main


def codes(report):
    return {d.code for d in report.diagnostics}


def test_source_context_risks_have_locations_and_never_execute(tmp_path):
    marker = tmp_path / "executed"
    source = f"""from pathlib import Path
from qiskit_aer.noise import NoiseModel as NM
from qiskit.circuit import Gate as G
Path({str(marker)!r}).write_text("unsafe")
model = NM()
class Custom(G):
    pass
for index in range(3):
    if index:
        compile("x=1", "test", "exec")
service.run(backend="device", noise_model=model)
"""
    report = oq.inspect_python(source, filename="program.py")
    assert not marker.exists()
    assert report.ast_parseable and report.static_scan_complete and not report.compatible
    assert codes(report) >= {
        "PYTHON_PROGRAM_UNSUPPORTED",
        "PYTHON_EXTERNAL_DEPENDENCY",
        "PYTHON_NOISE_CONTEXT",
        "PYTHON_CUSTOM_INSTRUCTION",
        "PYTHON_DYNAMIC_CONTROL",
        "PYTHON_DYNAMIC_EXECUTION",
        "PYTHON_HARDWARE_CONTEXT",
        "PYTHON_OPAQUE_DEFINITION",
    }
    assert all(d.location.startswith("program.py:") for d in report.diagnostics[1:])
    data = json.loads(report.to_json())
    assert data["code_executed"] is False and data["whole_program_conversion_supported"] is False
    assert data["runtime_validity"] == "not_proven"
    with pytest.raises(oq.ConversionError):
        oq.convert(source, source="python", target="ir")
    assert not marker.exists()


@pytest.mark.parametrize("source", ["", "x=1", "return 1", "values=[x for x in range(3)]"])
def test_ast_parseability_never_certifies_runtime_validity_or_conversion(source):
    report = oq.inspect_python(source)
    assert report.ast_parseable and report.static_scan_complete and not report.compatible
    assert "PYTHON_PROGRAM_UNSUPPORTED" in codes(report)
    if "for" in source:
        assert "PYTHON_DYNAMIC_CONTROL" in codes(report)


@pytest.mark.parametrize(
    "setting,source",
    [
        ({"max_source_bytes": 2}, "aaaa"),
        ({"max_source_bytes": 3}, "éé"),
        ({"max_tokens": 2}, "x=1"),
        ({"max_nesting": 2}, "x=(((1)))"),
        ({"max_statement_tokens": 3}, "x=1+2+3"),
    ],
)
def test_lexical_preflight_rejects_before_ast_allocation(monkeypatch, setting, source):
    monkeypatch.setattr(
        ast, "parse", lambda *a, **k: pytest.fail("Allocated AST after preflight rejection")
    )
    report = oq.inspect_python(source, **setting)
    assert not report.static_scan_complete and report.ast_parseable is None
    assert codes(report) & {"PYTHON_SOURCE_LIMIT", "PYTHON_PARSE_LIMIT"}


def test_node_and_diagnostic_limits_are_explicit():
    report = oq.inspect_python("a=1\nb=2", max_nodes=2)
    assert report.ast_parseable and not report.static_scan_complete
    assert "PYTHON_AST_LIMIT" in codes(report)
    report = oq.inspect_python("import opaque\n" * 6, max_diagnostics=3)
    assert report.static_scan_complete and report.diagnostics_truncated
    assert len(report.diagnostics) == 3 and report.diagnostics[-1].code == "PYTHON_DIAGNOSTIC_LIMIT"
    assert "PYTHON_SYNTAX" in codes(oq.inspect_python("def broken(:"))


def test_text_subclasses_cannot_run_overridden_text_methods():
    class Text(str):
        def encode(self, *args, **kwargs):
            pytest.fail("Ran an input object's overridden encoder")

    assert oq.inspect_python(Text("x=1")).ast_parseable


def test_classical_target_keyword_is_not_misidentified_as_hardware():
    report = oq.inspect_python("loss(predictions, target=labels)")
    assert "PYTHON_HARDWARE_CONTEXT" not in codes(report)


@pytest.mark.parametrize("kwargs", [{"max_tokens": 0}, {"max_nodes": True}, {"max_diagnostics": 1}])
def test_source_inspection_options_are_validated(kwargs):
    with pytest.raises(oq.ValidationError):
        oq.inspect_python("x=1", **kwargs)


@pytest.mark.parametrize(
    "option", ["noise_model", "backend", "calibrations", "execution_context", "hardware_options"]
)
def test_builtin_conversion_refuses_external_context_instead_of_dropping_it(option):
    with pytest.raises(oq.ConversionError) as caught:
        oq.plan_conversion(oq.Circuit(1).h(0), target="json", source_options={option: object()})
    assert caught.value.diagnostics[0].code == "EXTERNAL_EXECUTION_CONTEXT_UNSUPPORTED"
    assert caught.value.diagnostics[0].location == f"source_options.{option}"
    circuit = oq.Circuit(1).bit_flip(0.2, 0)
    assert oq.Circuit.from_json(oq.convert(circuit, target="json").value) == circuit


def test_native_calibrations_are_rejected_and_plan_scope_is_explicit():
    qiskit = pytest.importorskip("qiskit")

    class Calibrated(qiskit.QuantumCircuit):
        @property
        def calibrations(self):
            return {"pulse": object()}

    with pytest.raises(oq.ConversionError) as caught:
        oq.plan_conversion(Calibrated(1), source="qiskit", target="ir")
    assert caught.value.diagnostics[0].code == "HARDWARE_CALIBRATION_UNSUPPORTED"
    data = oq.plan_conversion(oq.Circuit(1).h(0), target="qiskit").to_dict()
    assert data["semantic_scope"]["unit"] == "circuit"
    assert data["semantic_scope"]["external_backend_configuration"] == "not_preserved"
    assert data["semantic_scope"]["numerical_verification"] == "not_run_by_planning"


def test_custom_adapter_keeps_its_explicit_context_contract():
    name = "context_test_adapter"
    adapter = oq.FrameworkAdapter(
        name, lambda value, noise_model: oq.Circuit(1).bit_flip(noise_model, 0), lambda value: value
    )
    oq.register_adapter(adapter)
    try:
        result = oq.convert(object(), source=name, target="ir", source_options={"noise_model": 0.2})
        assert result.value.has_effects and result.value.operations[0].name == "bit_flip"
    finally:
        oq.unregister_adapter(name)


def test_inspection_cli_reports_source_without_running_it(tmp_path, capsys):
    source, marker = tmp_path / "program.py", tmp_path / "side-effect"
    source.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
    assert main(["inspect-python", str(source)]) == 0
    data = json.loads(capsys.readouterr().out)
    assert not data["whole_program_conversion_supported"] and not marker.exists()
