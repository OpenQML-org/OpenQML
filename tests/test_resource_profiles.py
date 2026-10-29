"""Resource profiles separate program size, representation storage and policy."""

import json
import math

import pytest

import openqml as oq
from openqml.cli import main


@pytest.mark.parametrize(
    "width,batch,cap,budget,allowed",
    [
        (20, 1, 20, 256, True),
        (22, 1, 22, 256, True),
        (23, 1, 23, 256, False),
        (5, 3, 20, 256, True),
        (22, 1, 20, 256, False),
    ],
)
def test_dense_payload_and_four_buffer_policy(width, batch, cap, budget, allowed):
    report = oq.estimate_resources(
        oq.Circuit(width), batch_size=batch, max_qubits=cap, max_memory_mb=budget
    )
    assert report.statevector_bytes == batch * 16 * 2**width
    assert report.density_bytes == batch * 16 * 2 ** (2 * width)
    assert report.statevector_budget_bytes == 4 * report.statevector_bytes
    assert report.within_statevector_policy is allowed
    if allowed:
        report.require_statevector()
    else:
        with pytest.raises(oq.ResourceError):
            report.require_statevector()


def test_large_exponents_are_reported_without_large_state_allocations(monkeypatch):
    monkeypatch.setattr(oq, "statevector", lambda *a, **k: pytest.fail("Allocated dense state"))
    report = oq.estimate_resources(oq.Circuit(1024))
    assert report.statevector_bytes is report.density_bytes is None
    assert report.statevector_log2_bytes == 1028 and report.density_log2_bytes == 2052
    assert not report.within_statevector_policy
    assert json.loads(report.to_json())["scope"] == "storage_and_policy_only"


def test_budget_rounding_never_accepts_the_next_state_width():
    budget = math.nextafter(256.0, 0.0)
    assert oq.ExecutionOptions.for_statevector(budget).max_qubits == 21
    report = oq.estimate_resources(oq.Circuit(22), max_qubits=22, max_memory_mb=budget)
    assert not report.within_statevector_policy


def test_ir_operation_limits_and_density_effects_are_independent_of_storage():
    circuit = oq.CircuitBuilder(1).h(0).x(0).z(0).build()
    with pytest.raises(oq.ConversionError) as caught:
        oq.estimate_resources(circuit, max_operations=2)
    assert caught.value.diagnostics[0].code == "DECOMPOSITION_LIMIT"
    report = oq.estimate_resources(oq.Circuit(1).bit_flip(0.1, 0))
    assert not report.within_statevector_policy
    assert report.diagnostics[0].code == "DENSITY_BRANCH_COST"


def test_resource_cli_reports_theory_instead_of_attempting_execution(tmp_path, capsys):
    source = tmp_path / "large.json"
    source.write_text(oq.Circuit(128).to_json())
    assert main(["resources", str(source), "--from", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["num_qubits"] == 128 and data["statevector_log2_bytes"] == 132
    assert not data["within_statevector_policy"]
