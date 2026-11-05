"""Native dynamic-switch references and bounded shared-definition compilation."""

import importlib.util
import io
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import openqml as oq


def native_density(native):
    """Keep the independent Aer oracle's OpenMP runtime in its own process."""
    if importlib.util.find_spec("qiskit_aer") is None:
        pytest.skip("Qiskit Aer is required for the independent density reference")
    from qiskit import qpy

    buffer = io.BytesIO()
    qpy.dump(native, buffer)
    script = """import io, json, sys
import numpy as np
from qiskit import qpy
from qiskit_aer import AerSimulator
circuit = qpy.load(io.BytesIO(sys.stdin.buffer.read()))[0]
circuit.save_density_matrix()
result = AerSimulator(method='density_matrix').run(circuit, shots=16, seed_simulator=7).result()
matrix = np.asarray(result.data(0)['density_matrix'])
print(json.dumps([matrix.real.tolist(), matrix.imag.tolist()]))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        input=buffer.getvalue(),
        capture_output=True,
        check=True,
        timeout=90,
        env=dict(os.environ, OMP_NUM_THREADS="1"),
    )
    real, imag = json.loads(completed.stdout)
    return np.array(real) + 1j * np.array(imag)


@pytest.mark.parametrize("selector", range(4))
def test_grouped_switch_default_and_selector_write_match_native_density(selector):
    qiskit = pytest.importorskip("qiskit")
    from qiskit.circuit import CASE_DEFAULT

    from openqml.adapters._common import reverse_local_basis

    native = qiskit.QuantumCircuit(3, 2)
    for bit in range(2):
        if selector & (1 << bit):
            native.x(bit)
        native.measure(bit, bit)
    with native.switch(native.cregs[0]) as case:
        with case(0):
            native.x(2)
            native.measure(2, 0)
        with case(1, 2):
            native.y(2)
        with case(CASE_DEFAULT):
            native.h(2)
    plan = oq.plan_conversion(native, source="qiskit", target="openqasm3")
    assert "SWITCH_CASE_LOWERED" in {d.code for d in plan.diagnostics}
    assert plan.verify().matched
    expected = reverse_local_basis(native_density(native))
    np.testing.assert_allclose(oq.density_matrix(plan.circuit), expected, atol=2e-14)


def test_switch_symbolic_phases_bit_targets_and_empty_cases():
    qiskit = pytest.importorskip("qiskit")
    from qiskit.circuit import CASE_DEFAULT, Parameter

    theta = Parameter("theta")
    native = qiskit.QuantumCircuit(1, 1)
    with native.switch(native.clbits[0]) as case:
        with case(0):
            native.ry(theta.sin(), 0)
        with case(CASE_DEFAULT):
            native.rz(theta, 0)
    plan = oq.plan_conversion(native, source="qiskit", target="qiskit")
    assert plan.circuit.parameters == ("theta",)
    assert plan.verify({"theta": 0.3}).matched
    assert plan.circuit.has_effects


def test_switch_rejects_depth_expansion_and_expression_targets():
    qiskit = pytest.importorskip("qiskit")
    from qiskit.circuit.classical import expr

    native = qiskit.QuantumCircuit(1, 2)
    with native.switch(native.cregs[0]) as case:
        with case(0, 1, 2):
            native.x(0)
    for settings in ({"max_depth": 2}, {"max_operations": 3}):
        with pytest.raises(oq.ConversionError) as exc:
            oq.plan_conversion(native, source="qiskit", target="ir", **settings)
        assert any(d.code.endswith("_LIMIT") for d in exc.value.diagnostics)
    other = qiskit.QuantumCircuit(1, 2)
    with other.switch(expr.bit_and(other.cregs[0], 1)) as case:
        with case(0):
            other.x(0)
    with pytest.raises(oq.ConversionError) as exc:
        oq.convert(other, source="qiskit", target="ir")
    assert exc.value.diagnostics[0].code == "CLASSICAL_SWITCH_TARGET"


def test_repeated_definition_is_imported_once_with_wire_mapping_and_fresh_calls(monkeypatch):
    qiskit = pytest.importorskip("qiskit")
    from qiskit.circuit import Parameter
    from qiskit.quantum_info import Operator

    from openqml.adapters import qiskit as adapter

    theta = Parameter("theta")
    body = qiskit.QuantumCircuit(1)
    body.global_phase = theta / 3
    body.ry(theta.sin() + theta, 0)
    gate = body.to_gate()
    native = qiskit.QuantumCircuit(2)
    for i in range(12):
        native.append(gate, [i % 2], copy=False)
    calls, original = [], adapter.sympy_expression

    def tracked(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(adapter, "sympy_expression", tracked)
    imported = adapter.from_qiskit(native)
    assert len(calls) <= 3
    converted = adapter.to_qiskit(imported)
    np.testing.assert_allclose(
        Operator(converted.assign_parameters({converted.parameters[0]: 0.2})).data,
        Operator(native.assign_parameters({theta: 0.2})).data,
        atol=2e-14,
    )
    assert len(imported.operations) == 12
    with pytest.raises(oq.ConversionError, match="max_operations|budget"):
        adapter.from_qiskit(native, max_operations=11)
    native.x(0)
    assert len(adapter.from_qiskit(native).operations) == 13


def test_export_shared_expression_graph_keeps_symbolic_semantics():
    pytest.importorskip("qiskit")
    theta = oq.Parameter("theta")
    value = theta.sin() + theta.cos()
    for _ in range(8):
        value = value * value + theta
    circuit = oq.Circuit(1).ry(value, 0).rz(value, 0)
    plan = oq.plan_conversion(circuit, target="qiskit")
    assert plan.verify({"theta": 0.01}, mode="native").matched
