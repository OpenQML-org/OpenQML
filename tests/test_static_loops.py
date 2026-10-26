"""Finite SDK loops preserve phase, differentiability and resource contracts."""

import numpy as np
import pytest

from openqml import (
    Circuit,
    ConversionError,
    Parameter,
    QuantumProgram,
    convert,
    density_matrix,
    sample,
    statevector,
)


def test_parameterized_for_loop_preserves_reversed_wire_map_and_full_phase():
    sdk = pytest.importorskip("qiskit")
    from qiskit.circuit import Parameter as QParameter

    i, w = QParameter("iteration"), QParameter("w")
    body = sdk.QuantumCircuit(2)
    body.ry(w.sin() * i + i * i / 5, 1)
    body.cx(1, 0)
    body.global_phase = w.cos() * (i + 1) / 17
    native = sdk.QuantumCircuit(3)
    native.for_loop((-2, 0, 3), i, body, (2, 0), ())
    result = convert(native, target="ir")
    assert result.circuit.parameters == ("w",)
    assert [d.code for d in result.diagnostics] == ["STATIC_LOOP_UNROLLED"]
    t = Parameter("w")
    expected, phase = Circuit(3), t * 0
    for value in (-2, 0, 3):
        expected = expected.ry(t.sin() * value + value * value / 5, 0).cx(0, 2)
        phase += t.cos() * (value + 1) / 17
    expected = expected.with_global_phase(phase)
    for value in (-0.31, 0.47):
        np.testing.assert_allclose(
            statevector(result.value, {"w": value}), statevector(expected, {"w": value}), atol=2e-13
        )


def test_nested_static_loop_binds_each_lexical_index():
    sdk = pytest.importorskip("qiskit")
    native = sdk.QuantumCircuit(2)
    with native.for_loop(range(3)) as i:
        with native.for_loop((-1, 2)) as j:
            native.ry(i * 0.13 + j * 0.21, 1)
            native.cx(1, 0)
    result = convert(native, target="ir")
    expected = Circuit(2)
    for i in range(3):
        for j in (-1, 2):
            expected = expected.ry(i * 0.13 + j * 0.21, 1).cx(1, 0)
    np.testing.assert_allclose(statevector(result.value), statevector(expected), atol=1e-14)
    assert len(result.value.operations) == 12
    assert sum(d.code == "STATIC_LOOP_UNROLLED" for d in result.diagnostics) == 2


def test_static_loop_feedback_and_repeated_writes_match_native_unrolled_reference():
    sdk = pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")
    body = sdk.QuantumCircuit(2, 2)
    body.x(1)
    body.measure(1, 1)
    with body.if_test((body.clbits[0], 1)):
        body.x(0)
    native = sdk.QuantumCircuit(3, 4)
    native.h(2)
    native.measure(2, 3)
    native.for_loop(range(3), None, body, (2, 0), (3, 1))
    program = QuantumProgram.capture(native)
    ir = program.to_ir()
    expected = np.zeros((8, 8), complex)
    expected[4, 4] = 1
    np.testing.assert_allclose(density_matrix(ir), expected, atol=1e-14)
    local = sample(ir, 5000, seed=7)
    reference = sdk.QuantumCircuit(3, 4)
    reference.h(2)
    reference.measure(2, 3)
    for _ in range(3):
        reference.x(0)
        reference.measure(0, 1)
        with reference.if_test((reference.clbits[3], 1)):
            reference.x(2)
    native_counts = {
        k.replace(" ", "")[::-1]: v
        for k, v in QuantumProgram.capture(reference).run(shots=5000, seed=7).get_counts().items()
    }
    assert set(local) == set(native_counts) == {"0100", "0101"}
    for key in local:
        assert local[key] / 5000 == pytest.approx(native_counts[key] / 5000, abs=0.035)


@pytest.mark.parametrize("target", ["qiskit", "pennylane", "cirq", "openqasm2", "openqasm3"])
def test_static_loop_unrolls_to_other_ecosystems(target):
    sdk = pytest.importorskip("qiskit")
    if target in {"pennylane", "cirq"}:
        pytest.importorskip(target)
    native = sdk.QuantumCircuit(2)
    with native.for_loop(range(4)) as i:
        native.ry(i * 0.2, 1)
        native.cx(1, 0)
    result = QuantumProgram.capture(native).convert(target)
    assert result.source_format == "qiskit"
    assert "STATIC_LOOP_UNROLLED" in {d.code for d in result.diagnostics}
    restored = convert(result.value, source=target, target="ir").value
    expected = Circuit(2)
    for i in range(4):
        expected = expected.ry(i * 0.2, 1).cx(1, 0)
    np.testing.assert_allclose(statevector(restored), statevector(expected), atol=2e-13)


@pytest.mark.parametrize("parser", ["auto", "qiskit"])
def test_qasm3_static_loop_reports_frontend_and_expansion(parser):
    pytest.importorskip("qiskit_qasm3_import")
    source = (
        'OPENQASM 3.0; include "stdgates.inc"; input float w; qubit[2] q; '
        "for int i in [0:3] { ry(w*0.2*i) q[1]; cx q[1],q[0]; }"
    )
    result = convert(source, source="openqasm3", target="ir", source_options={"parser": parser})
    assert {d.code for d in result.diagnostics} == {"SDK_FRONTEND", "STATIC_LOOP_UNROLLED"}
    expected = Circuit(2)
    for i in range(4):
        expected = expected.ry(0.3 * 0.2 * i, 1).cx(1, 0)
    np.testing.assert_allclose(
        statevector(result.value, {"w": 0.3}), statevector(expected), atol=1e-14
    )


def test_empty_and_zero_wire_phase_only_loop_bodies_preserve_parent_interface():
    sdk = pytest.importorskip("qiskit")
    from qiskit.circuit import Parameter as QParameter

    index, w = QParameter("index"), QParameter("w")
    body = sdk.QuantumCircuit(global_phase=w * index)
    native = sdk.QuantumCircuit(3)
    native.for_loop((-1, 2), index, body, (), ())
    ir = QuantumProgram.capture(native).to_ir()
    assert ir.num_qubits == 3 and ir.operations == () and ir.parameters == ("w",)
    np.testing.assert_allclose(statevector(ir, {"w": 0.4}), np.eye(8)[0] * np.exp(0.4j), atol=1e-14)
    empty = sdk.QuantumCircuit(2)
    empty.for_loop((), index, sdk.QuantumCircuit(1), (1,), ())
    assert QuantumProgram.capture(empty).to_ir() == Circuit(2)


def test_nested_phase_only_work_and_huge_indexsets_are_budgeted_before_expansion(monkeypatch):
    sdk = pytest.importorskip("qiskit")
    native = sdk.QuantumCircuit(1)
    with native.for_loop(range(40)):
        with native.for_loop(range(40)):
            pass
    with pytest.raises(ConversionError, match="EXPANSION_LIMIT"):
        convert(native, target="ir", source_options={"max_operations": 1000})
    assert convert(native, target="ir", source_options={"max_operations": 1700}).value == Circuit(1)
    body = sdk.QuantumCircuit(1)
    body.unitary(np.eye(2), [0])
    huge = sdk.QuantumCircuit(1)
    huge.for_loop(range(10**12), None, body, (0,), ())

    def forbidden(*args, **kwargs):
        raise AssertionError("matrix allocated before loop budget validation")

    monkeypatch.setattr(sdk.circuit.library.UnitaryGate, "to_matrix", forbidden)
    with pytest.raises(ConversionError, match="EXPANSION_LIMIT"):
        convert(huge, target="ir", source_options={"max_operations": 100})


def test_loop_budget_depth_and_import_policy_survive_program_binding_and_conversion():
    pytest.importorskip("qiskit_qasm3_import")
    source = (
        'OPENQASM 3.0; include "stdgates.inc"; input float w; qubit q; '
        "for int i in [0:7] { ry(w*i) q; }"
    )
    program = QuantumProgram.from_qasm(source, max_operations=3)
    for action in (
        lambda: program.to_ir(),
        lambda: program.bind({"w": 0.2}).convert("ir"),
        lambda: convert(program, target="ir"),
    ):
        with pytest.raises(ConversionError, match="EXPANSION_LIMIT"):
            action()
    program = QuantumProgram.from_qasm(source, max_depth=0)
    with pytest.raises(ConversionError, match="EXPANSION_LIMIT"):
        program.to_ir()


@pytest.mark.parametrize("feature", ["break", "continue"])
def test_static_loops_do_not_miscompile_break_or_continue(feature):
    sdk = pytest.importorskip("qiskit")
    native = sdk.QuantumCircuit(1)
    with native.for_loop(range(3)):
        native.h(0)
        getattr(native, feature + "_loop")()
    with pytest.raises(ConversionError, match="CLASSICAL_CONTROL"):
        QuantumProgram.capture(native).to_ir()


def test_expansion_disabled_requires_native_loop_execution():
    sdk = pytest.importorskip("qiskit")
    body = sdk.QuantumCircuit(1)
    body.h(0)
    native = sdk.QuantumCircuit(1)
    native.for_loop(range(3), None, body, (0,), ())
    with pytest.raises(ConversionError, match="CLASSICAL_CONTROL"):
        convert(native, target="ir", source_options={"expand": False})


def test_loop_parameters_remain_trainable_in_torch():
    sdk = pytest.importorskip("qiskit")
    torch = pytest.importorskip("torch")
    from qiskit.circuit import Parameter as QParameter

    from openqml.interfaces.torch import QuantumLayer

    x, w = QParameter("x"), QParameter("w")
    native = sdk.QuantumCircuit(1)
    native.ry(x, 0)
    with native.for_loop(range(3)) as i:
        native.ry(w * (i + 1), 0)
    layer = QuantumLayer.from_program(native, (x,), (w,), dtype=torch.float64)
    inputs = torch.tensor([[0.3], [-0.4]], dtype=torch.float64, requires_grad=True)
    weights = torch.tensor([0.17], dtype=torch.float64, requires_grad=True)

    def function(x, w):
        return torch.func.functional_call(layer, {"weights": w}, (x,))

    assert torch.autograd.gradcheck(function, (inputs, weights))
    assert torch.autograd.gradgradcheck(function, (inputs, weights))
    torch.testing.assert_close(function(inputs, weights), torch.cos(inputs + 6 * weights))


def test_undefined_expression_at_a_loop_index_has_a_structured_diagnostic():
    sdk = pytest.importorskip("qiskit")
    from qiskit.circuit import Parameter as QParameter

    i = QParameter("index")
    body = sdk.QuantumCircuit(1)
    body.ry(1 / (i - 1), 0)
    native = sdk.QuantumCircuit(1)
    native.for_loop((0, 1), i, body, (0,), ())
    with pytest.raises(ConversionError, match="STATIC_LOOP_BINDING"):
        convert(native, target="ir")


def test_phase_only_unrolling_builds_a_balanced_expression_graph():
    sdk = pytest.importorskip("qiskit")
    from qiskit.circuit import Parameter as QParameter

    index, w = QParameter("index"), QParameter("w")
    body = sdk.QuantumCircuit(global_phase=(w * index).sin())
    native = sdk.QuantumCircuit(1)
    native.for_loop(range(128), index, body, (), ())
    ir = QuantumProgram.capture(native).to_ir()
    expected = np.exp(1j * np.sin(0.17 * np.arange(128)).sum())
    np.testing.assert_allclose(statevector(ir, {"w": 0.17}), [expected, 0], atol=2e-13)
    assert Circuit.from_json(ir.to_json()) == ir
