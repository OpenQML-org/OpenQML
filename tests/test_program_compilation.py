import numpy as np
import pytest

import openqml as oq


def test_native_export_preserves_symbolic_and_decomposition_phases():
    qiskit = pytest.importorskip("qiskit")
    from qiskit.circuit import Parameter
    from qiskit.quantum_info import Operator

    theta = Parameter("theta")
    sub = qiskit.QuantumCircuit(2, name="custom")
    sub.global_phase = theta**2 + 0.37
    sub.sx(0)
    sub.rxx(theta, 0, 1)
    native = qiskit.QuantumCircuit(3)
    native.global_phase = 0.19
    native.h(2)
    native.append(sub.to_gate(), [2, 0])
    native.cry(theta, 0, 1)
    result = oq.compile_program(native, target="openqasm3")
    assert result.route == "native_openqasm3" and "gphase" in result.value
    roundtrip = oq.compile_program(result.value, source="openqasm3", target="qiskit").value
    for value in (-0.53, 0.0, 0.81):
        left = native.assign_parameters({p: value for p in native.parameters})
        right = roundtrip.assign_parameters({p: value for p in roundtrip.parameters})
        np.testing.assert_allclose(Operator(left).data, Operator(right).data, atol=1e-12)


def test_dynamic_program_retains_loop_and_phase_scopes():
    qiskit = pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")
    from qiskit import qasm3, transpile
    from qiskit_aer import AerSimulator

    program = qiskit.QuantumCircuit(2, 2)
    program.x(0)
    program.measure(0, 0)
    body = qiskit.QuantumCircuit(1, 1)
    body.global_phase = 0.23
    body.x(0)
    body.measure(0, 0)
    program.while_loop((program.clbits[0], True), body, [0], [0])
    body = qiskit.QuantumCircuit(1)
    body.sx(0)
    program.for_loop(range(2), None, body, [1], [])
    program.measure(1, 1)
    captured = oq.QuantumProgram.from_callable(lambda: program)
    result = captured.compile("openqasm3")
    assert {"while_loop", "for_loop", "scoped_global_phase"} <= set(result.features)
    assert "while (" in result.value and "for int" in result.value
    roundtrip = oq.compile_program(result.value, source="openqasm3", target="qiskit").value
    simulator = AerSimulator()
    for circuit in (program, roundtrip):
        run = simulator.run(transpile(circuit, simulator), shots=64, seed_simulator=4).result()
        assert run.get_counts() == {"10": 64}
    loop = next(
        x.operation for x in qasm3.loads(result.value).data if x.operation.name == "while_loop"
    )
    assert float(loop.blocks[0].global_phase) == pytest.approx(0.23)


def test_native_program_keeps_large_loops_structured():
    qiskit = pytest.importorskip("qiskit")
    body = qiskit.QuantumCircuit(1)
    body.rx(0.1, 0)
    program = qiskit.QuantumCircuit(1)
    program.for_loop(range(10**9), None, body, [0], [])
    result = oq.compile_program(program, target="openqasm3", max_operations=100)
    assert result.static_work < 100 and len(result.value) < 1000
    assert "999999999" in result.value
    assert result.semantic_scope["runtime_loop_iterations"].startswith("not_bounded")
    with pytest.raises(oq.ConversionError):
        oq.compile_program(program, target="openqasm3", strategy="ir", max_operations=100)


def test_native_failures_are_explicit_and_source_unchanged():
    qiskit = pytest.importorskip("qiskit")
    from qiskit.circuit import Gate

    program = qiskit.QuantumCircuit(1)
    program.h(0)
    original = program.copy()
    with pytest.raises(oq.ConversionError):
        oq.compile_program(program, target="openqasm3", max_output_bytes=10)
    assert program == original
    program.append(Gate("opaque", 1, []), [0])
    with pytest.raises(oq.ConversionError, match="OPAQUE"):
        oq.compile_program(program, target="openqasm3")
    with pytest.raises(oq.ConversionError, match="NATIVE_ROUTE"):
        oq.compile_program(original, source="qiskit", target="cirq", strategy="native")
    captured = oq.QuantumProgram(original, "qiskit", (("max_operations", 1),))
    with pytest.raises(oq.ConversionError, match="WORK_LIMIT"):
        captured.compile("openqasm3")
    with pytest.raises(oq.ConversionError, match="SOURCE_OPTIONS"):
        oq.QuantumProgram(original, "qiskit", (("backend", "ignored"),)).compile("openqasm3")


def test_native_subclass_can_be_captured():
    qiskit = pytest.importorskip("qiskit")

    class CustomCircuit(qiskit.QuantumCircuit):
        pass

    def factory():
        circuit = CustomCircuit(1)
        circuit.h(0)
        return circuit

    assert oq.QuantumProgram.from_callable(factory).compile("openqasm3").route == "native_openqasm3"


def test_core_ir_program_compilation_without_optional_sdk():
    circuit = oq.CircuitBuilder(2).h(0).cx(0, 1).build()
    result = oq.compile_program(circuit, target="json")
    assert result.route == "circuit_ir"
    assert oq.Circuit.from_json(result.value) == circuit


def test_custom_gate_with_a_builtin_name_keeps_its_own_semantics():
    qiskit = pytest.importorskip("qiskit")
    from qiskit import qasm3
    from qiskit.circuit import Gate
    from qiskit.quantum_info import Operator

    definition = qiskit.QuantumCircuit(2)
    definition.x(1)
    custom = Gate("cx", 2, [])
    custom.definition = definition
    circuit = qiskit.QuantumCircuit(2)
    circuit.append(custom, [0, 1])
    result = oq.compile_program(circuit, target="openqasm3")
    np.testing.assert_allclose(
        Operator(qasm3.loads(result.value)).data, Operator(circuit).data, atol=1e-12
    )


def test_nonlinear_input_lifting_and_gate_scope_shadowing():
    qiskit = pytest.importorskip("qiskit")
    from qiskit.quantum_info import Operator

    source = """OPENQASM 3.0;
include "stdgates.inc";
input float[64] theta;
qubit q;
gphase(sin(theta)**2);
ry(sqrt(theta*theta+0.25)) q;
"""
    result = oq.compile_program(source, source="openqasm3", target="qiskit")
    assert "symbolic_angle_lifting" in result.features
    for value in (-0.8, 0.2):
        native = result.value.assign_parameters({p: value for p in result.value.parameters})
        expected = qiskit.QuantumCircuit(1, global_phase=np.sin(value) ** 2)
        expected.ry(np.sqrt(value**2 + 0.25), 0)
        np.testing.assert_allclose(Operator(native).data, Operator(expected).data, atol=1e-12)
    shadow = """OPENQASM 3.0;
include "stdgates.inc";
input float[64] theta;
gate local(theta) q { ry(2*theta) q; }
qubit q;
local(theta+1) q;
"""
    result = oq.compile_program(shadow, source="openqasm3", target="qiskit")
    native = result.value.assign_parameters({p: 0.2 for p in result.value.parameters})
    expected = qiskit.QuantumCircuit(1)
    expected.ry(2 * 1.2, 0)
    np.testing.assert_allclose(Operator(native).data, Operator(expected).data, atol=1e-12)
