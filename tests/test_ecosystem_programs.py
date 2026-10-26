import numpy as np
import pytest

from openqml import (
    Circuit,
    Parameter,
    PauliString,
    QuantumProgram,
    as_circuit,
    convert,
    expectation,
    parameter_shift,
    statevector,
)


def test_nonlinear_expression_roundtrip_derivative_and_batch():
    x, w = Parameter("inputs[0]"), Parameter("θ")
    angle = (x.sin() * w + x**2 / (1 + w**2)).exp().log()
    c = Circuit(1).ry(angle, 0)
    restored = Circuit.from_json(c.to_json())
    assert restored == c
    bindings = {"inputs[0]": 0.37, "θ": -0.61}
    g = parameter_shift(c, PauliString("Z"), bindings)
    for name in bindings:
        plus, minus = dict(bindings), dict(bindings)
        plus[name] += 1e-6
        minus[name] -= 1e-6
        finite = (
            expectation(c, PauliString("Z"), plus) - expectation(c, PauliString("Z"), minus)
        ) / 2e-6
        assert g[name] == pytest.approx(finite, abs=1e-8)
    values = statevector(c, {"inputs[0]": np.array([0.1, 0.2, 0.3]), "θ": -0.61})
    assert values.shape == (3, 2)
    for i, xvalue in enumerate([0.1, 0.2, 0.3]):
        np.testing.assert_allclose(
            values[i], statevector(c, {"inputs[0]": xvalue, "θ": -0.61}), atol=1e-12
        )


@pytest.mark.parametrize("framework", ["qiskit", "pennylane", "cirq"])
def test_arbitrary_complex_unitary_cross_framework(framework):
    pytest.importorskip(framework)
    rng = np.random.default_rng(19)
    matrix, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    c = Circuit(3).h(0).ry(0.43, 1).x(2).unitary(matrix, 2, 0).with_global_phase(0.27)
    assert Circuit.from_json(c.to_json()) == c
    result = convert(c, source="ir", target=framework)
    restored = convert(result.value, source=framework, target="ir").value
    np.testing.assert_allclose(statevector(restored), statevector(c), atol=2e-11)
    from openqml import execute

    np.testing.assert_allclose(
        execute(c, backend=framework).statevector, statevector(c), atol=2e-11
    )
    np.testing.assert_allclose(statevector(c.compose(c.inverse())), np.eye(8)[0], atol=2e-11)


def test_qiskit_python_factory_custom_gate_paramvector_and_nonlinearity():
    qiskit = pytest.importorskip("qiskit")
    from qiskit.circuit import ParameterVector
    from qiskit.quantum_info import Statevector

    x = ParameterVector("inputs", 2)
    w = ParameterVector("weights", 2)

    def factory():
        block = qiskit.QuantumCircuit(2)
        block.ry(x[0].sin() * w[0] + x[1] ** 2, 0)
        block.rzx(w[1].cos(), 0, 1)
        c = qiskit.QuantumCircuit(3)
        for targets in [(2, 0), (0, 1)]:
            c.append(block.to_gate(label="library_block"), targets)
        return c

    p = QuantumProgram.from_callable(factory)
    ir = p.to_ir()
    bindings = {x[0].name: 0.2, x[1].name: -0.43, w[0].name: 0.61, w[1].name: -0.27}
    bound = p.bind(bindings).native
    expected = (
        np.asarray(Statevector.from_instruction(bound).data)
        .reshape(2, 2, 2)
        .transpose(2, 1, 0)
        .reshape(-1)
    )
    np.testing.assert_allclose(statevector(ir, bindings), expected, atol=2e-12)
    exported = convert(ir, source="ir", target="openqasm3")
    assert exported.parameter_map and any(
        d.code == "PARAMETER_RENAMED" for d in exported.diagnostics
    )
    back = convert(exported.value, source="openqasm3", target="ir").value
    aliases = {exported.parameter_map.get(k, k): v for k, v in bindings.items()}
    np.testing.assert_allclose(statevector(back, aliases), expected, atol=2e-12)


@pytest.mark.parametrize("version", [2, 3])
def test_official_qasm_frontend_custom_gate_definitions(version):
    pytest.importorskip("qiskit")
    if version == 3:
        pytest.importorskip("qiskit_qasm3_import")
    include = "qelib1.inc" if version == 2 else "stdgates.inc"
    declaration = "qreg q[2];" if version == 2 else "qubit[2] q;"
    source = (
        f'OPENQASM {version}.0; include "{include}"; '
        f"gate learned(t) a,b {{ ry(t) a; cx a,b; }} {declaration} learned(0.43) q[0],q[1];"
    )
    p = QuantumProgram.from_qasm(source)
    expected = statevector(Circuit(2).ry(0.43, 0).cx(0, 1))
    np.testing.assert_allclose(statevector(p.to_ir()), expected, atol=1e-12)
    np.testing.assert_allclose(
        statevector(convert(source, source=f"openqasm{version}", target="ir").value),
        expected,
        atol=1e-12,
    )


def test_pennylane_templates_expand_and_qasm_synthesizes_matrix():
    qml = pytest.importorskip("pennylane")
    pytest.importorskip("qiskit")
    tape = qml.tape.QuantumScript(
        [
            qml.AngleEmbedding([0.23, -0.41], wires=[0, 1]),
            qml.StronglyEntanglingLayers(np.ones((2, 2, 3)) * 0.17, wires=[0, 1]),
        ],
        [],
    )
    ir = as_circuit(tape)
    device = qml.device("default.qubit", wires=2)
    expected = qml.execute((qml.tape.QuantumScript(tape.operations, [qml.state()]),), device)[0]
    np.testing.assert_allclose(statevector(ir), expected, atol=1e-12)
    matrix, _ = np.linalg.qr(np.random.default_rng(9).normal(size=(4, 4)) + 1j * np.eye(4))
    c = Circuit(2).h(0).unitary(matrix, 1, 0)
    qasm = convert(c, source="ir", target="openqasm3").value
    restored = convert(qasm, source="openqasm3", target="ir").value
    np.testing.assert_allclose(statevector(c), statevector(restored), atol=1e-10)
