import numpy as np
import pytest

from openqml import (
    Circuit,
    ConversionError,
    Parameter,
    PauliString,
    convert,
    execute,
    expectation,
    parameter_shift,
    statevector,
)
from openqml.adapters.openqasm import from_openqasm, to_openqasm
from openqml.ir import GATES

NEW_GATES = ("u", "sxdg", "cp", "crx", "cry", "crz", "ch", "cswap", "iswap")


@pytest.mark.parametrize("framework", ["qiskit", "pennylane", "cirq"])
@pytest.mark.parametrize("gate", NEW_GATES)
def test_new_gate_native_execution_and_roundtrip(framework, gate):
    pytest.importorskip(framework)
    arity, count = GATES[gate]
    circuit = (
        Circuit(3)
        .h(0)
        .ry(0.63, 1)
        .rx(-0.37, 2)
        .cx(0, 2)
        .gate(gate, *(2, 0, 1)[:arity], params=(0.51, -0.24, 0.83)[:count])
        .with_global_phase(0.27)
    )
    np.testing.assert_allclose(
        execute(circuit, backend=framework).statevector, statevector(circuit), atol=2e-12
    )
    restored = convert(
        convert(circuit, source="ir", target=framework).value, source=framework, target="ir"
    ).value
    np.testing.assert_allclose(statevector(restored), statevector(circuit), atol=2e-12)


@pytest.mark.parametrize("gate", ["crx", "cry", "crz"])
def test_controlled_rotation_shift_uses_half_frequency(gate):
    circuit = Circuit(2).h(0).gate(gate, 0, 1, params=(Parameter("theta"),))
    theta = 0.73
    observable = PauliString("XI")
    assert expectation(circuit, observable, {"theta": theta}) == pytest.approx(np.cos(theta / 2))
    actual = parameter_shift(circuit, observable, {"theta": theta})["theta"]
    assert actual == pytest.approx(-np.sin(theta / 2) / 2, abs=1e-12)


def test_u_and_shared_controlled_parameter_gradients_match_finite_differences():
    t, phi, lam = Parameter("theta"), Parameter("phi"), Parameter("lam")
    circuit = (
        Circuit(3)
        .h(0)
        .u(t, phi, lam, 2)
        .crx(2 * t - phi, 0, 2)
        .cry(lam + t, 2, 1)
        .crz(phi - t / 3, 1, 0)
        .cp(phi + lam, 0, 2)
    )
    bindings = {"theta": 0.43, "phi": -0.61, "lam": 0.18}
    observable, eps = PauliString("XZY"), 1e-6
    gradients = parameter_shift(circuit, observable, bindings)
    for name in bindings:
        plus, minus = dict(bindings), dict(bindings)
        plus[name] += eps
        minus[name] -= eps
        finite = (
            expectation(circuit, observable, plus) - expectation(circuit, observable, minus)
        ) / (2 * eps)
        assert gradients[name] == pytest.approx(finite, abs=2e-8)


@pytest.mark.parametrize("version", [2, 3])
def test_all_new_gates_export_to_explicit_standard_qasm(version):
    circuit = (
        Circuit(3)
        .h(0)
        .u(0.51, -0.24, 0.83, 1)
        .crx(0.43, 0, 2)
        .cry(-0.27, 2, 1)
        .crz(0.63, 1, 0)
        .cp(0.49, 2, 0)
        .ch(1, 2)
        .cswap(0, 2, 1)
        .iswap(2, 0)
        .sxdg(1)
    )
    output = to_openqasm(circuit, version=version)
    restored = from_openqasm(output)
    np.testing.assert_allclose(statevector(restored), statevector(circuit), atol=2e-12)
    if version == 2:
        pytest.importorskip("qiskit")
        from qiskit import qasm2

        native = qasm2.loads(output)
        restored = convert(native, source="qiskit", target="ir").value
        np.testing.assert_allclose(statevector(restored), statevector(circuit), atol=2e-12)


def test_qasm3_uppercase_u_and_qasm2_ambiguity_diagnostic():
    restored = from_openqasm("OPENQASM 3.0; qubit q; U(0.5,0.2,-0.3) q;")
    np.testing.assert_allclose(statevector(restored), statevector(Circuit(1).u(0.5, 0.2, -0.3, 0)))
    with pytest.raises(ConversionError, match="QASM2_U_PHASE_CONVENTION"):
        from_openqasm('OPENQASM 2.0; include "qelib1.inc"; qreg q[1]; U(0.5,0.2,-0.3) q[0];')


@pytest.mark.parametrize("target", ["cirq", "openqasm3"])
def test_successful_conversion_reports_exact_lowering(target):
    if target == "cirq":
        pytest.importorskip("cirq")
    result = convert(Circuit(1).u(0.2, 0.3, 0.4, 0), source="ir", target=target)
    assert result.diagnostics[0].code == "EXACT_GATE_DECOMPOSITION"
    assert result.diagnostics[0].severity == "info"
    assert result.diagnostics[0].location == "operations[0]"


def test_cirq_native_controlled_gates_preserve_conditional_phase():
    cirq = pytest.importorskip("cirq")
    from openqml.adapters.cirq import from_cirq

    q = cirq.LineQubit.range(2)
    for gate in (
        cirq.rx(0.6).controlled(),
        cirq.ry(0.43).controlled(),
        cirq.ZPowGate(exponent=0.37).controlled(),
        cirq.XPowGate(exponent=0.47, global_shift=0.19).controlled(),
        cirq.CZPowGate(exponent=0.38, global_shift=-0.23),
    ):
        native = cirq.Circuit(cirq.H(q[0]), cirq.ry(0.27)(q[1]), gate(*q))
        restored = from_cirq(native)
        expected = (
            cirq.Simulator(dtype=np.complex128).simulate(native, qubit_order=q).final_state_vector
        )
        np.testing.assert_allclose(statevector(restored), expected, atol=2e-12)


def test_pennylane_rot_and_u2_import():
    qml = pytest.importorskip("pennylane")
    from openqml.adapters.pennylane import from_pennylane

    for op in (qml.Rot(0.23, 0.71, -0.37, wires=0), qml.U2(0.38, -0.61, wires=0)):
        tape = qml.tape.QuantumScript([qml.Hadamard(0), op], [])
        actual = statevector(from_pennylane(tape))
        expected = qml.matrix(op) @ (np.array([1, 1]) / np.sqrt(2))
        np.testing.assert_allclose(actual, expected, atol=2e-12)


@pytest.mark.parametrize("target", ["qiskit", "cirq"])
def test_extended_symbolic_conversion_preserves_all_u_angles(target):
    pytest.importorskip(target)
    theta, phi, lam = Parameter("theta"), Parameter("phi"), Parameter("lam")
    circuit = (
        Circuit(3)
        .h(0)
        .u(theta, phi, lam, 2)
        .crx(theta + phi, 0, 2)
        .cry(lam - theta / 3, 2, 1)
        .cp(phi, 1, 0)
        .with_global_phase(lam / 2)
    )
    restored = convert(
        convert(circuit, source="ir", target=target).value, source=target, target="ir"
    ).value
    for bindings in (
        {"theta": 0.43, "phi": -0.27, "lam": 0.61},
        {"theta": -0.81, "phi": 0.14, "lam": -0.38},
    ):
        np.testing.assert_allclose(
            statevector(restored, bindings), statevector(circuit, bindings), atol=2e-12
        )


@pytest.mark.parametrize("alias", ["U2Gate", "U3Gate", "CU1Gate"])
def test_qiskit_legacy_aliases_preserve_matrix(alias):
    qiskit = pytest.importorskip("qiskit")
    from qiskit.circuit import library
    from qiskit.quantum_info import Statevector

    constructor = getattr(library, alias, None)
    if constructor is None:
        pytest.skip("Legacy class not provided by this Qiskit version")
    values = {"U2Gate": (0.31, -0.47), "U3Gate": (0.53, -0.61, 0.14), "CU1Gate": (0.37,)}
    gate = constructor(*values[alias])
    native = qiskit.QuantumCircuit(2)
    native.h(0)
    native.ry(0.63, 1)
    native.append(gate, list(range(gate.num_qubits)))
    restored = convert(native, source="qiskit", target="ir").value
    expected = np.asarray(Statevector.from_instruction(native).data).reshape(2, 2).T.reshape(-1)
    np.testing.assert_allclose(statevector(restored), expected, atol=2e-12)


@pytest.mark.parametrize("backend", ["qiskit", "pennylane", "cirq"])
def test_parameter_shift_uses_independent_framework_backend(backend):
    pytest.importorskip(backend)
    circuit = Circuit(2).h(0).u(Parameter("theta"), 0.31, -0.47, 1).crx(Parameter("theta"), 0, 1)
    observable, bindings = PauliString("XZ"), {"theta": 0.43}
    expected = parameter_shift(circuit, observable, bindings)
    actual = parameter_shift(circuit, observable, bindings, backend=backend)
    assert actual["theta"] == pytest.approx(expected["theta"], abs=2e-12)
