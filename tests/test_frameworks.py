import importlib.util

import numpy as np
import pytest

from openqml import Circuit, ConversionError, Parameter, convert, execute, sample, statevector


def test_cirq_symbolic_hardware_gates_preserve_full_unitaries_and_expansion_limits():
    cirq = pytest.importorskip("cirq")
    sympy = pytest.importorskip("sympy")
    a, b = cirq.LineQubit.range(2)
    theta, phi = sympy.symbols("theta phi")
    gates = [
        (cirq.FSimGate(theta, phi)(a, b), 3),
        (cirq.ISwapPowGate(exponent=theta, global_shift=0.125)(b, a), 2),
        (cirq.PhasedXPowGate(phase_exponent=phi, exponent=theta, global_shift=-0.2)(b), 3),
        (cirq.ControlledGate(cirq.YPowGate(exponent=theta))(b, a), 2),
    ]
    for operation, count in gates:
        source = cirq.Circuit(operation)
        ir = convert(source, target="ir", max_operations=count).value
        assert len(ir.operations) == count and all(op.name != "unitary" for op in ir.operations)
        with pytest.raises(ConversionError, match="EXPANSION_LIMIT"):
            convert(source, target="ir", max_operations=count - 1)
        for angle in (-0.73, 0.0, 1.28):
            bindings = {"theta": angle, "phi": angle - 0.32}
            target = convert(
                ir, target="cirq", bindings={k: bindings[k] for k in ir.parameters}
            ).value
            reference = cirq.resolve_parameters(source, bindings)
            np.testing.assert_allclose(cirq.unitary(target), cirq.unitary(reference), atol=2e-14)


def full_circuit():
    return (
        Circuit(3)
        .h(2)
        .x(1)
        .y(0)
        .z(2)
        .s(1)
        .sdg(0)
        .t(2)
        .tdg(1)
        .sx(0)
        .id(1)
        .rx(0.17, 2)
        .ry(0.43, 0)
        .rz(-0.82, 1)
        .p(0.27, 2)
        .cx(2, 0)
        .cy(1, 2)
        .cz(0, 2)
        .swap(1, 0)
        .ccx(2, 0, 1)
        .rxx(0.33, 2, 0)
        .ryy(-0.23, 1, 2)
        .rzz(0.41, 0, 1)
        .with_global_phase(0.37)
    )


@pytest.mark.parametrize("framework", ["qiskit", "pennylane", "cirq"])
def test_native_execution_and_roundtrip_all_gates(framework):
    pytest.importorskip(framework)
    circuit = full_circuit()
    np.testing.assert_allclose(
        execute(circuit, backend=framework).statevector, statevector(circuit), atol=1e-12
    )
    native = convert(circuit, source="ir", target=framework).value
    restored = convert(native, source=framework, target="ir").value
    np.testing.assert_allclose(statevector(restored), statevector(circuit), atol=1e-12)


@pytest.mark.parametrize("framework", ["qiskit", "pennylane", "cirq"])
def test_idle_wire_identity_and_measurement_sampling(framework):
    pytest.importorskip(framework)
    for circuit in (Circuit(4).x(2), Circuit(1)):
        native = convert(circuit, source="ir", target=framework).value
        restored = convert(native, source=framework, target="ir").value
        assert restored.num_qubits == circuit.num_qubits
        np.testing.assert_allclose(
            execute(circuit, backend=framework).statevector, statevector(circuit), atol=1e-12
        )
    assert sample(Circuit(3).x(0).measure(0, 2), 17, backend=framework) == {"001": 17}


@pytest.mark.parametrize("framework", ["qiskit", "cirq"])
def test_symbolic_roundtrip_shared_angles_and_phase(framework):
    pytest.importorskip(framework)
    t, p = Parameter("theta"), Parameter("phi")
    circuit = (
        Circuit(3).h(1).ry(t * 2 - p / 3 + 0.2, 2).p(p, 0).rxx(t, 1, 2).with_global_phase(t + p)
    )
    native = convert(circuit, source="ir", target=framework).value
    restored = convert(native, source=framework, target="ir").value
    assert restored.parameters == circuit.parameters
    for bindings in ({"theta": 0.2, "phi": -0.4}, {"theta": -0.9, "phi": 0.7}):
        np.testing.assert_allclose(
            statevector(restored, bindings), statevector(circuit, bindings), atol=1e-12
        )


@pytest.mark.parametrize("framework", ["qiskit", "cirq"])
def test_classical_mapping_roundtrip(framework):
    pytest.importorskip(framework)
    circuit = Circuit(3).x(1).measure(2, 0).measure(1, 2)
    restored = convert(
        convert(circuit, source="ir", target=framework).value, source=framework, target="ir"
    ).value
    assert restored.num_clbits == 3
    assert sample(restored, 10) == sample(circuit, 10) == {"001": 10}


def test_qiskit_unsupported_operations_accumulate_diagnostics():
    qiskit = pytest.importorskip("qiskit")
    native = qiskit.QuantumCircuit(2)
    native.delay(10, 0)
    native.delay(10, 1)
    with pytest.raises(ConversionError) as exc:
        convert(native, source="qiskit", target="ir")
    assert len(exc.value.diagnostics) == 2
    assert exc.value.diagnostics[1].location == "operations[1]"


def test_qiskit_nonlinear_parameter_diagnosed():
    qiskit = pytest.importorskip("qiskit")
    from qiskit.circuit import Parameter as QParameter

    native, theta = qiskit.QuantumCircuit(1), QParameter("theta")
    native.rx(theta * theta, 0)
    restored = convert(native, source="qiskit", target="ir").value
    np.testing.assert_allclose(
        statevector(restored, {"theta": 0.3}), statevector(Circuit(1).rx(0.09, 0))
    )


def test_qiskit_custom_gate_cannot_impersonate_standard_name():
    qiskit = pytest.importorskip("qiskit")
    from qiskit.circuit import Gate

    impostor = Gate("x", 1, [])
    definition = qiskit.QuantumCircuit(1)
    definition.h(0)
    impostor.definition = definition
    native = qiskit.QuantumCircuit(1)
    native.append(impostor, [0])
    restored = convert(native, source="qiskit", target="ir").value
    np.testing.assert_allclose(statevector(restored), statevector(Circuit(1).h(0)))


def test_cirq_inverted_measurements_preserve_physical_state_feedback_and_budgets():
    cirq = pytest.importorskip("cirq")
    sympy = pytest.importorskip("sympy")
    a, b = cirq.LineQubit.range(2)
    native = cirq.Circuit(
        cirq.X(b),
        cirq.measure(a, b, key="pair", invert_mask=(True,)),
        cirq.X(b).with_classical_controls(sympy.Eq(sympy.Symbol("pair"), 3)),
        cirq.measure(a, key="c0"),
        cirq.measure(b, key="c1"),
    )
    records = cirq.Simulator(seed=4).run(native, repetitions=8).measurements
    np.testing.assert_array_equal(records["pair"], np.ones((8, 2)))
    np.testing.assert_array_equal(records["c0"] + records["c1"], np.zeros((8, 1)))
    ir = convert(native, target="ir", source_options={"measurement_map": {"pair": (3, 2)}}).value
    assert sample(ir, 8, backend="numpy") == {"0011": 8}
    single = cirq.Circuit(cirq.measure(a, key="c0", invert_mask=(True,)))
    assert len(convert(single, target="ir", max_operations=3).value.operations) == 3
    with pytest.raises(ConversionError, match="EXPANSION_LIMIT"):
        convert(single, target="ir", max_operations=2)


def test_cirq_unsupported_measurement_transform():
    cirq = pytest.importorskip("cirq")
    q = cirq.LineQubit(0)
    for op in (
        cirq.measure(q, key="c0", confusion_map={(0,): np.eye(2)}),
        cirq.measure(q, key="arbitrary_name"),
    ):
        with pytest.raises(ConversionError):
            convert(cirq.Circuit(op), source="cirq", target="ir")


def test_pennylane_readout_and_graph_diagnostics():
    qml = pytest.importorskip("pennylane")
    with qml.QueuingManager.stop_recording():
        tape = qml.tape.QuantumScript([qml.RX(0.2, 0)], [qml.expval(qml.Z(0))])
    with pytest.raises(ConversionError, match="READOUT_UNSUPPORTED"):
        convert(tape, source="pennylane", target="ir")
    if importlib.util.find_spec("torch"):
        import torch

        with qml.QueuingManager.stop_recording():
            tape = qml.tape.QuantumScript([qml.RX(torch.tensor(0.2, requires_grad=True), 0)], [])
        with pytest.raises(ConversionError, match="AUTODIFF_GRAPH"):
            convert(tape, source="pennylane", target="ir")


def test_qasm2_output_accepted_by_independent_qiskit_parser():
    pytest.importorskip("qiskit")
    from qiskit import qasm2

    from openqml.adapters.openqasm import to_openqasm
    from openqml.adapters.qiskit import from_qiskit

    circuit = full_circuit().with_global_phase(0)
    parsed = from_qiskit(qasm2.loads(to_openqasm(circuit, version=2)))
    np.testing.assert_allclose(statevector(parsed), statevector(circuit), atol=1e-12)


@pytest.mark.parametrize("framework", ["qiskit", "pennylane", "cirq"])
@pytest.mark.parametrize("seed", [7, 39, 81])
def test_randomized_nonadjacent_gate_execution(framework, seed):
    pytest.importorskip(framework)
    rng = np.random.default_rng(seed)
    circuit = Circuit(4)
    for _ in range(50):
        name = rng.choice(["rx", "ry", "rz", "p", "rxx", "ryy", "rzz", "cx", "cy", "swap"])
        one_qubit = name in {"rx", "ry", "rz", "p"}
        qubits = tuple(int(q) for q in rng.choice(4, size=1 if one_qubit else 2, replace=False))
        params = (rng.uniform(-3, 3),) if name not in {"cx", "cy", "swap"} else ()
        circuit = circuit.gate(str(name), *qubits, params=params)
    circuit = circuit.with_global_phase(rng.uniform(-3, 3))
    np.testing.assert_allclose(
        execute(circuit, backend=framework).statevector, statevector(circuit), atol=1e-12
    )
