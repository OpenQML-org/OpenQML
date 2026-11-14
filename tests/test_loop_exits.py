"""Finite-loop exit semantics across mutable selectors, nesting and wire maps."""

import numpy as np
import pytest

import openqml as oq


@pytest.mark.parametrize("attempts", [1, 3, 6])
@pytest.mark.parametrize("angle", [0, 0.4, 1.1, np.pi])
def test_bounded_measurement_loop_break_has_exact_terminal_success_probability(attempts, angle):
    sdk = pytest.importorskip("qiskit")
    native = sdk.QuantumCircuit(1, 1)
    with native.for_loop(range(attempts)):
        native.ry(angle, 0)
        native.measure(0, 0)
        with native.if_test((native.clbits[0], 1)):
            native.break_loop()
    circuit = oq.convert(native, target="ir").value
    failure = np.cos(angle / 2) ** (2 * attempts)
    np.testing.assert_allclose(
        oq.density_matrix(circuit), np.diag([failure, 1 - failure]), atol=2e-13
    )


@pytest.mark.parametrize("attempts", [2, 5])
def test_continue_discards_iteration_suffix_but_keeps_next_measurement(attempts):
    sdk = pytest.importorskip("qiskit")
    native = sdk.QuantumCircuit(2, 1)
    angle = 0.7
    with native.for_loop(range(attempts)):
        native.ry(angle, 0)
        native.measure(0, 0)
        with native.if_test((native.clbits[0], 1)):
            native.reset(0)
            native.continue_loop()
        native.x(1)
    circuit = oq.convert(native, target="ir").value
    chance = np.cos(angle / 2) ** 2
    odd = (1 - (1 - 2 * chance) ** attempts) / 2
    np.testing.assert_allclose(
        oq.density_matrix(circuit), np.diag([1 - odd, odd, 0, 0]), atol=2e-13
    )


def test_switch_selects_once_even_when_payload_overwrites_selector_and_exits_loop():
    sdk = pytest.importorskip("qiskit")
    native = sdk.QuantumCircuit(4, 1)
    with native.for_loop(range(5)):
        with native.switch(native.clbits[0]) as case:
            with case(0):
                native.x(1)
                native.x(0)
                native.measure(0, 0)
                native.continue_loop()
            with case(1):
                native.x(2)
                native.break_loop()
        native.h(3)
    native.x(3)
    converted = oq.convert(native, target="ir").value
    expected = oq.Circuit(4).x(0).x(1).x(2).x(3)
    np.testing.assert_allclose(
        oq.density_matrix(converted), oq.density_matrix(expected), atol=1e-14
    )


def test_nested_exits_bind_closest_loop_preserve_symbolic_phase_and_reversed_resources():
    sdk = pytest.importorskip("qiskit")
    from qiskit.circuit import BreakLoopOp, ContinueLoopOp, Parameter

    i, j, w = Parameter("i"), Parameter("j"), Parameter("w")
    inner = sdk.QuantumCircuit(2, 2)
    inner.rz(w * (j + 1), 1)
    inner.global_phase = w / 7
    jump = sdk.QuantumCircuit(2, 2)
    jump.append(BreakLoopOp(2, 2), jump.qubits, jump.clbits)
    inner.if_else((inner.clbits[1], 0), jump, None, (0, 1), (0, 1))
    inner.x(0)
    outer = sdk.QuantumCircuit(2, 2)
    outer.ry(w * (i + 1), 1)
    outer.for_loop(range(4), j, inner, (0, 1), (0, 1))
    outer.global_phase = w * i / 9
    jump = sdk.QuantumCircuit(2, 2)
    jump.append(ContinueLoopOp(2, 2), jump.qubits, jump.clbits)
    outer.if_else((outer.clbits[1], 0), jump, None, (0, 1), (0, 1))
    outer.x(0)
    native = sdk.QuantumCircuit(3, 4)
    native.for_loop((0, 1, 3), i, outer, (2, 0), (3, 1))
    native.rx(w / 2, 0)
    converted = oq.convert(native, target="ir").value
    parameter = oq.Parameter("w")
    expected = oq.Circuit(3)
    for index in (0, 1, 3):
        expected = expected.ry(parameter * (index + 1), 0).rz(parameter, 0)
    expected = expected.rx(parameter / 2, 0).with_global_phase(parameter * (3 / 7 + 4 / 9))

    def select_zero_bits(circuit):
        result = oq.Circuit(circuit.num_qubits).with_global_phase(circuit.global_phase)
        for operation in circuit.operations:
            if operation.name == "if_else":
                block = operation.blocks[int(operation.condition.value != 0)]
                result = result.compose(select_zero_bits(block))
            else:
                result = result.append(operation)
        return result

    np.testing.assert_allclose(
        oq.statevector(select_zero_bits(converted), {"w": 0.4}),
        oq.statevector(expected, {"w": 0.4}),
        atol=1e-13,
    )
    np.testing.assert_allclose(
        oq.density_matrix(converted, {"w": 0.4}),
        oq.density_matrix(expected, {"w": 0.4}),
        atol=1e-13,
    )


@pytest.mark.parametrize("bit", [0, 1])
def test_cirq_preserves_loop_exits_when_the_continuation_has_no_measurements(bit):
    sdk = pytest.importorskip("qiskit")
    pytest.importorskip("cirq")
    native = sdk.QuantumCircuit(2, 1)
    if bit:
        native.x(0)
    native.measure(0, 0)
    with native.for_loop(range(3)):
        native.ry(0.4, 1)
        with native.if_test((native.clbits[0], 1)):
            native.break_loop()
        native.rx(0.2, 1)
    exported = oq.convert(native, target="cirq").value
    restored = oq.convert(exported, source="cirq", target="ir").value
    expected = oq.Circuit(2)
    if bit:
        expected = expected.x(0).ry(0.4, 1)
    else:
        for _ in range(3):
            expected = expected.ry(0.4, 1).rx(0.2, 1)
    np.testing.assert_allclose(oq.density_matrix(restored), oq.density_matrix(expected), atol=1e-13)


def test_typed_predicate_loop_exit_can_modify_its_read_bits():
    sdk = pytest.importorskip("qiskit")
    from qiskit.circuit.classical import expr

    native = sdk.QuantumCircuit(3, 2)
    native.x(0)
    native.measure(0, 0)
    with native.for_loop(range(4)):
        with native.if_test(expr.bit_xor(native.clbits[0], native.clbits[1])):
            native.x(0)
            native.measure(0, 0)
            native.continue_loop()
        native.x(2)
        native.break_loop()
    converted = oq.convert(native, target="ir").value
    np.testing.assert_allclose(
        oq.density_matrix(converted), oq.density_matrix(oq.Circuit(3).x(2)), atol=1e-13
    )


@pytest.mark.parametrize("target", ["qasm3", "qiskit"])
def test_bounded_exit_program_survives_external_round_trip(target):
    sdk = pytest.importorskip("qiskit")
    if target == "qasm3":
        pytest.importorskip("openqasm3")
    native = sdk.QuantumCircuit(1, 1)
    with native.for_loop(range(3)):
        native.ry(0.4, 0)
        native.measure(0, 0)
        with native.if_test((native.clbits[0], 1)):
            native.break_loop()
    converted = oq.convert(native, target=target).value
    restored = oq.convert(converted, source=target, target="ir").value
    failure = np.cos(0.2) ** 6
    np.testing.assert_allclose(
        oq.density_matrix(restored), np.diag([failure, 1 - failure]), atol=2e-13
    )


def test_loop_exit_expansion_rejects_unbounded_or_excessive_work():
    sdk = pytest.importorskip("qiskit")
    from qiskit.circuit import BreakLoopOp

    stray = sdk.QuantumCircuit(1)
    stray.append(BreakLoopOp(1, 0), [0])
    with pytest.raises(oq.ConversionError, match="CLASSICAL_CONTROL"):
        oq.convert(stray, target="ir")
    native = sdk.QuantumCircuit(1, 1)
    with native.for_loop(range(1000)):
        with native.if_test((native.clbits[0], 1)):
            native.break_loop()
        native.h(0)
    for options in ({"max_operations": 20}, {"max_depth": 4}, {"expand": False}):
        with pytest.raises(oq.ConversionError):
            oq.convert(native, target="ir", source_options=options)
    runtime = sdk.QuantumCircuit(1, 1)
    with runtime.while_loop((runtime.clbits[0], 0)):
        runtime.break_loop()
    with pytest.raises(oq.ConversionError, match="CLASSICAL_CONTROL"):
        oq.convert(runtime, target="ir")
