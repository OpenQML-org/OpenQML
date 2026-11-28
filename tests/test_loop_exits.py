"""Finite-loop exit semantics across mutable selectors, nesting and wire maps."""

import numpy as np
import pytest

import openqml as oq


def test_unconditional_loop_keeps_its_resources_and_batches_density_readouts():
    from openqml.ir._structure import connected_components

    body = oq.Circuit(2).x(1).measure(1, 0).break_loop()
    source = oq.Circuit(2).while_loop(None, body)
    assert connected_components([source]) == (((0, 1), (1, 0)),)
    expected = [[-1], [1]]
    circuits = [source, source.x(1)]
    np.testing.assert_allclose(oq.expectations_batch(circuits, ["IZ"], backend="numpy"), expected)


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


def test_loop_scope_finite_expansion_limits_and_runtime_exit():
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
    circuit = oq.convert(runtime, target="ir").value
    assert circuit.operations[0].name == "while_loop"
    np.testing.assert_allclose(oq.density_matrix(circuit), np.diag([1, 0]))


def test_repeat_until_measurement_preserves_parity_and_termination():
    guard = oq.ClassicalCondition((0,), 1)
    body = oq.Circuit(2).reset(0).h(0).measure(0, 0).x(1).if_else(guard, oq.Circuit(2).break_loop())
    circuit = oq.Circuit(2).while_loop(None, body).measure(1, 1)
    counts = oq.sample(circuit, shots=12_000, seed=7)
    assert counts == oq.sample(circuit, shots=12_000, seed=7)
    assert set(counts) == {"10", "11"} and sum(counts.values()) == 12_000
    # The parity qubit is one after an odd number of fair trials.
    assert abs(counts["11"] / 12_000 - 2 / 3) < 0.02
    np.testing.assert_allclose(oq.density_matrix(circuit), np.diag([0, 0, 1 / 3, 2 / 3]))
    plan = oq.plan_conversion(circuit, target="openqasm3")
    assert plan.verify(max_work=48_000_000).matched
    with pytest.raises(oq.ResourceError, match="recurrent"):
        oq.density_matrix(oq.Circuit(1).while_loop(None, oq.Circuit(1)))
    with pytest.raises(oq.ResourceError, match="max_loop_steps"):
        oq.sample(oq.Circuit(1).while_loop(None, oq.Circuit(1)), shots=3, max_loop_steps=9)


def test_absorption_checks_rare_exits_and_detects_probability_leakage():
    guard = oq.ClassicalCondition((0,), 1)
    retry = oq.Circuit(1).reset(0).ry(2 * np.arcsin(np.sqrt(1e-10)), 0).measure(0, 0)
    loop = oq.Circuit(1).while_loop(None, retry.if_else(guard, oq.Circuit(1).break_loop()))
    np.testing.assert_allclose(oq.density_matrix(loop), np.diag([0, 1]), atol=3e-12)
    assert oq.plan_conversion(loop, target="json").verify().matched

    initial = oq.Circuit(1).h(0).measure(0, 0)
    dark = initial.while_loop(guard, oq.Circuit(1).measure(0, 0))
    with pytest.raises(oq.ResourceError, match="probability-one termination"):
        oq.density_matrix(dark)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_absorbing_loop_preserves_native_gradient_and_sampling(backend):
    xp = pytest.importorskip(backend)
    guard = oq.ClassicalCondition((0,), 1)
    angle = oq.Parameter("angle")
    body = (
        oq.Circuit(2)
        .reset(0)
        .ry(angle, 0)
        .measure(0, 0)
        .x(1)
        .if_else(guard, oq.Circuit(2).break_loop())
    )
    circuit = oq.Circuit(2).while_loop(None, body).measure(1, 1)
    value = 0.8
    probability = np.sin(value / 2) ** 2

    def odd(x):
        return oq.density_matrix(circuit, {"angle": x}, backend=backend).real[3, 3]

    if backend == "torch":
        x = xp.tensor(value, dtype=xp.float64, requires_grad=True)
        result = odd(x)
        (gradient,) = xp.autograd.grad(result, x)
        actual = float(result.detach()), float(gradient)
    else:
        x = xp.numpy.asarray(value)
        result, gradient = xp.jit(xp.value_and_grad(odd))(x)
        actual = float(result), float(gradient)
    np.testing.assert_allclose(
        actual,
        [1 / (2 - probability), np.sin(value) / (2 * (2 - probability) ** 2)],
        atol=4e-6,
    )
    counts = oq.sample(circuit, 3000, {"angle": x}, backend=backend, seed=44)
    assert set(counts) == {"10", "11"} and sum(counts.values()) == 3000
    assert abs(counts["11"] / 3000 - 1 / (2 - probability)) < 0.04


def test_cirq_native_repeat_preserves_terminal_measurement():
    cirq = pytest.importorskip("cirq")
    guard = oq.ClassicalCondition((0,), 1)
    retry = oq.Circuit(1).reset(0).h(0).measure(0, 0)
    circuit = oq.Circuit(1).while_loop(None, retry.if_else(guard, oq.Circuit(1).break_loop()))
    plan = oq.plan_conversion(circuit, target="cirq")
    native = plan.export().value
    assert next(native.all_operations()).repeat_until is not None
    assert cirq.Simulator(seed=7).simulate(native).measurements["c0"][-1] == 1
    assert plan.verify().matched


@pytest.mark.parametrize("target", ["cirq", "pennylane", "openqasm2"])
def test_portable_runtime_export_retains_control_and_executes_without_sdk(target):
    guard = oq.ClassicalCondition((0,), 1)
    body = oq.Circuit(1).x(0).measure(0, 0)
    circuit = body.while_loop(guard, body)
    plan = oq.plan_conversion(circuit, target=target, portable=True)
    assert plan.compatible and not plan.required_integrations
    transport = oq.PortableProgram.from_json(plan.export().value.to_json())
    assert transport.target_format == target and transport.circuit == circuit
    assert transport.execute(shots=5).counts == {"0": 5}
    assert plan.verify().matched


def test_finite_measurement_loop_can_revisit_the_same_classical_record():
    guard = oq.ClassicalCondition((0,), 1)
    body = oq.Circuit(2).measure(0, 0).x(0).x(1)
    circuit = oq.Circuit(2).classical_set(0, 1).h(0).while_loop(guard, body)

    # Outcome 0 exits at once in |11>. Outcome 1 revisits the same guard
    # with the first qubit flipped, then exits in |10>.
    np.testing.assert_allclose(oq.density_matrix(circuit), np.diag([0, 0, 0.5, 0.5]))
    assert oq.plan_conversion(circuit, target="openqasm3").verify().matched


def test_runtime_schema_and_unsupported_targets():
    inner = oq.Circuit(1).x(0).break_loop()
    outer = oq.Circuit(1).while_loop(None, inner).x(0).break_loop()
    circuit = oq.CircuitBuilder(1).while_loop(None, outer).build()
    assert oq.sample(circuit, shots=7) == {"0": 7}
    assert oq.Circuit.from_json(circuit.to_json()) == circuit
    old = circuit.to_dict()
    assert old["version"] == 6 and oq.Circuit(1).x(0).to_dict()["version"] == 5
    old["version"] = 5
    with pytest.raises(oq.ValidationError):
        oq.Circuit.from_dict(old)
    for target in ("pennylane", "openqasm2"):
        plan = oq.plan_conversion(circuit, target=target)
        assert plan.compatible and not plan.prepared_circuit.has_runtime_control
    assert oq.plan_conversion(circuit, target="openqasm2").verify().matched
    with pytest.raises(oq.ConversionError, match="CLASSICAL_CONTROL"):
        oq.convert(oq.Circuit(1).break_loop(), target="ir")
    with pytest.raises(oq.ResourceError, match="max_work"):
        oq.plan_conversion(circuit, target="openqasm3").verify(max_work=8)


def test_qiskit_mixed_loop_exits_match_aer():
    sdk = pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")
    native = sdk.QuantumCircuit(2, 2)
    with native.for_loop(range(2)):
        with native.while_loop((native.clbits[0], 0)):
            with native.for_loop(range(3)):
                native.x(1)
                native.break_loop()
            native.x(0)
            native.measure(0, 0)
            native.continue_loop()
            native.x(1)
        native.continue_loop()
        native.x(1)
    native.measure(1, 1)
    plan = oq.plan_conversion(native, target="qiskit")
    assert plan.verify().matched
    for backend in ("numpy", oq.AerBackend()):
        assert oq.sample(plan.circuit, backend=backend, shots=31, classical_bits=(0, 1)) == {
            "11": 31
        }
