"""Classical mutation preserves quantum instruments, entry values and storage layout."""

import numpy as np
import pytest

import openqml as oq


def _sdk():
    qiskit = pytest.importorskip("qiskit")
    from qiskit.circuit.classical import expr, types

    return qiskit, expr, types


def test_classical_assignment_changes_storage_without_changing_quantum_state():
    circuit = oq.Circuit(1).h(0).classical_set(3, 1).classical_set(1, 1).classical_set(3, 0)
    assert not circuit.has_measurements and circuit.has_classical_outputs
    result = oq.execute(circuit, shots=23)
    assert result.counts == {"0100": 23}
    np.testing.assert_allclose(result.density_matrix, np.ones((2, 2)) / 2, atol=1e-15)
    assert oq.execute(circuit, shots=23, classical_bits=(3, 1)).counts == {"01": 23}
    builder = oq.CircuitBuilder(1).h(0).classical_set(1, 1).build()
    assert oq.execute(builder, shots=7).counts == {"01": 7}


def test_storage_overwrites_merge_quantum_branches_and_preserve_atomic_guards():
    circuit = (
        oq.Circuit(1)
        .h(0)
        .measure(0, 0)
        .if_else(
            oq.ClassicalCondition((0,), 1),
            oq.Circuit(1).classical_set(0, 0).x(0),
            oq.Circuit(1).classical_set(0, 0),
        )
    )
    result = oq.execute(circuit, shots=19)
    assert result.counts == {"0": 19}
    np.testing.assert_allclose(result.probabilities, [1, 0], atol=1e-15)
    discarded = oq.Circuit(1)
    for bit in range(10):
        discarded = discarded.h(0).measure(0, bit)
    for bit in range(10):
        discarded = discarded.classical_set(bit, 0)
    plan = oq.plan_density(discarded, retain_classical=True, max_branches=1)
    assert plan.peak_branches == 1 and plan.output_branches == 1
    assert oq.execute(discarded, shots=5, max_branches=1).counts == {"0000000000": 5}


def test_storage_schema_and_unsupported_targets():
    circuit = oq.Circuit(1).classical_set(0, 1)
    assert oq.Circuit.from_json(circuit.to_json()) == circuit
    payload = circuit.to_dict()
    assert payload["version"] == 5
    payload["version"] = 4
    with pytest.raises(oq.ValidationError, match="classical assignment"):
        oq.Circuit.from_dict(payload)
    payload = oq.Circuit(1).h(0).to_dict()
    payload["version"] = 4
    assert oq.Circuit.from_dict(payload) == oq.Circuit(1).h(0)
    for value in (-1, 2, oq.Parameter("bit")):
        with pytest.raises(oq.ValidationError):
            oq.Circuit(1).classical_set(0, value)
    for target in ("cirq", "pennylane", "openqasm2"):
        assert any(d.code == "CLASSICAL_STORE_UNSUPPORTED" for d in oq.diagnose(circuit, target))


@pytest.mark.parametrize("value", range(4))
def test_register_store_resolves_all_rhs_bits_before_any_target_write(value):
    qiskit, expr, _ = _sdk()
    native = qiskit.QuantumCircuit(2, 2)
    for bit in range(2):
        if (value >> bit) & 1:
            native.x(bit)
        native.measure(bit, bit)
    native.store(native.cregs[0], expr.add(native.cregs[0], 1))
    original = native.copy()
    result = oq.convert(native, target="ir")
    assert native == original
    circuit = result.value
    assert circuit.num_qubits == 2 and circuit.num_clbits == 4
    assert any(d.code == "CLASSICAL_STORAGE_LOWERED" for d in result.diagnostics)
    output = (value + 1) % 4
    expected = "".join(str((output >> i) & 1) for i in range(2))
    assert oq.execute(circuit, shots=11, classical_bits=(0, 1)).counts == {expected: 11}


def test_runtime_variables_stores_and_loop_exits():
    qiskit, expr, types = _sdk()
    if not hasattr(expr, "add"):
        pytest.skip("Typed arithmetic requires a recent Qiskit.")
    native = qiskit.QuantumCircuit(2, 2)
    counter = native.add_var("counter", expr.Value(0, types.Uint(3)))
    with native.for_loop(range(5)):
        native.store(counter, expr.add(counter, 1))
        with native.if_test(expr.equal(counter, 2)):
            native.continue_loop()
        with native.if_test(expr.equal(counter, 3)):
            native.break_loop()
        native.x(0)
    with native.if_test(expr.equal(counter, 3)):
        native.x(1)
    native.measure([0, 1], [0, 1])
    circuit = oq.convert(native, target="ir").value
    assert oq.execute(circuit, shots=13, classical_bits=(0, 1)).counts == {"11": 13}


def test_switch_variable_selector_is_atomic_when_payload_mutates_it():
    qiskit, expr, types = _sdk()
    native = qiskit.QuantumCircuit(2, 2)
    selector = native.add_var("selector", expr.Value(1, types.Uint(2)))
    with native.switch(selector) as case:
        with case(1):
            native.store(selector, 2)
            native.x(0)
        with case(2):
            native.x(1)
    native.measure([0, 1], [0, 1])
    circuit = oq.convert(native, target="ir").value
    assert oq.execute(circuit, shots=13, classical_bits=(0, 1)).counts == {"10": 13}


def test_classical_inputs_and_definite_assignment():
    qiskit, expr, types = _sdk()
    native = qiskit.QuantumCircuit(1, 1)
    selector = native.add_input("selector", types.Uint(2))
    with native.if_test(expr.equal(selector, 2)):
        native.x(0)
    native.measure(0, 0)
    with pytest.raises(oq.ConversionError) as error:
        oq.convert(native, target="ir")
    assert error.value.diagnostics[0].code == "CLASSICAL_INPUT_UNBOUND"
    result = oq.convert(native, target="ir", source_options={"classical_inputs": {"selector": 2}})
    assert oq.execute(result.value, shots=5, classical_bits=(0,)).counts == {"1": 5}
    uninitialized = qiskit.QuantumCircuit(1, 1)
    variable = expr.Var.new("missing", types.Bool())
    uninitialized.add_uninitialized_var(variable)
    with uninitialized.if_test((uninitialized.clbits[0], 1)):
        uninitialized.store(variable, True)
    with uninitialized.if_test(variable):
        uninitialized.x(0)
    with pytest.raises(oq.ConversionError) as error:
        oq.convert(uninitialized, target="ir")
    assert error.value.diagnostics[0].code == "CLASSICAL_UNINITIALIZED"


def test_stored_feedback_roundtrips_as_a_complete_quantum_instrument():
    qiskit, expr, types = _sdk()
    native = qiskit.QuantumCircuit(1, 1)
    variable = native.add_var("flag", expr.Value(False, types.Bool()))
    native.h(0)
    native.measure(0, 0)
    native.store(variable, expr.logic_not(native.clbits[0]))
    with native.if_test(variable):
        native.x(0)
    native.store(native.clbits[0], variable)
    circuit = oq.convert(native, target="ir").value
    for target in ("qiskit", "openqasm3", "json"):
        plan = oq.plan_conversion(circuit, target=target)
        checked = plan.verify(max_branches=8)
        assert checked.matched, checked
    qasm = oq.convert(circuit, target="openqasm3").value
    imported = oq.convert(
        qasm, source="openqasm3", target="ir", source_options={"parser": "builtin"}
    ).value
    assert imported == circuit


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_stores_and_feedback_have_native_parameter_gradients(backend):
    angle = oq.Parameter("angle")
    circuit = (
        oq.Circuit(1)
        .ry(angle, 0)
        .measure(0, 0)
        .if_else(
            oq.ClassicalCondition((0,), 1),
            oq.Circuit(1).classical_set(0, 0).x(0),
            oq.Circuit(1).classical_set(0, 1),
        )
        .if_else(oq.ClassicalCondition((0,), 1), oq.Circuit(1).x(0))
    )
    if backend == "torch":
        torch = pytest.importorskip("torch")
        value = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
        result = oq.expectation(circuit, oq.PauliString("Z"), {"angle": value}, backend="torch")
        gradient = torch.autograd.grad(result, value)[0]
        assert gradient.item() == pytest.approx(np.sin(0.4), abs=1e-12)
    else:
        jax = pytest.importorskip("jax")
        import jax.numpy as jnp

        def function(x):
            return oq.expectation(circuit, oq.PauliString("Z"), {"angle": x}, backend="jax")

        gradient = jax.jit(jax.grad(function))(jnp.array(0.4))
        assert float(gradient) == pytest.approx(np.sin(0.4), abs=2e-6)


def test_storage_and_expression_expansion_budgets():
    qiskit, expr, types = _sdk()
    native = qiskit.QuantumCircuit(1)
    native.add_var("wide", expr.Value(1, types.Uint(1000)))
    with pytest.raises(oq.ConversionError) as error:
        oq.convert(native, target="ir", source_options={"max_operations": 20})
    assert error.value.diagnostics[0].code == "EXPANSION_LIMIT"
