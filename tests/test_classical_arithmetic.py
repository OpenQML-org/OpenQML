"""Exhaustive finite-width arithmetic and dynamic selector semantics."""

import numpy as np
import pytest

import openqml as oq


def _sdk():
    qiskit = pytest.importorskip("qiskit")
    from qiskit.circuit.classical import expr, types

    if not all(hasattr(expr, name) for name in ("add", "sub", "mul", "index")):
        pytest.skip("The installed Qiskit predates typed arithmetic/index expressions.")
    return qiskit, expr, types


@pytest.mark.parametrize("kind", ["add", "sub", "mul", "left", "right", "index", "nested"])
def test_register_arithmetic_is_modular_and_branch_selection_precedes_writes(kind):
    qiskit, expr, types = _sdk()
    for a in range(4):
        for b in range(4):
            ar, br = qiskit.ClassicalRegister(2, "a"), qiskit.ClassicalRegister(2, "b")
            source = qiskit.QuantumCircuit(qiskit.QuantumRegister(5), ar, br)
            for i in range(4):
                if ((a | (b << 2)) >> i) & 1:
                    source.x(i)
                source.measure(i, i)
            amount = expr.cast(br[0], types.Uint(1))
            if kind == "add":
                predicate = expr.less(expr.add(ar, br), 2)
                expected = (a + b) % 4 < 2
            elif kind == "sub":
                predicate = expr.equal(expr.sub(ar, br), 3)
                expected = (a - b) % 4 == 3
            elif kind == "mul":
                predicate = expr.equal(expr.mul(ar, br), 2)
                expected = (a * b) % 4 == 2
            elif kind == "left":
                predicate = expr.equal(expr.shift_left(ar, amount), 2)
                expected = (a << (b & 1)) % 4 == 2
            elif kind == "right":
                predicate = expr.equal(expr.shift_right(ar, amount), 1)
                expected = a >> (b & 1) == 1
            elif kind == "index":
                predicate = expr.index(ar, amount)
                expected = bool((a >> (b & 1)) & 1)
            else:
                predicate = (
                    expr.logic_xor(expr.equal(ar, br), expr.equal(ar, 0))
                    if hasattr(expr, "logic_xor")
                    else expr.bit_xor(expr.equal(ar, br), expr.equal(ar, 0))
                )
                predicate = expr.logic_and(
                    predicate, expr.not_equal(expr.mul(expr.add(ar, 1), br), 0)
                )
                expected = ((a == b) != (a == 0)) and ((a + 1) * b) % 4 != 0
            with source.if_test(predicate) as otherwise:
                source.x(4)
                source.measure(4, 0)
                source.measure(4, 2)
            with otherwise:
                source.z(4)
                source.measure(4, 0)
                source.measure(4, 2)
            original = source.copy()
            ir = oq.convert(source, target="ir").value
            assert source == original
            want = -1 if expected else 1
            obs = oq.PauliString("IIIIZ")
            assert oq.expectation(ir, obs) == pytest.approx(want, abs=2e-12), (kind, a, b)
            qasm = oq.convert(ir, target="openqasm3").value
            back = oq.convert(qasm, source="openqasm3", target="ir").value
            assert oq.expectation(back, obs) == pytest.approx(want, abs=2e-12)


def test_provable_index_and_shift_ranges_and_expanded_work_have_bounded_diagnostics():
    qiskit, expr, types = _sdk()
    source = qiskit.QuantumCircuit(1, 2)
    for predicate in (
        expr.index(source.cregs[0], 2),
        expr.equal(expr.shift_left(source.cregs[0], 2), 0),
    ):
        circuit = source.copy()
        with circuit.if_test(predicate):
            circuit.x(0)
        with pytest.raises(oq.ConversionError, match="RANGE"):
            oq.convert(circuit, target="ir")
    dynamic = qiskit.QuantumCircuit(
        qiskit.QuantumRegister(1),
        qiskit.ClassicalRegister(2, "a"),
        qiskit.ClassicalRegister(2, "b"),
    )
    with dynamic.if_test(expr.index(dynamic.cregs[0], dynamic.cregs[1])):
        dynamic.x(0)
    with pytest.raises(oq.ConversionError, match="RANGE"):
        oq.convert(dynamic, target="ir")
    large = qiskit.QuantumCircuit(1)
    zero = expr.Value(0, types.Uint(200))
    with large.if_test(expr.equal(expr.mul(zero, zero), zero)):
        large.x(0)
    with pytest.raises(oq.ConversionError, match="LIMIT"):
        oq.convert(large, target="ir", max_operations=1000)


def test_constant_unsigned_overflow_folds_phase_and_cirq_preserves_arithmetic_guards():
    qiskit, expr, types = _sdk()
    source = qiskit.QuantumCircuit(1, global_phase=0.2)
    one, three = expr.Value(1, types.Uint(2)), expr.Value(3, types.Uint(2))
    with source.if_test(expr.equal(expr.add(three, one), expr.Value(0, types.Uint(2)))):
        source.x(0)
    ir = oq.convert(source, target="ir").value
    np.testing.assert_allclose(oq.statevector(ir), [0, np.exp(0.2j)], atol=2e-14)
    pytest.importorskip("cirq")
    source = qiskit.QuantumCircuit(3, 2)
    source.h(0)
    source.h(1)
    source.measure(0, 0)
    source.measure(1, 1)
    with source.if_test(expr.equal(expr.add(source.cregs[0], 1), 0)):
        source.x(2)
    ir = oq.convert(source, target="ir").value
    converted = oq.convert(ir, target="cirq").value
    back = oq.convert(converted, source="cirq", target="ir").value
    assert oq.expectation(back, oq.PauliString("IIZ")) == pytest.approx(0.5)


def test_lowered_arithmetic_runs_as_native_aer_equality_feedforward():
    qiskit, expr, _ = _sdk()
    pytest.importorskip("qiskit_aer")
    a, out = qiskit.ClassicalRegister(2, "a"), qiskit.ClassicalRegister(1, "out")
    source = qiskit.QuantumCircuit(qiskit.QuantumRegister(3), a, out)
    source.h(0)
    source.h(1)
    source.measure(0, a[0])
    source.measure(1, a[1])
    with source.if_test(expr.equal(expr.add(a, 1), 0)):
        source.x(2)
    source.measure(2, out[0])
    ir = oq.convert(source, target="ir").value
    expected = {"000", "100", "010", "111"}
    assert set(oq.sample(ir, shots=128, seed=17)) == expected
    assert set(oq.sample(ir, shots=128, seed=17, backend="aer")) == expected
