"""Bounded symbolic division and correlation-aware selector range proofs."""

import pytest

import openqml as oq


def _sdk():
    qiskit = pytest.importorskip("qiskit")
    from qiskit.circuit.classical import expr, types

    if not hasattr(expr, "div"):
        pytest.skip("This Qiskit predates typed integer arithmetic.")
    return qiskit, expr, types


def test_variable_integer_division_is_exact_for_every_three_bit_input():
    qiskit, expr, types = _sdk()
    for a in range(8):
        for b in range(4):
            ar, br = qiskit.ClassicalRegister(3, "a"), qiskit.ClassicalRegister(2, "b")
            source = qiskit.QuantumCircuit(qiskit.QuantumRegister(6), ar, br)
            for i in range(5):
                if ((a | (b << 3)) >> i) & 1:
                    source.x(i)
                source.measure(i, i)
            positive = expr.add(expr.cast(br, types.Uint(3)), 1)
            with source.if_test(expr.equal(expr.div(ar, positive), 2)):
                source.x(5)
                source.measure(5, 0)
            ir = oq.convert(source, target="ir").value
            expected = -1 if a // (b + 1) == 2 else 1
            observable = oq.PauliString("IIIIIZ")
            assert oq.expectation(ir, observable) == pytest.approx(expected, abs=2e-12)
            qasm = oq.convert(ir, target="openqasm3").value
            back = oq.convert(qasm, source="openqasm3", target="ir").value
            assert oq.expectation(back, observable) == pytest.approx(expected, abs=2e-12)


@pytest.mark.parametrize("divisor", [1, 2, 3, 4, 5, 7])
def test_constant_divisors_preserve_unsigned_quotients(divisor):
    qiskit, expr, _ = _sdk()
    for a in range(8):
        source = qiskit.QuantumCircuit(4, 3)
        for i in range(3):
            if (a >> i) & 1:
                source.x(i)
            source.measure(i, i)
        with source.if_test(expr.equal(expr.div(source.cregs[0], divisor), 1)):
            source.x(3)
        ir = oq.convert(source, target="ir").value
        assert oq.expectation(ir, oq.PauliString("IIIZ")) == pytest.approx(
            -1 if a // divisor == 1 else 1
        )


def test_correlated_selector_bits_prove_a_range_that_independent_bit_bounds_miss():
    qiskit, expr, types = _sdk()
    for data in range(8):
        for selector in range(2):
            ar, br = qiskit.ClassicalRegister(3, "data"), qiskit.ClassicalRegister(1, "selector")
            source = qiskit.QuantumCircuit(qiskit.QuantumRegister(5), ar, br)
            for i in range(4):
                if ((data | (selector << 3)) >> i) & 1:
                    source.x(i)
                source.measure(i, i)
            low = expr.cast(br[0], types.Uint(2))
            high = expr.shift_left(expr.cast(expr.logic_not(br[0]), types.Uint(2)), 1)
            index = expr.bit_or(low, high)  # Exactly 1 or 2, never 0 or 3.
            with source.if_test(expr.index(ar, index)):
                source.x(4)
            ir = oq.convert(source, target="ir").value
            bit = 1 if selector else 2
            assert oq.expectation(ir, oq.PauliString("IIIIZ")) == pytest.approx(
                -1 if (data >> bit) & 1 else 1
            )


def test_undefined_divisors_and_quadratic_work_retain_explicit_diagnostics():
    qiskit, expr, types = _sdk()
    source = qiskit.QuantumCircuit(1, 3)
    with source.if_test(expr.equal(expr.div(source.cregs[0], source.cregs[0]), 1)):
        source.x(0)
    with pytest.raises(oq.ConversionError, match="DIVISOR"):
        oq.convert(source, target="ir")
    wide = qiskit.QuantumCircuit(1)
    with wide.if_test(
        expr.equal(expr.div(expr.Value(0, types.Uint(64)), expr.Value(3, types.Uint(64))), 0)
    ):
        wide.x(0)
    with pytest.raises(oq.ConversionError, match="LIMIT"):
        oq.convert(wide, target="ir", max_operations=1000)
