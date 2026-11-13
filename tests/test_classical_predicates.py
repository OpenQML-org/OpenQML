"""Portable typed classical predicates preserve atomic feedforward and bounded work."""

import numpy as np
import pytest

import openqml as oq


def _condition(expr, circuit, kind):
    bits, register = circuit.clbits, circuit.cregs[0]
    choices = {
        "xor": (lambda: expr.bit_xor(bits[0], bits[1]), lambda v: bool((v & 1) ^ ((v >> 1) & 1))),
        "or": (lambda: expr.logic_or(bits[0], bits[2]), lambda v: bool(v & 1 or v & 4)),
        "not": (lambda: expr.logic_not(bits[1]), lambda v: not bool(v & 2)),
        "mask": (lambda: expr.equal(expr.bit_and(register, 5), 4), lambda v: v & 5 == 4),
        "uintxor": (lambda: expr.equal(expr.bit_xor(register, 5), 2), lambda v: v ^ 5 == 2),
        "less": (lambda: expr.less(register, 3), lambda v: v < 3),
        "le": (lambda: expr.less_equal(register, 3), lambda v: v <= 3),
        "greater": (lambda: expr.greater(register, 3), lambda v: v > 3),
        "ge": (lambda: expr.greater_equal(register, 3), lambda v: v >= 3),
        "neq": (lambda: expr.not_equal(register, 3), lambda v: v != 3),
        "complement": (lambda: expr.equal(expr.bit_not(register), 2), lambda v: v ^ 7 == 2),
        "contradiction": (
            lambda: expr.logic_and(bits[0], expr.logic_not(bits[0])),
            lambda v: False,
        ),
    }
    build, truth = choices[kind]
    return build(), truth


@pytest.mark.parametrize(
    "kind",
    [
        "xor",
        "or",
        "not",
        "mask",
        "uintxor",
        "less",
        "le",
        "greater",
        "ge",
        "neq",
        "complement",
        "contradiction",
    ],
)
def test_unsigned_predicate_truth_tables_and_atomic_selector_writes(kind):
    qiskit = pytest.importorskip("qiskit")
    from qiskit.circuit.classical import expr

    for value in range(8):
        source = qiskit.QuantumCircuit(4, 3)
        for bit in range(3):
            if value & (1 << bit):
                source.x(bit)
            source.measure(bit, bit)
        condition, truth = _condition(expr, source, kind)
        with source.if_test(condition) as otherwise:
            source.x(3)
            source.measure(3, 0)  # Overwrite a predicate input inside its payload.
        with otherwise:
            source.z(3)
            source.measure(3, 0)
        original = source.copy()
        ir = oq.convert(source, target="ir").value
        assert source == original
        expected = -1 if truth(value) else 1
        assert oq.expectation(ir, oq.PauliString("IIIZ")) == pytest.approx(expected, abs=2e-12)
        # Every emitted guard is in the portable equality subset. QASM remains
        # round-trippable without depending on an SDK's rich-expression importer.
        qasm = oq.convert(ir, target="openqasm3").value
        roundtrip = oq.convert(qasm, source="openqasm3", target="ir").value
        assert oq.expectation(roundtrip, oq.PauliString("IIIZ")) == pytest.approx(
            expected, abs=2e-12
        )


def test_predicate_native_aer_sampling_and_cirq_roundtrip_preserve_measured_feedforward():
    qiskit = pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")
    pytest.importorskip("cirq")
    from qiskit.circuit.classical import expr

    source = qiskit.QuantumCircuit(3, 3)
    source.h(0)
    source.h(1)
    source.measure(0, 0)
    source.measure(1, 1)
    with source.if_test(expr.bit_xor(source.clbits[0], source.clbits[1])):
        source.x(2)
    source.measure(2, 2)
    ir = oq.convert(source, target="ir").value
    obs = [oq.PauliString("IIZ"), oq.PauliString("ZIZ")]
    reference = oq.expectations(ir, obs)
    expected_counts = oq.sample(ir, shots=256, seed=41)
    native_counts = oq.sample(ir, shots=256, seed=41, backend="aer")
    assert len(expected_counts) == 4 and set(native_counts) == set(expected_counts)
    artifact = oq.convert(ir, target="cirq")
    imported = oq.convert(artifact.value, source="cirq", target="ir").value
    np.testing.assert_allclose(oq.expectations(imported, obs), reference, atol=2e-12)


def test_standalone_storage_and_expression_work_fail_with_explicit_diagnostics():
    qiskit = pytest.importorskip("qiskit")
    from qiskit.circuit.classical import expr, types

    source = qiskit.QuantumCircuit(1)
    flag = source.add_input("flag", types.Bool())
    with source.if_test(expr.logic_not(flag)):
        source.x(0)
    with pytest.raises(oq.ConversionError, match="CLASSICAL"):
        oq.convert(source, target="ir")
    source = qiskit.QuantumCircuit(1, 3)
    with source.if_test(expr.less(source.cregs[0], 5)):
        source.x(0)
    with pytest.raises(oq.ConversionError, match="LIMIT|DEPTH"):
        oq.convert(source, target="ir", source_options={"max_operations": 5})


def test_register_comparisons_casts_and_constant_phase_folding_are_exact():
    qiskit = pytest.importorskip("qiskit")
    from qiskit.circuit.classical import expr, types

    for left in range(4):
        for right in range(4):
            source = qiskit.QuantumCircuit(5)
            a, b = qiskit.ClassicalRegister(2, "a"), qiskit.ClassicalRegister(2, "b")
            source.add_register(a, b)
            for bit in range(4):
                value = left if bit < 2 else right
                if value & (1 << (bit % 2)):
                    source.x(bit)
                source.measure(bit, bit)
            condition = expr.logic_and(expr.less(a, b), expr.cast(a, types.Bool()))
            with source.if_test(condition):
                source.x(4)
            ir = oq.convert(source, target="ir").value
            assert oq.expectation(ir, oq.PauliString("IIIIZ")) == pytest.approx(
                -1 if left < right and left else 1, abs=2e-12
            )
    source = qiskit.QuantumCircuit(1, 1)
    source.global_phase = 0.2
    with source.if_test(expr.logic_or(source.clbits[0], expr.logic_not(source.clbits[0]))):
        source.ry(0.4, 0)
    source.data[-1].operation.blocks[0].global_phase = 0.37
    ir = oq.convert(source, target="ir").value
    assert not ir.has_effects
    expected = np.exp(0.57j) * oq.statevector(oq.Circuit(1).ry(0.4, 0))
    np.testing.assert_allclose(oq.statevector(ir), expected, atol=2e-12)
