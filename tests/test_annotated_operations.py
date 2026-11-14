"""Deferred modifiers preserve symbolic parameters, phases and operand order."""

import itertools

import numpy as np
import pytest

import openqml as oq


@pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
@pytest.mark.parametrize("power", [-2, 0, 3])
def test_ordered_deferred_modifiers_match_bound_native_operators(order, power):
    qiskit = pytest.importorskip("qiskit")
    from qiskit.circuit import AnnotatedOperation, ControlModifier, InverseModifier, PowerModifier
    from qiskit.quantum_info import Operator

    angle, phase = qiskit.circuit.Parameter("angle"), qiskit.circuit.Parameter("phase")
    base = qiskit.QuantumCircuit(1, global_phase=phase)
    base.ry(angle, 0)
    base.rz(angle / 3, 0)
    modifiers = [InverseModifier(), PowerModifier(power), ControlModifier(2, 1)]
    annotated = AnnotatedOperation(base.to_gate(), [modifiers[i] for i in order])
    source = qiskit.QuantumCircuit(3, global_phase=0.13)
    source.append(annotated, [2, 0, 1])
    converted = oq.convert(source, target="ir").value
    bindings = {"angle": 0.43, "phase": -0.29}
    bound = {n: bindings[n] for n in converted.parameters}
    target = oq.convert(converted.bind(bound), target="qiskit").value
    expected = Operator(source.assign_parameters({angle: 0.43, phase: -0.29})).data
    np.testing.assert_allclose(Operator(target).data, expected, atol=2e-12, rtol=2e-12)
    assert len(source.data[0].operation.modifiers) == 3
    assert set(converted.parameters) == (set() if power == 0 else set(bindings))


def test_nested_fractional_rotation_modifiers_roundtrip_through_qasm_and_cirq():
    qiskit = pytest.importorskip("qiskit")
    pytest.importorskip("cirq")
    from qiskit.circuit import AnnotatedOperation, ControlModifier, InverseModifier, PowerModifier
    from qiskit.circuit.library import RYGate
    from qiskit.quantum_info import Operator

    angle = qiskit.circuit.Parameter("rotation")
    base = AnnotatedOperation(RYGate(angle), PowerModifier(0.5))
    source = qiskit.QuantumCircuit(2)
    source.append(AnnotatedOperation(base, [InverseModifier(), ControlModifier(1, 0)]), [1, 0])
    ir = oq.convert(source, target="ir").value.bind({"rotation": 0.7})
    reference = qiskit.QuantumCircuit(2)
    reference.append(
        AnnotatedOperation(
            AnnotatedOperation(RYGate(0.7), PowerModifier(0.5)),
            [InverseModifier(), ControlModifier(1, 0)],
        ),
        [1, 0],
    )
    expected = Operator(reference).data
    for target in ("openqasm3", "cirq"):
        output = oq.convert(ir, target=target).value
        restored = oq.convert(output, source=target, target="qiskit").value
        np.testing.assert_allclose(Operator(restored).data, expected, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("options", [{"max_operations": 4}, {"max_depth": 0}, {"expand": False}])
def test_deferred_synthesis_obeys_expansion_admission(options):
    qiskit = pytest.importorskip("qiskit")
    from qiskit.circuit import AnnotatedOperation, ControlModifier, PowerModifier
    from qiskit.circuit.library import RYGate

    source = qiskit.QuantumCircuit(3)
    source.append(
        AnnotatedOperation(RYGate(0.3), [PowerModifier(100), ControlModifier(2)]),
        range(3),
    )
    with pytest.raises(oq.ConversionError) as error:
        oq.convert(source, target="ir", source_options=options)
    assert any(d.code.endswith("_LIMIT") for d in error.value.diagnostics)


def test_deferred_numeric_matrix_power_respects_matrix_width():
    qiskit = pytest.importorskip("qiskit")
    from qiskit.circuit import AnnotatedOperation, PowerModifier
    from qiskit.circuit.library import UnitaryGate

    source = qiskit.QuantumCircuit(3)
    source.append(AnnotatedOperation(UnitaryGate(np.eye(8)), PowerModifier(0.5)), range(3))
    with pytest.raises(oq.ConversionError) as error:
        oq.convert(source, target="ir", source_options={"max_matrix_qubits": 2})
    assert any(d.code.endswith("_LIMIT") for d in error.value.diagnostics)
