"""Bounded normalization of deferred unitary modifiers."""

from ...errors import ConversionError, ValidationError, conversion_error
from ...ir.parameters import real
from ._common import check_matrix_width


def annotated_definition(operation, *, max_operations, max_depth, max_matrix_qubits, location):
    from qiskit import QuantumCircuit
    from qiskit.circuit import (
        AnnotatedOperation,
        ControlModifier,
        Gate,
        InverseModifier,
        PowerModifier,
    )
    from qiskit.circuit.library import UnitaryGate

    max_operations, max_depth = int(max_operations), int(max_depth)
    spent = 0

    def charge(amount):
        nonlocal spent
        if amount > max_operations - spent:
            raise conversion_error(
                "EXPANSION_LIMIT", "Deferred modifiers exceed max_operations.", location
            )
        spent += amount

    def resolve(value, depth):
        if depth > max_depth:
            raise conversion_error(
                "EXPANSION_LIMIT", "Deferred modifiers exceed max_depth.", location
            )
        charge(1)
        if not isinstance(value, AnnotatedOperation):
            if not isinstance(value, Gate) or value.num_clbits:
                raise conversion_error(
                    "ANNOTATED_UNITARY", "Deferred modifiers require a unitary gate.", location
                )
            if isinstance(value, UnitaryGate):
                check_matrix_width(value.num_qubits, max_matrix_qubits, location)
            definition = getattr(value, "_definition", None)
            cost = max(1, len(definition.data)) if definition is not None else 1
            charge(cost)
            return value, cost
        result, cost = resolve(value.base_op, depth + 1)
        for modifier in value.modifiers:
            charge(1)
            if isinstance(modifier, InverseModifier):
                result = result.inverse()
            elif isinstance(modifier, ControlModifier):
                count = modifier.num_ctrl_qubits
                if count == 0:
                    continue
                if count < 1 or count >= max(1, max_operations.bit_length()):
                    raise conversion_error(
                        "EXPANSION_LIMIT", "Deferred control width exceeds its budget.", location
                    )
                cost *= 1 << count
                charge(cost)
                result = result.control(count, ctrl_state=modifier.ctrl_state, annotated=False)
            elif isinstance(modifier, PowerModifier):
                try:
                    power = real(modifier.power)
                except ValidationError as exc:
                    raise conversion_error(
                        "ANNOTATED_POWER",
                        "Deferred powers require finite numeric exponents.",
                        location,
                    ) from exc
                if power == int(power):
                    count = abs(int(power))
                    cost *= max(1, count)
                    charge(cost)
                    base = result.inverse() if power < 0 else result
                    circuit = QuantumCircuit(base.num_qubits)
                    for _ in range(count):
                        circuit.append(base, circuit.qubits)
                    result = circuit.to_gate()
                else:
                    check_matrix_width(result.num_qubits, max_matrix_qubits, location)
                    result = result.power(power, annotated=False)
            else:
                raise conversion_error(
                    "ANNOTATED_MODIFIER", "Unsupported deferred gate modifier.", location
                )
            if isinstance(result, AnnotatedOperation):
                raise conversion_error(
                    "ANNOTATED_SYNTHESIS", "The SDK did not resolve this modifier.", location
                )
        return result, cost

    try:
        gate, _ = resolve(operation, 0)
        circuit = QuantumCircuit(gate.num_qubits)
        circuit.append(gate, circuit.qubits)
        return circuit
    except ConversionError:
        raise
    except Exception as exc:
        raise conversion_error(
            "ANNOTATED_SYNTHESIS", f"Cannot synthesize deferred gate: {exc}", location
        ) from exc
