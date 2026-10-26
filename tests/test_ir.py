import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from openqml import Circuit, Expression, Operation, Parameter, ValidationError


def test_immutable_builder_and_symbolic_roundtrip():
    theta, phi = Parameter("theta"), Parameter("phi")
    original = Circuit(3)
    circuit = original.h(0).ry(2 * theta - phi / 3 + 0.2, 2).cx(2, 0).measure(2, 1)
    circuit = circuit.with_global_phase(phi)
    assert original.operations == ()
    assert circuit.parameters == ("phi", "theta")
    assert Circuit.from_json(circuit.to_json()) == circuit
    assert circuit.bind({"phi": 0.3}, partial=True).parameters == ("theta",)
    assert circuit.bind({"phi": 0.3, "theta": 0.4}).operations[1].params[
        0
    ].constant == pytest.approx(0.9)
    with pytest.raises(FrozenInstanceError):
        circuit.num_qubits = 5


@pytest.mark.parametrize(
    "function",
    [
        lambda: Circuit(0),
        lambda: Circuit(True),
        lambda: Circuit(2).cx(0, 0),
        lambda: Circuit(2).x(2),
        lambda: Circuit(2).measure(-1),
        lambda: Operation("reset", (0,), params=(0.1,)),
        lambda: Operation("rx", (0,)),
        lambda: Parameter(""),
        lambda: Parameter("bad\nname"),
        lambda: Expression(op="__import__", args=()),
        lambda: Expression(constant=np.nan),
        lambda: Circuit(1).ry(np.inf, 0),
        lambda: Circuit(1).ry(True, 0),
        lambda: Circuit(1).ry(Parameter("t"), 0).bind({"extra": 1}),
    ],
)
def test_invalid_inputs_are_rejected(function):
    with pytest.raises(ValidationError):
        function()


@pytest.mark.parametrize(
    "field,value",
    [("version", 99), ("version", True), ("bit_order", "little"), ("num_qubits", -1), ("extra", 1)],
)
def test_json_schema_is_strict(field, value):
    payload = Circuit(1).to_dict()
    payload[field] = value
    with pytest.raises(ValidationError):
        Circuit.from_json(json.dumps(payload))


def test_expression_canonicalization_and_binding_errors():
    a = Parameter("a")
    assert (a - a).parameters == frozenset()
    assert Expression((("a", 2), ("a", -1))) == a + 0
    with pytest.raises(ValidationError):
        (a + 1).evaluate({})
    with pytest.raises(ValidationError):
        a / 0


def test_duplicate_json_fields_rejected():
    text = Circuit(1).to_json().replace('"version": 4', '"version": 9, "version": 4')
    with pytest.raises(ValidationError, match="Duplicate JSON field"):
        Circuit.from_json(text)


@pytest.mark.parametrize("field", ["qubits", "clbits", "params"])
def test_json_operands_cannot_be_coerced_from_empty_objects(field):
    payload = Circuit(1).h(0).to_dict()
    payload["operations"][0][field] = {}
    with pytest.raises(ValidationError, match="JSON arrays"):
        Circuit.from_dict(payload)


def test_json_operations_must_be_an_array():
    payload = Circuit(1).to_dict()
    payload["operations"] = {}
    with pytest.raises(ValidationError, match="JSON array"):
        Circuit.from_dict(payload)
