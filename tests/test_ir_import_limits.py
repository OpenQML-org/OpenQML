"""Serialized circuit budgets apply before IR object or matrix construction."""

import json

import numpy as np
import pytest

from openqml import (
    Circuit,
    ClassicalCondition,
    ConversionError,
    Parameter,
    ValidationError,
    plan_conversion,
)


def no_construction(monkeypatch):
    import openqml.ir as ir

    def fail(*args, **kwargs):
        raise AssertionError("IR construction started before preflight finished")

    monkeypatch.setattr(ir, "Operation", fail)


@pytest.mark.parametrize("encoded", [False, True])
def test_nested_operation_and_depth_limits_precede_construction(monkeypatch, encoded):
    circuit = Circuit(1).x(0).if_else(ClassicalCondition((0,), 0), Circuit(1).ry(0.3, 0))
    value = circuit.to_json() if encoded else circuit.to_dict()
    load = Circuit.from_json if encoded else Circuit.from_dict
    no_construction(monkeypatch)
    for options, location in (
        ({"max_operations": 2}, "operations[1].blocks[0].operations[0]"),
        ({"max_depth": 0}, "operations[1].blocks["),
    ):
        with pytest.raises(ValidationError) as exc:
            load(value, **options)
        assert location in str(exc.value)


def test_shared_matrix_entry_and_width_budgets_precede_decoding(monkeypatch):
    identity = np.eye(2)
    circuit = Circuit(1).unitary(identity, 0).channel([identity / np.sqrt(2)] * 2, 0)
    payload = circuit.to_dict()
    no_construction(monkeypatch)
    for options, label in (
        ({"max_matrix_entries": 11}, "max_matrix_entries"),
        ({"max_matrix_qubits": 0}, "max_matrix_qubits"),
    ):
        with pytest.raises(ValidationError, match=label):
            Circuit.from_dict(payload, **options)
    payload["operations"][0]["qubits"] = list(range(100_000))
    with pytest.raises(ValidationError, match="max_matrix_entries"):
        Circuit.from_dict(payload, max_matrix_qubits=100_000)


def test_matrix_and_kraus_boundaries_preserve_density_semantics():
    from openqml import density_matrix

    identity = np.eye(2)
    original = Circuit(1).ry(0.4, 0).unitary(identity, 0).channel([identity / np.sqrt(2)] * 2, 0)
    restored = Circuit.from_json(original.to_json(), max_matrix_qubits=1, max_matrix_entries=12)
    assert restored == original
    np.testing.assert_allclose(density_matrix(restored), density_matrix(original))
    assert Circuit.from_dict(
        Circuit(1).to_dict(), max_operations=0, max_matrix_entries=0
    ) == Circuit(1)


def test_expression_depth_node_and_cycle_budgets_precede_construction(monkeypatch):
    payload = Circuit(1).ry(Parameter("theta"), 0).to_dict()
    expression = payload["operations"][0]["params"][0]
    deep = expression
    for _ in range(65):
        deep = {"op": "sin", "args": [deep]}
    no_construction(monkeypatch)
    for graph in (deep, {"constant": 0, "terms": {f"p_{i}": 1 for i in range(4096)}}):
        payload["operations"][0]["params"] = [graph]
        with pytest.raises(ValidationError, match="expression.*budget"):
            Circuit.from_dict(payload)
    cycle = {"op": "sin", "args": []}
    cycle["args"].append(cycle)
    payload["operations"][0]["params"] = [cycle]
    with pytest.raises(ValidationError, match="expression.*budget"):
        Circuit.from_dict(payload)


def test_control_cycle_is_bounded_without_recursion_errors(monkeypatch):
    payload = Circuit(1).if_else(ClassicalCondition((0,), 0), Circuit(1).x(0)).to_dict()
    payload["operations"][0]["blocks"][0] = payload
    no_construction(monkeypatch)
    with pytest.raises(ValidationError, match="max_depth"):
        Circuit.from_dict(payload, max_depth=3)


def test_json_byte_budget_counts_utf8_and_rejects_invalid_text():
    payload = Circuit(1).ry(Parameter("\u03b8"), 0).to_dict()
    raw = json.dumps(payload, ensure_ascii=False)
    with pytest.raises(ValidationError, match="max_bytes"):
        Circuit.from_json(raw, max_bytes=len(raw))
    assert Circuit.from_json(raw, max_bytes=len(raw.encode())).parameters == ("\u03b8",)
    for value in (
        b"{}",
        None,
        raw + "\ud800",
        "[",
        '{"num_qubits":' + "9" * 5000 + "}",
        raw.replace('"num_qubits": 1', '"num_qubits": 1, "num_qubits": 1'),
    ):
        with pytest.raises(ValidationError):
            Circuit.from_json(value)


def test_conversion_propagates_source_and_global_limits_before_construction(monkeypatch):
    raw = Circuit(1).h(0).ry(0.4, 0).to_json()
    no_construction(monkeypatch)
    with pytest.raises(ValidationError, match="max_operations"):
        plan_conversion(
            raw,
            source="json",
            target="openqasm3",
            max_operations=1,
            source_options={"max_operations": 10},
        )
    with pytest.raises(ValidationError, match="max_bytes"):
        plan_conversion(raw, source="json", target="ir", source_options={"max_source_bytes": 1})
    with pytest.raises(ConversionError, match="IR_FRONTEND"):
        plan_conversion(raw, source="json", target="ir", source_options={"unexpected": 1})


@pytest.mark.parametrize(
    "label", ["max_operations", "max_depth", "max_matrix_qubits", "max_matrix_entries", "max_bytes"]
)
@pytest.mark.parametrize("value", [True, -1, 1.5])
def test_ir_budget_options_require_nonnegative_integers(label, value):
    with pytest.raises(ValidationError):
        Circuit.from_json(Circuit(1).to_json(), **{label: value})


def test_ir_import_options_flow_through_layer_capture():
    from openqml import LayerSpec

    raw = Circuit(1).ry(Parameter("feature"), 0).to_json()
    spec = LayerSpec.from_program(
        raw, ["feature"], framework="json", max_source_bytes=len(raw.encode())
    )
    assert spec.input_parameters == ("feature",)
    with pytest.raises(ValidationError, match="max_bytes"):
        LayerSpec.from_program(raw, ["feature"], framework="json", max_source_bytes=1)


def test_portable_layer_and_checkpoint_readers_share_matrix_budgets():
    from openqml import LayerCheckpoint, LayerSpec

    spec = LayerSpec(Circuit(1).unitary(np.eye(2), 0), (), ())
    checkpoint = LayerCheckpoint(spec, ())
    for model in (spec, checkpoint):
        assert type(model).from_json(model.to_json(), max_matrix_entries=4) == model
        with pytest.raises(ValidationError, match="max_matrix_entries"):
            type(model).from_json(model.to_json(), max_matrix_entries=3)
        with pytest.raises(ValidationError, match="max_matrix_qubits"):
            type(model).from_dict(model.to_dict(), max_matrix_qubits=0)


def test_ir_budgets_accept_numpy_integer_configuration():
    circuit = Circuit(1).unitary(np.eye(2), 0)
    assert (
        Circuit.from_json(
            circuit.to_json(),
            max_operations=np.int64(1),
            max_depth=np.int64(0),
            max_matrix_qubits=np.int64(1),
            max_matrix_entries=np.int64(4),
            max_bytes=np.int64(1000),
        )
        == circuit
    )
