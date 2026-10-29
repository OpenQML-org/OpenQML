"""Binding and conversion share validated immutable payloads without weakening checks."""

import numpy as np
import pytest

from openqml import (
    Circuit,
    ClassicalCondition,
    LayerSpec,
    Operation,
    Parameter,
    PauliString,
    ValidationError,
    density_matrix,
    expectations_batch,
    plan_conversion,
    statevector,
)
from openqml.compiler import rename_parameters
from openqml.parameters import Expression


def matrix_program():
    x, w = Parameter("feature"), Parameter("weight")
    return (
        Circuit(2)
        .unitary(np.array([[0, 1j], [1j, 0]]), 0)
        .channel([np.eye(2)], 1)
        .ry(x, 1)
        .measure(1, 2)
        .if_else(
            ClassicalCondition((2,), 1),
            Circuit(2)
            .unitary(np.diag([1j, -1j]), 1)
            .channel([np.eye(2)], 0)
            .rx(w, 0)
            .with_global_phase(w / 7),
        )
        .with_global_phase(x / 3)
    )


def reject_matrix_revalidation(monkeypatch):
    original = Operation.__post_init__

    def guarded(operation):
        if operation.name in {"unitary", "kraus"}:
            pytest.fail("Revalidated unchanged immutable matrix data")
        return original(operation)

    monkeypatch.setattr(Operation, "__post_init__", guarded)


def test_nested_binding_and_renaming_share_matrix_payloads_and_preserve_density(monkeypatch):
    source = matrix_program()
    snapshot = source.to_json()
    expected = density_matrix(source, {"feature": 0.4, "weight": -0.2})
    with monkeypatch.context() as local:
        reject_matrix_revalidation(local)
        bound = source.bind({"feature": 0.4, "weight": -0.2})
        renamed = rename_parameters(source, {"feature": "x", "weight": "w", "unused": "ignored"})
        partial = source.bind({"feature": 0.4}, partial=True)
        for target in (bound, renamed, partial):
            for path, operation in source.walk_operations():
                if operation.name in {"unitary", "kraus"}:
                    assert target.operation_at(path) is operation
        assert source.bind({}, partial=True) is source
    np.testing.assert_allclose(density_matrix(bound), expected, atol=1e-12)
    np.testing.assert_allclose(density_matrix(renamed, {"x": 0.4, "w": -0.2}), expected, atol=1e-12)
    np.testing.assert_allclose(density_matrix(partial, {"weight": -0.2}), expected, atol=1e-12)
    assert source.to_json() == snapshot


def test_vectorized_batches_reuse_constant_matrices_and_keep_physical_values(monkeypatch):
    matrix = np.array([[0, 1j], [1j, 0]])
    source = Circuit(2).unitary(matrix, 0).ry(Parameter("weight"), 1)
    matrix[:] = 0
    rows = [{"weight": x} for x in (0.2, 0.4, -0.3)]
    with monkeypatch.context() as local:
        reject_matrix_revalidation(local)
        result = expectations_batch(source, [PauliString("ZI"), PauliString("IZ")], rows)
    np.testing.assert_allclose(result, [[-1, np.cos(row["weight"])] for row in rows], atol=1e-12)
    assert source.operations[0].matrix[0][1] == 1j


def test_conversion_binding_and_parameter_aliasing_reuse_payloads_with_full_phase(monkeypatch):
    x, w = Parameter("input"), Parameter("weight")
    source = (
        Circuit(2).unitary(np.diag([1, 1j, -1j, -1]), 1, 0).ry(x + w, 0).with_global_phase(x / 3)
    )
    values = {"input": 0.3, "weight": -0.2}
    expected = statevector(source, values)
    with monkeypatch.context() as local:
        reject_matrix_revalidation(local)
        bound = plan_conversion(source, target="ir", bindings=values).export().value
        plan = plan_conversion(source, target="openqasm3")
        assert bound.operations[0] is source.operations[0]
        assert plan.prepared_circuit.operations[0] is source.operations[0]
        assert "input" in plan.parameter_map
    np.testing.assert_allclose(statevector(bound), expected, atol=1e-12)
    mapped = {plan.parameter_map.get(name, name): value for name, value in values.items()}
    np.testing.assert_allclose(statevector(plan.prepared_circuit, mapped), expected, atol=1e-12)


@pytest.mark.parametrize(
    "bindings",
    [
        None,
        [],
        {1: 0.2, "weight": 0.3},
        {"weight": True},
        {"weight": np.inf},
        {"weight": [0.3]},
        {"missing": 0.3},
        {},
    ],
)
def test_binding_still_rejects_invalid_or_mismatched_values(bindings):
    with pytest.raises(ValidationError):
        Circuit(1).ry(Parameter("weight"), 0).bind(bindings)


def test_changed_channels_and_new_payloads_are_validated():
    circuit = Circuit(1).bit_flip(Parameter("noise"), 0)
    with pytest.raises(ValidationError, match="probability"):
        circuit.bind({"noise": 1.2})
    with pytest.raises(ValidationError, match="unitary"):
        Circuit(1).unitary([[1, 1], [0, 1]], 0)
    with pytest.raises(ValidationError, match="trace-preserving"):
        Circuit(1).channel([np.eye(2) * 2], 0)


def test_numeric_binding_preserves_affine_accumulation_and_source_interfaces():
    angle = Expression((("a", 1.0), ("b", 1.0)), 1e16)
    circuit = Circuit(1).ry(angle, 0).with_global_phase(angle)
    bound = circuit.bind({"a": -1e16, "b": 1.0})
    assert bound.operations[0].params[0].constant == 0.0 and bound.global_phase.constant == 0.0
    np.testing.assert_allclose(statevector(bound), [1, 0], atol=1e-12)
    spec = LayerSpec(Circuit(1).ry(Parameter("x") + Parameter("w"), 0), ("x",), ("w",))
    assert spec.circuit.bind({"x": 0.2}, partial=True).parameters == ("w",)
    assert spec.input_parameters == ("x",) and spec.weight_parameters == ("w",)
