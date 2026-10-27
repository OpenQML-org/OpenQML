"""Global conversion budgets constrain built-in importers before expansion."""

import numpy as np
import pytest

from openqml import (
    Circuit,
    ConversionError,
    QuantumProgram,
    ValidationError,
    convert,
    density_matrix,
    plan_conversion,
)


@pytest.mark.parametrize("framework", ["qiskit", "cirq", "pennylane"])
@pytest.mark.parametrize(
    "source_limits,expected",
    [
        ({}, (3, 2)),
        ({"max_operations": 9, "max_depth": 7}, (3, 2)),
        ({"max_operations": 1, "max_depth": 0}, (1, 0)),
    ],
)
def test_sdk_importers_receive_the_stricter_global_and_source_limits(
    monkeypatch, framework, source_limits, expected
):
    import openqml.adapters.qiskit as qiskit_adapter
    import openqml.conversion as conversion

    seen = []

    def importer(value, source, **options):
        seen.append((value, source, options))
        return Circuit(1).h(0)

    monkeypatch.setattr(conversion, "import_native", importer)
    monkeypatch.setattr(qiskit_adapter, "static_loop_notes", lambda _: ())
    native = object()
    snapshot = dict(source_limits)
    plan = plan_conversion(
        native,
        source=framework,
        target="ir",
        source_options=source_limits,
        max_operations=3,
        max_depth=2,
    )
    assert plan.compatible and len(seen) == 1
    assert seen[0][:2] == (native, framework)
    assert seen[0][2] == dict(zip(("max_operations", "max_depth"), expected, strict=True))
    assert source_limits == snapshot


def test_captured_program_limits_and_public_convert_wrapper_are_forwarded(monkeypatch):
    import openqml.conversion as conversion

    seen = []

    def importer(value, source, **options):
        seen.append(options)
        return Circuit(1).h(0)

    monkeypatch.setattr(conversion, "import_native", importer)
    program = QuantumProgram(object(), "pennylane", (("max_operations", 5), ("max_depth", 3)))
    result = program.convert(
        "openqasm3",
        source_options={"max_depth": 1},
        max_operations=2,
        max_depth=2,
        max_output_statements=20,
    )
    assert result.target_format == "openqasm3" and seen == [{"max_operations": 2, "max_depth": 1}]
    assert program.import_options == (("max_operations", 5), ("max_depth", 3))
    with pytest.raises(ConversionError, match="QASM_EXPORT_LIMIT"):
        program.convert("openqasm3", max_output_statements=1)


def test_numpy_integer_limits_remain_json_serializable_and_reach_builtin_importers(monkeypatch):
    import openqml.conversion as conversion

    seen = []

    def importer(value, source, **options):
        seen.append(options)
        return Circuit(1).h(0)

    monkeypatch.setattr(conversion, "import_native", importer)
    plan = plan_conversion(
        object(),
        source="pennylane",
        target="ir",
        max_operations=np.int64(3),
        max_depth=np.int64(2),
        max_output_statements=np.int64(9),
        source_options={"max_operations": np.int64(7), "max_depth": np.int64(1)},
    )
    assert plan.to_json() and seen == [{"max_operations": 3, "max_depth": 1}]
    assert all(type(value) is int for value in seen[0].values())


@pytest.mark.parametrize("framework", ["cirq", "qiskit", "pennylane", "openqasm3", "json"])
@pytest.mark.parametrize(
    "name,value", [("max_operations", True), ("max_depth", -1), ("max_depth", 1.5)]
)
def test_invalid_source_limits_fail_before_any_frontend_runs(monkeypatch, framework, name, value):
    import openqml.conversion as conversion

    monkeypatch.setattr(
        conversion,
        "import_native",
        lambda *args, **kwargs: pytest.fail("Imported invalid source limits"),
    )
    with pytest.raises(ValidationError):
        plan_conversion(object(), source=framework, target="ir", source_options={name: value})


def test_custom_adapter_option_contract_is_not_silently_extended():
    from openqml import FrameworkAdapter, register_adapter, unregister_adapter

    seen = []

    def importer(value, *, custom):
        seen.append(custom)
        return Circuit(1).h(0)

    register_adapter(FrameworkAdapter("bounded-test", importer, lambda circuit: circuit))
    try:
        result = convert(
            object(),
            source="bounded-test",
            target="ir",
            source_options={"custom": "value"},
            max_operations=1,
            max_depth=0,
        )
        assert result.value == Circuit(1).h(0) and seen == ["value"]
    finally:
        unregister_adapter("bounded-test")


def test_qiskit_large_static_loops_are_limited_before_lowering_the_body(monkeypatch):
    qiskit = pytest.importorskip("qiskit")
    from qiskit.circuit import Parameter

    native = qiskit.QuantumCircuit(1)
    index = Parameter("index")
    with native.for_loop(range(1000), index):
        native.ry(index / 1000, 0)

    def fail(*args, **kwargs):
        pytest.fail("Bound a loop body before checking its global expansion limit")

    monkeypatch.setattr(Circuit, "bind", fail)
    with pytest.raises(ConversionError, match="EXPANSION_LIMIT"):
        convert(native, target="ir", max_operations=2)
    with pytest.raises(ConversionError, match="EXPANSION_LIMIT"):
        QuantumProgram.capture(native).convert("ir", max_depth=0)


@pytest.mark.parametrize("parser", ["builtin", "qiskit", "auto"])
def test_qasm_global_operation_limits_apply_to_every_parser(parser):
    if parser == "qiskit":
        pytest.importorskip("qiskit_qasm3_import")
    source = 'OPENQASM 3.0; include "stdgates.inc"; qubit q; h q; x q;'
    with pytest.raises(ConversionError, match="LIMIT"):
        convert(
            source,
            source="openqasm3",
            target="ir",
            source_options={"parser": parser},
            max_operations=1,
        )
    result = convert(
        source, source="openqasm3", target="ir", source_options={"parser": parser}, max_operations=2
    )
    assert len(result.value.operations) == 2


def test_cirq_predicate_depth_is_checked_before_constructing_decision_nodes(monkeypatch):
    cirq = pytest.importorskip("cirq")
    import openqml.adapters._cirq_control as controls
    from openqml.adapters.cirq import from_cirq

    q = cirq.LineQubit.range(3)
    native = cirq.Circuit(
        cirq.H(q[0]),
        cirq.H(q[1]),
        cirq.measure(q[0], q[1], key="bits"),
        cirq.X(q[2]).with_classical_controls("bits"),
    )
    options = {"measurement_map": {"bits": (0, 1)}}
    reference = from_cirq(native, max_depth=2, **options)
    expected = density_matrix(reference)
    original = controls.Operation

    def guarded(name, *args, **kwargs):
        if name == "if_else":
            pytest.fail("Constructed decisions before checking their depth")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(controls, "Operation", guarded)
    with pytest.raises(ConversionError, match="CONTROL_DEPTH_LIMIT"):
        convert(native, target="ir", max_depth=1, source_options=options)
    with pytest.raises(ConversionError, match="CONTROL_DEPTH_LIMIT"):
        from_cirq(native, max_depth=1, **options)
    monkeypatch.setattr(controls, "Operation", original)
    actual = convert(native, target="ir", max_depth=2, source_options=options).value
    np.testing.assert_allclose(density_matrix(actual), expected, atol=1e-12)


def test_cirq_depth_counts_constructed_nodes_instead_of_irrelevant_predicate_bits():
    cirq = pytest.importorskip("cirq")
    sp = pytest.importorskip("sympy")
    q = cirq.LineQubit.range(4)
    options = {"measurement_map": {"bits": (0, 1, 2)}}
    predicate = cirq.SympyCondition(sp.Eq(sp.Mod(sp.Symbol("bits"), 2), 1))
    # The portable predicate grammar intentionally rejects Mod, even with a budget.
    invalid = cirq.Circuit(
        cirq.measure(*q[:3], key="bits"), cirq.X(q[3]).with_classical_controls(predicate)
    )
    with pytest.raises(ConversionError, match="CLASSICAL_EXPRESSION"):
        convert(invalid, target="ir", max_depth=1, source_options=options)
    predicate = cirq.SympyCondition(sp.Ge(sp.Symbol("bits"), 4))
    native = cirq.Circuit(
        cirq.H(q[0]),
        cirq.measure(*q[:3], key="bits"),
        cirq.X(q[3]).with_classical_controls(predicate),
    )
    result = convert(native, target="ir", max_depth=1, source_options=options).value
    assert sum(op.name == "if_else" for _, op in result.walk_operations()) == 1
    with pytest.raises(ConversionError, match="CONTROL_DEPTH_LIMIT"):
        convert(native, target="ir", max_depth=0, source_options=options)


def test_pennylane_decomposition_obeys_global_source_expansion_limits():
    qml = pytest.importorskip("pennylane")
    native = qml.tape.QuantumScript([qml.AngleEmbedding([0.2, 0.3, 0.4], wires=range(3))], [])
    with pytest.raises(ConversionError, match="EXPANSION_LIMIT"):
        convert(native, target="ir", max_operations=2)
    with pytest.raises(ConversionError, match="EXPANSION_LIMIT"):
        convert(native, target="ir", max_depth=0)
    assert len(convert(native, target="ir", max_operations=3).value.operations) == 3
