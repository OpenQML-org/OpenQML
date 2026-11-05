"""Source-name binding, topology reuse and ordered sweep export contracts."""

import numpy as np
import pytest

import openqml as oq


@pytest.mark.parametrize(
    "target", ["ir", "json", "openqasm2", "openqasm3", "qiskit", "cirq", "pennylane"]
)
def test_sweep_roundtrip_preserves_each_binding_and_phase(target):
    if target in {"qiskit", "cirq", "pennylane"}:
        pytest.importorskip(target)
    theta = oq.Parameter("theta")
    circuit = oq.Circuit(2).h(0).ry(theta, 1).cx(0, 1)
    plan = oq.plan_conversion(circuit, target=target)
    rows = [{"theta": angle} for angle in (0.0, 0.2, -0.7)]
    results = plan.export_many(rows)
    assert len(results) == 3
    for row, result in zip(rows, results, strict=True):
        restored = oq.convert(result.value, source=target, target="ir").value
        np.testing.assert_allclose(
            oq.statevector(restored), oq.statevector(circuit.bind(row)), atol=1e-10
        )
        assert result.circuit == circuit.bind(row)
    assert plan.circuit is circuit and circuit.parameters == ("theta",)


def test_reserved_names_partial_bindings_and_lowering_reuse(monkeypatch):
    from openqml import compiler

    a, b = oq.Parameter("input"), oq.Parameter("second angle")
    circuit = oq.Circuit(2).crx(a + b, 0, 1).with_global_phase(a / 2)
    plan = oq.plan_conversion(circuit, target="openqasm3")
    assert plan.parameter_map

    def repeated(*args, **kwargs):
        pytest.fail("Compatible binding repeated structural lowering")

    monkeypatch.setattr(compiler, "decompose", repeated)
    bound = plan.bind({"input": 0.4}, partial=True)
    assert bound.circuit.parameters == ("second angle",)
    final = bound.bind({"second angle": 0.3})
    assert not final.prepared_circuit.parameters and final.compatible
    monkeypatch.undo()
    np.testing.assert_allclose(
        oq.statevector(final.prepared_circuit),
        oq.statevector(circuit, {"input": 0.4, "second angle": 0.3}),
        atol=1e-10,
    )


def test_binding_rechecks_target_and_preserves_permanent_incompatibilities():
    theta = oq.Parameter("theta")
    plan = oq.plan_conversion(oq.Circuit(1).ry(theta, 0), target="openqasm2")
    assert not plan.compatible and plan.bind({"theta": 0.4}).compatible
    phased = oq.plan_conversion(oq.Circuit(1).with_global_phase(theta), target="openqasm2")
    assert phased.bind({"theta": 0}).compatible
    with pytest.raises(oq.ConversionError):
        phased.bind({"theta": 0.4}).export()
    channel = oq.plan_conversion(oq.Circuit(1).bit_flip(theta, 0), target="openqasm3")
    with pytest.raises(oq.ConversionError):
        channel.export_many([{"theta": 0.2}])


def test_stream_is_lazy_ordered_bounded_and_uses_original_source_once():
    calls = []
    name = "sweep_test"
    adapter = oq.FrameworkAdapter(
        name,
        lambda value: calls.append(value) or oq.Circuit(1).ry(oq.Parameter("theta"), 0),
        lambda value: value,
    )
    oq.register_adapter(adapter)
    try:
        plan = oq.plan_conversion("native", source=name, target="ir")
        consumed = []

        def rows():
            for angle in range(4):
                consumed.append(angle)
                yield {"theta": angle}

        stream = plan.iter_exports(rows(), max_bindings=2)
        assert not consumed
        assert next(stream).source_format == name
        assert next(stream).value == oq.Circuit(1).ry(1, 0)
        with pytest.raises(oq.ResourceError, match="max_bindings"):
            next(stream)
        assert calls == ["native"] and consumed == [0, 1, 2]
    finally:
        oq.unregister_adapter(name)


@pytest.mark.parametrize("rows", [{"theta": 1}, "text", None, [{"wrong": 0}], [{}]])
def test_sweep_validates_input_binding_names(rows):
    plan = oq.plan_conversion(oq.Circuit(1).ry(oq.Parameter("theta"), 0), target="ir")
    with pytest.raises(oq.ValidationError):
        plan.export_many(rows)
