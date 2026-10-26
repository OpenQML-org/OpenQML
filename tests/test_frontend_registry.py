from dataclasses import dataclass

import pytest

from openqml import (
    Circuit,
    ConversionError,
    Diagnostic,
    FrameworkAdapter,
    Parameter,
    QuantumProgram,
    ValidationError,
    convert,
    diagnose,
    list_formats,
    register_adapter,
    unregister_adapter,
)


@dataclass
class External:
    data: dict


def test_custom_ecosystem_import_export_auto_detection_and_native_program():
    adapter = FrameworkAdapter(
        "external",
        lambda v: Circuit.from_dict(v.data),
        lambda c: External(c.to_dict()),
        lambda v: isinstance(v, External),
    )
    register_adapter(adapter)
    try:
        circuit = Circuit(2).ry(Parameter("x").sin(), 1).cx(1, 0).with_global_phase(0.37)
        assert "external" in list_formats()
        native = convert(circuit, target="external").value
        assert convert(native, target="ir").circuit == circuit
        assert QuantumProgram.capture(native).to_ir() == circuit
        assert diagnose(circuit, "external") == ()
        bound_native = convert(circuit.bind({"x": 0.41}), target="external").value
        assert (
            QuantumProgram.capture(bound_native).run(shots=17, seed=1, backend="numpy") is not None
        )
    finally:
        unregister_adapter("external")


def test_custom_adapter_diagnostics_contract_and_name_protection():
    diagnostic = Diagnostic("EXTERNAL_FEATURE", "This target cannot retain symbolic parameters.")
    adapter = FrameworkAdapter(
        "numeric-only",
        lambda v: v,
        lambda c: c,
        diagnose=lambda c: (diagnostic,) if c.parameters else (),
    )
    register_adapter(adapter)
    try:
        with pytest.raises(ConversionError, match="EXTERNAL_FEATURE"):
            convert(Circuit(1).ry(Parameter("x"), 0), target="numeric-only")
        with pytest.raises(ConversionError, match="ADAPTER_CONTRACT"):
            convert("wrong", source="numeric-only", target="ir")
        with pytest.raises(ValidationError, match="already registered"):
            register_adapter(adapter)
    finally:
        unregister_adapter(adapter.name)
    with pytest.raises(ValidationError, match="protected"):
        register_adapter(FrameworkAdapter("qiskit", lambda v: v, lambda c: c))
