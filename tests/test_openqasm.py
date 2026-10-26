import numpy as np
import pytest

from openqml import Circuit, ConversionError, Parameter, convert, diagnose, statevector
from openqml.adapters.openqasm import from_openqasm, to_openqasm


@pytest.mark.parametrize("version", [2, 3])
def test_qasm_roundtrip_all_gates(version):
    circuit = (
        Circuit(3)
        .id(0)
        .h(2)
        .x(1)
        .y(0)
        .z(1)
        .s(0)
        .sdg(2)
        .t(1)
        .tdg(0)
        .sx(1)
        .rx(0.2, 0)
        .ry(0.3, 1)
        .rz(-0.7, 2)
        .p(0.4, 0)
        .cx(2, 0)
        .cy(1, 2)
        .cz(0, 1)
        .swap(2, 1)
        .ccx(2, 0, 1)
        .rxx(0.3, 0, 2)
        .ryy(-0.41, 2, 1)
        .rzz(0.23, 1, 0)
        .barrier()
    )
    restored = from_openqasm(to_openqasm(circuit, version=version))
    np.testing.assert_allclose(statevector(restored), statevector(circuit), atol=1e-12)
    measured = from_openqasm(to_openqasm(circuit.measure(2, 1), version=version))
    assert measured.operations[-1] == circuit.measure(2, 1).operations[-1]


def test_symbolic_qasm3_and_diagnostics():
    t = Parameter("theta")
    circuit = Circuit(2).ry(t * 2 + 0.4, 1).with_global_phase(t / 3)
    restored = from_openqasm(to_openqasm(circuit))
    assert restored == circuit
    with pytest.raises(ConversionError) as exc:
        to_openqasm(circuit, version=2)
    assert {d.code for d in exc.value.diagnostics} == {
        "UNBOUND_PARAMETER",
        "GLOBAL_PHASE_UNSUPPORTED",
    }
    assert (
        diagnose(Circuit(1).ry(Parameter("input"), 0), "openqasm3")[0].code == "RESERVED_IDENTIFIER"
    )


def test_multiple_registers_and_broadcast():
    text = (
        'OPENQASM 2.0; include "qelib1.inc"; qreg a[2]; qreg b[2]; '
        "creg c[2]; h a; cx a,b; measure b -> c;"
    )
    circuit = from_openqasm(text)
    assert circuit.num_qubits == 4
    assert [op.qubits for op in circuit.operations[2:4]] == [(0, 2), (1, 3)]
    assert circuit.num_clbits == 2


@pytest.mark.parametrize(
    "body",
    [
        'include "../../secret.inc";',
        "delay[10ns] q[0];",
        "if (1) x q[0];",
        "gate custom a { x a; };",
        'rx(__import__("os")) q[0];',
        "rx(1/0) q[0];",
        "rx(1e999) q[0];",
        "x q[1];",
        "rx(foo) q[0];",
        "x q[0]",
        "/* unterminated",
    ],
)
def test_unsupported_or_malicious_qasm_is_rejected(body):
    with pytest.raises(ConversionError):
        from_openqasm("OPENQASM 3.0; qubit[1] q; " + body)


def test_parser_limits_and_format_contract():
    with pytest.raises(ConversionError):
        from_openqasm("OPENQASM 3.0; qubit[999999] q;")
    with pytest.raises(ConversionError):
        from_openqasm("OPENQASM 3.0; qubit[5] q; h q;", max_operations=2)
    with pytest.raises(ConversionError):
        convert("OPENQASM 3.0; qubit q;", source="openqasm2", target="ir")
    result = convert(Circuit(2).h(0), source="ir", target="openqasm3")
    assert from_openqasm(result.value) == result.circuit


def test_unused_input_is_diagnosed_and_phase_whitespace_parses():
    with pytest.raises(ConversionError, match="UNUSED_INPUT_DECLARATION"):
        from_openqasm("OPENQASM 3.0; input float theta; qubit q;")
    circuit = from_openqasm("OPENQASM 3.0; qubit q; gphase (pi/3);")
    np.testing.assert_allclose(statevector(circuit), [np.exp(1j * np.pi / 3), 0])


@pytest.mark.parametrize("name", ["x", "y", "p", "h", "phase", "id", "rx", "U", "u3", "CX"])
def test_qasm3_standard_library_input_collisions_have_explicit_aliases(name):
    from openqml import Parameter, convert, diagnose, statevector

    original = Circuit(1).ry(Parameter(name), 0)
    assert "RESERVED_IDENTIFIER" in {d.code for d in diagnose(original, "openqasm3")}
    result = convert(original, target="openqasm3")
    alias = result.parameter_map[name]
    assert alias != name and f"input float[64] {name};" not in result.value
    restored = convert(
        result.value, source="openqasm3", target="ir", source_options={"parser": "builtin"}
    ).value
    np.testing.assert_allclose(
        statevector(restored, {alias: 0.7}), statevector(original, {name: 0.7})
    )


def test_qasm3_gate_name_aliases_parse_with_official_sdk_and_avoid_existing_names():
    pytest.importorskip("qiskit_qasm3_import")
    from openqml import Parameter, QuantumProgram, convert, statevector

    c = Circuit(1).ry(Parameter("x") + Parameter("parameter_1"), 0).rz(Parameter("p"), 0)
    result = convert(c, target="openqasm3")
    assert not (set(result.parameter_map.values()) & set(c.parameters))
    restored = QuantumProgram.from_qasm(result.value).to_ir()
    values = {"x": 0.3, "parameter_1": 0.2, "p": 0.1}
    renamed = {result.parameter_map.get(name, name): v for name, v in values.items()}
    np.testing.assert_allclose(statevector(restored, renamed), statevector(c, values), atol=1e-13)
