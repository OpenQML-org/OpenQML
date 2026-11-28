"""Core-only OpenQASM feedback, source limits and exact atomic block semantics."""

import subprocess
import sys

import numpy as np
import pytest

from openqml import (
    Circuit,
    ClassicalCondition,
    ConversionError,
    Parameter,
    convert,
    density_matrix,
)


@pytest.mark.parametrize("value", range(4))
def test_core_qasm3_register_equality_atomic_overwrite_and_phase(value):
    c = (
        Circuit(3)
        .h(0)
        .h(1)
        .measure(0, 0)
        .measure(1, 1)
        .if_else(
            ClassicalCondition((0, 1), value),
            Circuit(3).x(0).measure(0, 0).ry(Parameter("theta") ** 2, 2).with_global_phase(0.3),
            Circuit(3).rx(Parameter("theta").sin(), 2),
        )
        .measure(2, 2)
    )
    text = convert(c, target="openqasm3").value
    restored = convert(
        text, source="openqasm3", target="ir", source_options={"parser": "builtin"}
    ).value
    values = {"theta": 0.6}
    np.testing.assert_allclose(
        density_matrix(restored, values), density_matrix(c, values), atol=2e-13
    )
    from openqml.density import _simulate

    states, _, _ = _simulate(restored, values, retain_classical=True)
    distribution = {
        "".join(str(int(bit in ones)) for bit in range(3)): np.trace(rho.reshape(8, 8)).real
        for ones, rho in states.items()
    }
    expected = {}
    for x in (0, 1):
        for y in (0, 1):
            selected = x + 2 * y == value
            final_x = 1 - x if selected else x
            p1 = np.sin((0.6**2 if selected else np.sin(0.6)) / 2) ** 2
            for z, prob in [(0, 1 - p1), (1, p1)]:
                key = f"{final_x}{y}{z}"
                expected[key] = expected.get(key, 0) + prob / 4
    for key in expected:
        assert distribution.get(key, 0) == pytest.approx(expected[key], abs=2e-13)
    # A direct register guard has LSB-first equality semantics as in Qiskit.
    direct = (
        "OPENQASM 3; qubit[3] q; bit[2] c; h q[0]; h q[1]; "
        f"c[0]=measure q[0]; c[1]=measure q[1]; if(c == {value}) {{ x q[2]; }}"
    )
    ir = convert(
        direct, source="openqasm3", target="ir", source_options={"parser": "builtin"}
    ).value
    expected = (
        Circuit(3)
        .h(0)
        .h(1)
        .measure(0, 0)
        .measure(1, 1)
        .if_else(ClassicalCondition((0, 1), value), Circuit(3).x(2))
    )
    np.testing.assert_allclose(density_matrix(ir), density_matrix(expected), atol=1e-13)


def test_nested_bare_bit_empty_else_phase_only_and_later_global_wires():
    source = """OPENQASM 3.0;
    qubit q; bit c;
    h q; c = measure q;
    if (c) {
      if(c == false) {} else { gphase(0.3); x q; }
    } else {}
    qubit later; bit after;
    after = measure later;
    """
    ir = convert(
        source, source="openqasm3", target="ir", source_options={"parser": "builtin"}
    ).value
    assert (ir.num_qubits, ir.num_clbits) == (2, 2)
    np.testing.assert_allclose(density_matrix(ir), np.diag([1, 0, 0, 0]), atol=1e-13)
    for _, op in ir.walk_operations():
        assert all(b.num_qubits == 2 and b.num_clbits == 2 for b in op.blocks)


def test_compact_guards_preserve_sparse_bit_order_and_atomic_writes_with_native_parser():
    from openqml.adapters.openqasm import from_openqasm, to_openqasm

    circuit = (
        Circuit(3)
        .h(0)
        .h(1)
        .measure(0, 2)
        .measure(1, 0)
        .if_else(
            ClassicalCondition((2, 0), 2),
            Circuit(3).x(0).measure(0, 2).y(2),
            Circuit(3).h(2),
        )
    )
    text = to_openqasm(circuit)
    assert "c[{2, 0}] == 2" in text
    assert text.count("h q[2]") == 1
    np.testing.assert_allclose(density_matrix(from_openqasm(text)), density_matrix(circuit))
    qasm3 = pytest.importorskip("qiskit.qasm3")
    from openqml.adapters.qiskit import from_qiskit

    np.testing.assert_allclose(
        density_matrix(from_qiskit(qasm3.loads(text))), density_matrix(circuit), atol=2e-14
    )


@pytest.mark.parametrize(
    "guard,code",
    [
        ("(c == true) && (c == false)", "CLASSICAL_CONDITION"),
        ("c || c", "UNSUPPORTED_STATEMENT"),
        ("c &&", "SYNTAX"),
        ("(c && c", "SYNTAX"),
        ("c) && c", "UNSUPPORTED_STATEMENT"),
    ],
)
def test_composite_guard_rejects_conflicts_and_malformed_grouping(guard, code):
    from openqml.adapters.openqasm import from_openqasm

    with pytest.raises(ConversionError, match=code):
        from_openqasm(f"OPENQASM 3; qubit q; bit c; if ({guard}) {{ x q; }}")


def test_consistent_nested_guard_terms_merge_before_branch_import():
    from openqml.adapters.openqasm import from_openqasm

    ir = from_openqasm(
        "OPENQASM 3; qubit q; bit[3] c; "
        "if (((c[2] == false) && (c[0] == true)) && c[0] == true) { x q; }"
    )
    assert ir.operations[0].condition == ClassicalCondition((2, 0), 2)


@pytest.mark.parametrize(
    "tail,code",
    [
        ("if(c) { x q;", "SYNTAX"),
        ("if(c) { x q }", "SYNTAX"),
        ("} else {}", "SYNTAX"),
        ("if(c) { bit local; }", "BLOCK_DECLARATION"),
        ("if(c == 2) {}", "CLASSICAL_CONDITION"),
        ("if(c) {} else { qubit local; }", "BLOCK_DECLARATION"),
    ],
)
def test_conditional_syntax_errors_have_structured_diagnostics(tail, code):
    with pytest.raises(ConversionError, match=code):
        convert(
            "OPENQASM 3; qubit q; bit c; " + tail,
            source="openqasm3",
            target="ir",
            source_options={"parser": "builtin"},
        )


def test_global_budget_counts_empty_decisions_phases_and_nested_gates():
    prefix = "OPENQASM 3; qubit q; bit c; "
    cases = [("if(c) {} if(c) {}", 1), ("if(c) { x q; }", 1), ("gphase(.1); gphase(.2);", 1)]
    for code, limit in cases:
        with pytest.raises(ConversionError, match="OPERATION_LIMIT"):
            convert(
                prefix + code,
                source="openqasm3",
                target="ir",
                source_options={"parser": "builtin", "max_operations": limit},
            )
    with pytest.raises(ConversionError, match="CONTROL_DEPTH_LIMIT"):
        convert(
            prefix + "if(c) { if(c) {} }",
            source="openqasm3",
            target="ir",
            source_options={"parser": "builtin", "max_depth": 1},
        )


def test_original_lines_survive_multiline_comments_and_blocks_without_sdk_import():
    source = "OPENQASM 3;\nqubit q; bit c;\n/* a\nb */\nif(c) {\nunknown q;\n}\n"
    with pytest.raises(ConversionError) as error:
        convert(source, source="openqasm3", target="ir", source_options={"parser": "builtin"})
    assert error.value.diagnostics[0].location == "line 6"
    script = """import sys
from openqml import convert
source='OPENQASM 3; qubit q; bit c; h q; c=measure q; if(c) { gphase(.3); x q; }'
result=convert(source,source='openqasm3',target='ir',source_options={'parser':'builtin'})
assert result.value.has_measurements
assert not any(name in sys.modules for name in ['qiskit','cirq','pennylane','torch','jax'])
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("selector,expected", [(0, "01"), (1, "10"), (2, "01"), (3, "11")])
def test_nested_switch_evaluates_one_arm_and_exits_nearest_loop(selector, expected):
    import openqml as oq

    src = f"""OPENQASM 3.1; include "stdgates.inc";
    qubit[2] q; bit[2] selector; bit[2] out; uint[2] i=0;
    selector={selector};
    while(i<2) {{
        i+=1;
        switch(selector) {{
            case 0,2 {{ selector=1; continue; }}
            case 1 {{ switch(i) {{ case 1 {{ x q[0]; }} default {{ x q[1]; }} }} break; }}
            default {{ x q[0]; x q[1]; break; }}
        }}
        x q[0];
    }}
    out=measure q;"""
    plan = oq.plan_conversion(src, target="openqasm3")
    restored = oq.convert(plan.export().value, target="ir").value
    assert oq.sample(restored, shots=23, classical_bits=(2, 3)) == {expected: 23}
    assert plan.verify().matched


def test_measurement_selects_correlated_quantum_and_classical_addresses():
    import openqml as oq

    source = """OPENQASM 3.0; include "stdgates.inc";
    qubit a; qubit[2] q; bit address; bit[2] record;
    h a; address=measure a; x q[uint(address)]; record[uint(address)]=measure q[uint(address)];"""
    plan = oq.plan_conversion(source, target="openqasm3")
    counts = oq.sample(plan.circuit, shots=512, seed=42)
    assert set(counts) == {"010", "101"} and sum(counts.values()) == 512
    assert plan.verify().matched
    with pytest.raises(ConversionError, match="INVALID_OPERATION"):
        convert(
            source.replace("x q[uint(address)]", "cx q[uint(address)], q[uint(address)]"),
            target="ir",
        )


def test_index_guard_expires_after_writes_and_loop_back_edges():
    import openqml as oq

    header = 'OPENQASM 3.0; include "stdgates.inc"; qubit[2] q; bit[2] out; uint[2] i=0; '
    good = header + "while(i<2){x q[i]; i=i+1;} out=measure q;"
    plan = oq.plan_conversion(good, target="json", optimize=True)
    restored = convert(plan.export().value, source="json", target="ir").value
    assert oq.sample(restored, shots=19, classical_bits=(0, 1)) == {"11": 19}
    assert plan.verify().matched
    for body in ("while(i<2){i+=1; x q[i];}", "if(i<2){while(true){x q[i]; i+=1;}}"):
        checked = convert(header + body, target="ir", source_options={"parser": "builtin"}).value
        assert checked.to_dict()["version"] == 7
        with pytest.raises(oq.ValidationError, match="CLASSICAL_INDEX_RANGE"):
            oq.sample(checked, shots=8)
    for options in ({"max_depth": 1}, {"max_operations": 4}):
        with pytest.raises(ConversionError, match="LIMIT|DEPTH"):
            convert(good, target="ir", **options)


def test_64_bit_unsigned_counter_wraps_and_drives_qubit_index():
    import openqml as oq

    src = """OPENQASM 3.0; include "stdgates.inc"; qubit[2] q; bit[2] out;
    uint[64] i=18446744073709551615; i+=1;
    while(i<2){x q[i]; i+=1;} i-=2; if(i==0){x q[0];} out=measure q;"""
    circuit = convert(src, target="ir", max_depth=8).value
    assert sum(1 for _ in circuit.walk_operations()) < 1000
    assert oq.sample(circuit, shots=12, classical_bits=(0, 1)) == {"01": 12}


def test_wide_dynamic_address_checks_high_bits_without_enumerating_them():
    import openqml as oq

    source = "OPENQASM 3; qubit[3] q; bit[3] out; uint[64] i=2; x q[i]; out=measure q;"
    plan = oq.plan_conversion(source, target="json", max_depth=6)
    restored = Circuit.from_json(plan.export().value)
    assert oq.sample(restored, shots=12, classical_bits=(0, 1, 2)) == {"001": 12}
    unsafe = oq.convert(source.replace("i=2", "i=1099511627778"), target="ir").value
    with pytest.raises(oq.ValidationError, match="CLASSICAL_INDEX_RANGE"):
        oq.density_matrix(unsafe)
    assert "RUNTIME_INDEX_CHECK_UNSUPPORTED" in {
        d.code for d in oq.plan_conversion(unsafe, target="openqasm3").diagnostics
    }
    assert oq.plan_conversion(unsafe, target="json").verify().matched


@pytest.mark.parametrize("arms", ["case 0 {} case 0 {}", "default {} default {}", "x q;"])
def test_switch_rejects_duplicate_or_invalid_arms(arms):
    with pytest.raises(ConversionError, match="CLASSICAL_SWITCH_LABEL|SYNTAX"):
        convert(
            "OPENQASM 3.1; qubit q; bit c; switch(c){" + arms + "}",
            target="ir",
            source_options={"parser": "builtin"},
        )
