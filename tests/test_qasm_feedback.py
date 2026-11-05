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
