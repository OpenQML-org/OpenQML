"""Logical feedback rewrites preserve joint records, phases and derivatives."""

import json
from pathlib import Path

import numpy as np
import pytest

from openqml import (
    Circuit,
    ClassicalCondition,
    ConversionError,
    Parameter,
    PauliString,
    QuantumProgram,
    ValidationError,
    convert,
    density_matrix,
    optimize,
    simplify,
    statevector,
)


def exact_records(circuit, values=None):
    from openqml.density import _simulate

    values = {} if values is None else {k: v for k, v in values.items() if k in circuit.parameters}
    states, _, _ = _simulate(circuit, values, retain_classical=True)
    return {bits: np.asarray(rho) for bits, rho in states.items()}


def assert_equivalent(left, right, values=None):
    a, b = exact_records(left, values), exact_records(right, values)
    assert a.keys() == b.keys()
    for bits in a:
        np.testing.assert_allclose(a[bits], b[bits], atol=2e-12)


def repeated_feedback(repeats=12):
    x, w = Parameter("feature"), Parameter("weight")
    c = Circuit(2).ry(x, 0).measure(0, 0)
    for _ in range(repeats):
        c = c.if_else(
            ClassicalCondition((0,), 1),
            Circuit(2).ry(w * w, 1).with_global_phase(w / 17),
            Circuit(2).rx(w / 8, 1),
        )
    return c


def test_sparse_initial_zero_folds_unreachable_symbols_and_preserves_phase():
    w, dead = Parameter("weight"), Parameter("dead")
    c = Circuit(1, num_clbits=1_000_000_000).if_else(
        ClassicalCondition((999_999_999,), 0),
        Circuit(1).ry(w, 0).with_global_phase(w.sin()),
        Circuit(1).rx(dead, 0),
    )
    report = optimize(c)
    assert report.folded_conditionals == 1
    assert report.removed_parameters == ("dead",)
    assert report.circuit.num_clbits == c.num_clbits
    assert report.circuit.global_phase == c.operations[0].blocks[0].global_phase
    expected = Circuit(1).ry(w, 0).with_global_phase(w.sin())
    np.testing.assert_allclose(
        statevector(report.circuit, {"weight": 0.3}),
        statevector(expected, {"weight": 0.3}),
        atol=1e-13,
    )
    assert report.converged and optimize(report.circuit).circuit == report.circuit


@pytest.mark.parametrize("value", range(4))
def test_false_multibit_guards_do_not_imply_all_bits_are_complemented(value):
    c = (
        Circuit(3)
        .h(0)
        .h(1)
        .measure(0, 0)
        .measure(1, 1)
        .if_else(
            ClassicalCondition((0, 1), value),
            Circuit(3).x(2).if_else(ClassicalCondition((0,), value & 1), Circuit(3).ry(0.4, 2)),
            Circuit(3).if_else(ClassicalCondition((0,), 1), Circuit(3).x(2)),
        )
        .measure(2, 2)
    )
    result = optimize(c)
    assert result.folded_conditionals >= 1
    assert_equivalent(c, result.circuit)


def test_single_unknown_false_branch_refinement_and_join_kill_measurement_facts():
    c = (
        Circuit(3)
        .h(0)
        .measure(0, 0)
        .if_else(
            ClassicalCondition((0, 2), 1),
            Circuit(3).h(1).measure(1, 1),
            Circuit(3).if_else(ClassicalCondition((0,), 0), Circuit(3).x(2)),
        )
        .if_else(ClassicalCondition((1,), 0), Circuit(3).ry(0.8, 2))
        .measure(2, 2)
    )
    result = optimize(c)
    assert result.folded_conditionals == 1
    assert result.after.conditionals == 2
    assert_equivalent(c, result.circuit)


def test_atomic_selection_and_later_overwrite_prevent_invalid_conditional_merging():
    c = (
        Circuit(2)
        .h(0)
        .measure(0, 0)
        .if_else(ClassicalCondition((0,), 1), Circuit(2).x(0).measure(0, 0).x(1))
        .if_else(ClassicalCondition((0,), 1), Circuit(2).x(1))
        .measure(1, 1)
    )
    result = optimize(c)
    assert result.merged_conditionals == 0
    assert result.after.conditionals == 2
    assert_equivalent(c, result.circuit)


def test_matching_guards_with_different_bit_order_merge_without_losing_records():
    c = (
        Circuit(3)
        .h(0)
        .h(1)
        .measure(0, 2)
        .measure(1, 0)
        .if_else(ClassicalCondition((2, 0), 1), Circuit(3).ry(0.2, 2), Circuit(3).rx(0.1, 2))
        .if_else(ClassicalCondition((0, 2), 2), Circuit(3).ry(0.3, 2), Circuit(3).rx(0.4, 2))
    )
    result = optimize(c)
    assert result.merged_conditionals == 1 and result.after.conditionals == 1
    assert result.before.gates - result.after.gates == 2
    assert_equivalent(c, result.circuit)


@pytest.mark.parametrize("separator", ["barrier", "channel", "reset", "measure"])
def test_quantum_rewrites_respect_explicit_effect_and_scheduling_boundaries(separator):
    w = Parameter("weight")
    block = Circuit(2).ry(w, 1)
    if separator == "barrier":
        block = block.barrier()
    elif separator == "channel":
        block = block.amplitude_damping(0.2, 1)
    elif separator == "reset":
        block = block.reset(1)
    else:
        block = block.measure(1, 1)
    c = Circuit(2).h(0).measure(0, 0).if_else(ClassicalCondition((0,), 1), block)
    c = c.if_else(ClassicalCondition((0,), 1), Circuit(2).ry(-w, 1))
    result = optimize(c)
    # Decisions can merge, but their quantum rotations cannot cross the effect.
    assert result.merged_conditionals == 1
    assert sum(op.name == "ry" for _, op in result.circuit.walk_operations()) == 2
    assert_equivalent(c, result.circuit, {"weight": 0.7})


@pytest.mark.parametrize("seed", range(8))
def test_random_nested_programs_preserve_every_unnormalized_record_density(seed):
    rng = np.random.default_rng(seed)

    def body(depth):
        c = Circuit(3)
        for _ in range(5):
            q = int(rng.integers(3))
            choice = int(rng.integers(6))
            if choice == 0:
                c = c.ry(float(rng.normal()), q)
            elif choice == 1:
                c = c.measure(q, int(rng.integers(3)))
            elif choice == 2:
                c = c.amplitude_damping(float(rng.uniform()), q)
            elif choice == 3:
                c = c.reset(q)
            elif choice == 4 and depth:
                bits = tuple(int(x) for x in rng.choice(3, size=2, replace=False))
                c = c.if_else(
                    ClassicalCondition(bits, int(rng.integers(4))), body(depth - 1), body(depth - 1)
                )
            else:
                c = c.x(q).x(q)
        return c.with_global_phase(float(rng.normal()))

    c = Circuit(3).h(0).h(1).h(2).compose(body(2))
    result = optimize(c)
    assert_equivalent(c, result.circuit)
    assert optimize(result.circuit).circuit == result.circuit


def test_rewrite_limit_validation_precedes_recursive_statistics(monkeypatch):
    import openqml.optimization as module

    c = Circuit(1).if_else(ClassicalCondition((0,), 1), Circuit(1).x(0))

    def forbidden(_):
        raise AssertionError("statistics must follow validation")

    monkeypatch.setattr(module, "circuit_stats", forbidden)
    for options in ({"max_operations": 1}, {"max_depth": 0}):
        with pytest.raises(ConversionError, match="OPTIMIZATION_LIMIT"):
            optimize(c, **options)
    for options in ({"max_passes": 0}, {"max_passes": 33}, {"max_depth": True}):
        with pytest.raises(ValidationError):
            optimize(c, **options)
    with pytest.raises(ValidationError):
        optimize("not a circuit")


def test_pass_limit_is_reported_and_later_pass_exposes_conditional_fusion():
    c = (
        Circuit(2)
        .h(0)
        .measure(0, 0)
        .if_else(ClassicalCondition((0,), 1), Circuit(2).ry(0.3, 1))
        .x(0)
        .x(0)
        .if_else(ClassicalCondition((0,), 1), Circuit(2).ry(0.4, 1))
    )
    partial = optimize(c, max_passes=1)
    final = optimize(c)
    assert not partial.converged
    assert "OPTIMIZATION_PASS_LIMIT" in {d.code for d in partial.diagnostics}
    assert final.converged and final.after.conditionals == 1
    assert_equivalent(c, partial.circuit)
    assert_equivalent(c, final.circuit)


def test_large_nonlinear_rotation_runs_are_balanced_and_phase_exact():
    w = Parameter("weight")
    c = Circuit(1)
    for _ in range(1024):
        c = c.ry(w * w, 0)
    result = simplify(c)
    angle = result.operations[0].params[0]
    assert len(result.operations) == 1
    assert angle.evaluate({"weight": 0.2}) == pytest.approx(1024 * 0.04)
    assert angle.derivative("weight").evaluate({"weight": 0.2}) == pytest.approx(1024 * 0.4)
    restored = Circuit.from_json(result.to_json())
    np.testing.assert_allclose(
        statevector(restored, {"weight": 0.2}),
        statevector(Circuit(1).ry(1024 * 0.04, 0)),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        statevector(simplify(Circuit(1).rx(np.pi, 0).rx(np.pi, 0))), [-1, 0], atol=1e-13
    )


def test_conversion_and_captured_program_return_optimization_and_eliminated_symbols():
    dead = Parameter("dead")
    c = Circuit(1).if_else(
        ClassicalCondition((0,), 1), Circuit(1).rx(dead, 0), Circuit(1).ry(0.3, 0)
    )
    result = QuantumProgram.capture(c).convert("openqasm3", optimize=True)
    assert result.source_format == "ir" and result.optimization.removed_parameters == ("dead",)
    assert "PARAMETER_ELIMINATED" in {d.code for d in result.diagnostics}
    restored = convert(result.value, source="openqasm3", target="ir").value
    assert_equivalent(c, restored, {"dead": 0.9})
    for kwargs in (
        {"optimize": "yes"},
        {"optimization_options": {"max_passes": 1}},
        {"optimize": True, "optimization_options": {"unknown": 1}},
    ):
        with pytest.raises(ValidationError):
            convert(c, target="json", **kwargs)


@pytest.mark.parametrize("target", ["qiskit", "cirq", "openqasm3", "json"])
def test_optimized_feedback_roundtrip_and_native_import(target):
    if target == "qiskit":
        pytest.importorskip("qiskit")
    if target == "cirq":
        pytest.importorskip("cirq")
    c = repeated_feedback(6)
    exported = convert(c, target=target, optimize=True)
    assert exported.optimization.merged_conditionals == 5
    restored = convert(exported.value, source=target, target="ir").value
    values = {"feature": 0.7, "weight": 0.3}
    np.testing.assert_allclose(
        density_matrix(restored, values), density_matrix(c, values), atol=2e-12
    )
    if target in {"qiskit", "cirq"}:
        imported = convert(convert(c, target=target).value, target="ir", optimize=True)
        assert imported.source_format == target and imported.optimization.after.conditionals == 1
        assert_equivalent(c, imported.circuit, values)


def test_compiled_feedback_torch_first_second_derivatives_and_zero_branch_endpoint():
    torch = pytest.importorskip("torch")
    from openqml.interfaces.torch import QuantumLayer

    original = repeated_feedback(8)
    compiled = optimize(original).circuit
    assert compiled.parameters == original.parameters
    a = QuantumLayer(original, ["feature"], ["weight"], dtype=torch.float64)
    b = QuantumLayer(compiled, ["feature"], ["weight"], dtype=torch.float64)
    x = torch.tensor([[0.0], [0.7]], dtype=torch.float64, requires_grad=True)
    w = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)

    def call(layer, x, w):
        return torch.func.functional_call(layer, {"weights": w}, (x,))

    torch.testing.assert_close(call(a, x, w), call(b, x, w))
    assert torch.autograd.gradcheck(lambda x, w: call(b, x, w), (x, w))
    assert torch.autograd.gradgradcheck(lambda x, w: call(b, x, w), (x, w))
    ha = torch.autograd.functional.hessian(lambda x: call(a, x, w).sum(), x)
    hb = torch.autograd.functional.hessian(lambda x: call(b, x, w).sum(), x)
    torch.testing.assert_close(ha, hb, atol=2e-11, rtol=2e-11)
    assert abs(hb[0, 0, 0, 0]) > 0.01


def test_compiled_feedback_jax_jit_hessian_matches_independent_analytic_formula():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from openqml.interfaces.jax import QuantumLayer

    reps = 8
    layer = QuantumLayer(
        optimize(repeated_feedback(reps)).circuit,
        ["feature"],
        ["weight"],
        observables=(PauliString("IZ"),),
    )
    x = jnp.array([[0.7]], dtype=jnp.float32)
    w = jnp.array([0.3], dtype=jnp.float32)

    def reference(t):
        prob = jnp.sin(x[0, 0] / 2) ** 2
        return prob * jnp.cos(reps * t[0] ** 2) + (1 - prob) * jnp.cos(reps * t[0] / 8)

    np.testing.assert_allclose(
        jax.jit(jax.hessian(lambda t: layer(t, x).sum()))(w), jax.hessian(reference)(w), atol=3e-6
    )


def test_cli_uses_feedback_optimizer_and_reports_its_actual_work(tmp_path, capsys):
    from openqml.cli import main

    path = tmp_path / "feedback.json"
    path.write_text(repeated_feedback(4).to_json())
    assert main(["optimize", str(path), "--from", "json", "--diagnostics"]) == 0
    captured = capsys.readouterr()
    result = Circuit.from_json(captured.out)
    assert optimize(result).before.conditionals == 1
    diagnostics = json.loads(captured.err)
    assert "merged 3" in diagnostics[0]["message"]


def test_readme_python_examples_execute_when_torch_is_available():
    pytest.importorskip("torch")
    path = Path(__file__).resolve().parents[1] / "README.md"
    if not path.exists():
        pytest.skip("README is not installed as a Python module")
    import re

    examples = re.findall(r"```python\n(.*?)```", path.read_text(), re.S)
    assert len(examples) == 2
    for block in examples:
        exec(compile(block, "README.md", "exec"), {})


def test_large_conditional_phase_run_balances_without_repeated_scope_copies():
    w = Parameter("phase")
    c = Circuit(1).h(0).measure(0, 0)
    for _ in range(1024):
        c = c.if_else(ClassicalCondition((0,), 1), Circuit(1).with_global_phase(w * w))
    result = optimize(c)
    assert result.merged_conditionals == 1023 and result.after.conditionals == 1
    phase = result.circuit.operations[-1].blocks[0].global_phase
    assert phase.evaluate({"phase": 0.3}) == pytest.approx(1024 * 0.09)
    assert phase.derivative("phase").evaluate({"phase": 0.3}) == pytest.approx(1024 * 0.6)
    restored = Circuit.from_json(result.circuit.to_json())
    assert_equivalent(result.circuit, restored, {"phase": 0.3})


def test_distinct_nonlinear_angles_roundtrip_without_optional_qasm_sdk():
    w = Parameter("weight")
    circuit = Circuit(1)
    for i in range(48):
        circuit = circuit.ry((w * (i + 1) / 100).sin(), 0)
    exported = convert(circuit, target="openqasm3", optimize=True)
    restored = convert(
        exported.value, source="openqasm3", target="ir", source_options={"parser": "builtin"}
    ).value
    np.testing.assert_allclose(
        statevector(circuit, {"weight": 0.2}), statevector(restored, {"weight": 0.2}), atol=1e-12
    )


@pytest.mark.parametrize("limit", ["characters", "nodes", "depth"])
def test_angle_budgets_do_not_fall_back_to_sdk(limit, monkeypatch):
    import openqml.program as module

    if limit == "characters":
        angle = " " * 16_385 + "weight"
    elif limit == "depth":
        angle = "sin(" * 80 + "weight" + ")" * 80
    else:
        pieces = ["w"] * 2048
        while len(pieces) > 1:
            pieces = ["(" + pieces[i] + "+" + pieces[i + 1] + ")" for i in range(0, len(pieces), 2)]
        angle = pieces[0]

    def forbidden(*args, **kwargs):
        raise AssertionError("expression resources must not trigger a permissive SDK retry")

    monkeypatch.setattr(module.QuantumProgram, "from_qasm", forbidden)
    symbol = "w" if limit == "nodes" else "weight"
    source = "OPENQASM 3; input float " + symbol + "; qubit q; ry(" + angle + ") q;"
    with pytest.raises(ConversionError, match="EXPRESSION_LIMIT"):
        convert(source, source="openqasm3", target="ir")
