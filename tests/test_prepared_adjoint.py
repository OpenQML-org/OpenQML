"""Prepared batches keep numerical semantics, memory limits and model ownership."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import openqml as oq
from openqml import (
    AdjointPlan,
    Circuit,
    Expression,
    Hamiltonian,
    LayerCheckpoint,
    LayerSpec,
    Parameter,
    PauliString,
    ValidationError,
    adjoint_jvp,
    adjoint_vjp,
    expectations,
    parameter_shift_jacobian,
)


def workload():
    x, w, phase = (Parameter(name) for name in ("feature", "weight", "phase"))
    circuit = (
        Circuit(2)
        .h(0)
        .ry(0.3, 1)
        .u(x * w, w.sin(), x + w, 1)
        .crx(x.sin() + w**2, 1, 0)
        .cp(w, 0, 1)
        .ryy(x, 1, 0)
        .with_global_phase(phase + x * w)
    )
    obs = (PauliString("XY", 0.4), Hamiltonian((PauliString("YZ", -0.2), PauliString("II", 0.7))))
    rows = [
        {"feature": float(x), "weight": float(w), "phase": 0.3}
        for x, w in zip(np.linspace(-0.4, 0.3, 7), np.linspace(0.1, 0.6, 7), strict=True)
    ]
    cotangents = np.array(
        [[0.3, -0.2], [0.0, 0.0], [0.2, 0.1], [-0.3, 0.2], [0.1, -0.4], [0.4, 0.2], [-0.2, 0.5]]
    )
    tangents = np.arange(21, dtype=float).reshape(7, 3) / 17
    return circuit, obs, rows, cotangents, tangents


@pytest.mark.parametrize("chunk", [1, 2, 4, 128, None])
@pytest.mark.parametrize("columns", [False, True])
def test_prepared_batch_forms_and_partial_chunks_match_shift_derivatives(chunk, columns):
    circuit, obs, rows, cotangents, tangents = workload()
    names = ("weight", "phase", "feature")
    plan = AdjointPlan(circuit, list(obs), names)
    jac = parameter_shift_jacobian(circuit, obs, rows, parameters=names)
    bindings = (
        {name: np.array([row[name] for row in rows]) for name in circuit.parameters}
        if columns
        else rows
    )
    np.testing.assert_allclose(
        plan.vjp(bindings, cotangents, batch_chunk_size=chunk),
        np.einsum("bmp,bm->bp", jac, cotangents),
        atol=3e-12,
    )
    np.testing.assert_allclose(
        plan.jvp(bindings, tangents, batch_chunk_size=chunk),
        np.einsum("bmp,bp->bm", jac, tangents),
        atol=3e-12,
    )
    np.testing.assert_allclose(
        plan.expectations(bindings, batch_chunk_size=chunk),
        np.stack([expectations(circuit, obs, row) for row in rows]),
        atol=3e-12,
    )
    assert plan.observables == obs and plan.parameters == names


def test_reuse_and_column_selection_never_repeat_symbolic_differentiation(monkeypatch):
    circuit, obs, rows, cotangents, _ = workload()
    plan = AdjointPlan(circuit, obs)
    expected = parameter_shift_jacobian(circuit, obs, rows, parameters=("weight", "feature"))
    monkeypatch.setattr(
        Expression, "derivative", lambda *args: pytest.fail("Repeated symbolic differentiation")
    )
    for scale in (1.0, -0.3):
        np.testing.assert_allclose(
            plan.vjp(rows, cotangents * scale, parameters=("weight", "feature")),
            np.einsum("bmp,bm->bp", expected, cotangents * scale),
            atol=3e-12,
        )
        directions = np.full((7, 2), scale)
        np.testing.assert_allclose(
            plan.jvp(rows, directions, parameters=("weight", "feature")),
            np.einsum("bmp,bp->bm", expected, directions),
            atol=3e-12,
        )
    with pytest.raises(ValidationError):
        plan.vjp(rows, cotangents, parameters=("missing",))
    with pytest.raises(FrozenInstanceError):
        plan.parameters = ("feature",)


@pytest.mark.parametrize("kind", ["vjp", "jvp", "expectations"])
def test_working_budget_shrinks_chunks_without_changing_results(monkeypatch, kind):
    import openqml.adjoint as engine

    circuit = Circuit(2).ry(Parameter("w"), 0).cx(0, 1)
    obs = (PauliString("ZZ"), PauliString("XX"))
    rows = [{"w": float(x)} for x in np.linspace(0.1, 0.7, 9)]
    product = np.ones((9, 2 if kind == "vjp" else 1))
    normal = AdjointPlan(circuit, obs)
    small = AdjointPlan(circuit, obs, max_memory_mb=0.006)
    reference = (
        getattr(normal, kind)(rows, product)
        if kind != "expectations"
        else normal.expectations(rows)
    )
    apply, sizes = engine._apply_matrix, []

    def tracked(state, *args):
        sizes.append(state.shape[0])
        return apply(state, *args)

    monkeypatch.setattr(engine, "_apply_matrix", tracked)
    result = (
        getattr(small, kind)(rows, product, batch_chunk_size=50)
        if kind != "expectations"
        else small.expectations(rows, batch_chunk_size=50)
    )
    assert sizes and max(sizes) <= 2 and 1 in sizes
    np.testing.assert_allclose(result, reference, atol=2e-12)


def test_jvp_gate_work_is_independent_of_the_number_of_readouts(monkeypatch):
    import openqml.adjoint as engine

    circuit = (
        Circuit(2)
        .ry(Parameter("a"), 0)
        .crx(Parameter("b"), 0, 1)
        .rz(Parameter("a") * Parameter("b"), 1)
    )
    rows = [{"a": 0.3, "b": 0.2}] * 5
    directions = np.full((5, 2), 0.4)
    apply, calls = engine._apply_matrix, []

    def tracked(*args):
        calls.append(1)
        return apply(*args)

    monkeypatch.setattr(engine, "_apply_matrix", tracked)
    obs = [PauliString(a + b) for a in "IXYZ" for b in "IXYZ"]
    one = AdjointPlan(circuit, obs[:1]).jvp(rows, directions)
    first = len(calls)
    calls.clear()
    many = AdjointPlan(circuit, obs).jvp(rows, directions)
    assert len(calls) == first and first <= 3 * len(circuit.operations)
    np.testing.assert_allclose(many[:, :1], one, atol=1e-12)


def test_canceled_directions_skip_quantum_states_and_inactive_singular_columns_are_safe(
    monkeypatch,
):
    import openqml.adjoint as engine

    a, b = Parameter("a"), Parameter("b")
    plan = AdjointPlan(Circuit(1).ry(a - b, 0).rx((a - b).sin(), 0), [PauliString("Z")])
    with monkeypatch.context() as local:
        local.setattr(
            engine, "_tangent_forward", lambda *args: pytest.fail("Evolved a canceled direction")
        )
        np.testing.assert_array_equal(plan.jvp([{"a": 0.2, "b": 0.3}], [[1, 1]]), [[0]])
    plan = AdjointPlan(Circuit(1).ry(a.sqrt() + b, 0), [PauliString("Z")])
    with np.errstate(divide="raise", invalid="raise"):
        actual = plan.jvp([{"a": 0.0, "b": 0.2}, {"a": 1.0, "b": 0.3}], [[0, 1], [1, 0.4]])
    np.testing.assert_allclose(actual, [[-np.sin(0.2)], [-0.9 * np.sin(1.3)]], atol=1e-12)


def test_scalar_columns_empty_batches_and_parameter_free_circuits():
    plan = AdjointPlan(Circuit(1).ry(Parameter("w"), 0), [PauliString("Z")])
    result = plan.expectations({"w": 0.3}, batch_size=7, batch_chunk_size=3)
    np.testing.assert_allclose(result, np.full((7, 1), np.cos(0.3)))
    assert plan.expectations({"w": []}).shape == (0, 1)
    assert plan.vjp({"w": []}, np.empty((0, 1))).shape == (0, 1)
    assert plan.jvp({"w": []}, np.empty((0, 1))).shape == (0, 1)
    constant = AdjointPlan(Circuit(1).h(0).with_global_phase(0.2), [PauliString("X")])
    np.testing.assert_allclose(constant.expectations({}, batch_size=5), np.ones((5, 1)))
    assert constant.vjp({}, np.ones((5, 1)), batch_size=5).shape == (5, 0)
    np.testing.assert_array_equal(
        constant.jvp({}, np.empty((5, 0)), batch_size=5), np.zeros((5, 1))
    )


@pytest.mark.parametrize(
    "columns",
    [{"w": True}, {"w": [1j]}, {"w": [np.inf]}, {"w": [[0.3]]}, {"w": "0.3"}, {"unknown": 0.3}],
)
def test_invalid_binding_columns_fail_before_state_allocation(monkeypatch, columns):
    import openqml.adjoint as engine

    monkeypatch.setattr(engine, "_forward", lambda *args: pytest.fail("Allocated invalid inputs"))
    with pytest.raises(ValidationError):
        AdjointPlan(Circuit(1).ry(Parameter("w"), 0), [PauliString("Z")]).expectations(columns)


def test_mismatched_and_late_invalid_columns_precede_all_state_work(monkeypatch):
    import openqml.adjoint as engine

    a, b = Parameter("a"), Parameter("b")
    plan = AdjointPlan(Circuit(1).ry(a.sqrt() + b, 0), [PauliString("Z")])
    monkeypatch.setattr(
        engine,
        "_forward",
        lambda *args: pytest.fail("Allocated before validating the complete batch"),
    )
    with pytest.raises(ValidationError):
        plan.expectations({"a": [0.1, 0.2], "b": [0.3, 0.4, 0.5]})
    with np.errstate(invalid="ignore"), pytest.raises(ValidationError):
        plan.expectations({"a": [0.1, 0.2, -0.1], "b": 0.3}, batch_chunk_size=1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_chunk_size": True},
        {"batch_chunk_size": 0},
        {"batch_size": -1},
        {"batch_size": True},
        {"batch_size": 2},
    ],
)
def test_invalid_batch_options_do_not_silently_change_row_counts(kwargs):
    with pytest.raises(ValidationError):
        AdjointPlan(Circuit(1).ry(Parameter("w"), 0), [PauliString("Z")]).expectations(
            [{"w": 0.3}], **kwargs
        )


def test_plan_never_caches_inputs_between_calls_and_standalone_chunks_agree():
    plan = AdjointPlan(Circuit(1).ry(Parameter("w") ** 2, 0), [PauliString("Z")])
    values = np.array([0.2, 0.4])
    original = plan.expectations({"w": values})
    values[:] = [0.6, 0.8]
    updated = plan.expectations({"w": values})
    np.testing.assert_allclose(original[:, 0], np.cos(np.array([0.2, 0.4]) ** 2))
    np.testing.assert_allclose(updated[:, 0], np.cos(values**2))
    rows = [{"w": float(value)} for value in values]
    np.testing.assert_allclose(
        adjoint_vjp(plan.circuit, plan.observables, rows, [[1], [1]], batch_chunk_size=1),
        plan.vjp(rows, [[1], [1]]),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        adjoint_jvp(plan.circuit, plan.observables, rows, [[1], [1]], batch_chunk_size=1),
        plan.jvp(rows, [[1], [1]]),
        atol=1e-12,
    )


def test_torch_reuses_plan_and_column_bindings_for_forward_backward_and_selection(monkeypatch):
    torch = pytest.importorskip("torch")
    import openqml.interfaces._spec as bridges
    from openqml.interfaces.torch import QuantumLayer

    x, w = Parameter("x"), Parameter("w")
    spec = LayerSpec(Circuit(1).ry(x.sin() + x * w + w**2, 0), ("x",), ("w",))
    layer = QuantumLayer.from_checkpoint(
        LayerCheckpoint(spec, [0.3], "float64"), backend="numpy", gradient_method="adjoint"
    )
    plan = layer.adjoint_plan
    monkeypatch.setattr(
        Expression, "derivative", lambda *args: pytest.fail("Recompiled gradient expressions")
    )
    monkeypatch.setattr(
        bridges, "numeric_rows", lambda *args: pytest.fail("Constructed scalar binding rows")
    )
    monkeypatch.setattr(
        Circuit, "bind", lambda *args, **kwargs: pytest.fail("Constructed bound IR copies")
    )
    for input_grad in (False, True):
        inputs = torch.tensor([[0.2], [-0.4]], dtype=torch.float64, requires_grad=input_grad)
        actual = layer(inputs)
        expected = torch.cos(torch.sin(inputs) + inputs * layer.weights + layer.weights**2)
        selected = (inputs, layer.weights) if input_grad else (layer.weights,)
        got = torch.autograd.grad(actual.sum(), selected)
        wanted = torch.autograd.grad(expected.sum(), selected)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        for a, b in zip(got, wanted, strict=True):
            torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
        assert layer.adjoint_plan is plan
    assert layer.checkpoint().spec == spec


def test_jax_compiled_calls_use_new_bindings_with_one_prepared_plan(monkeypatch):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from openqml.interfaces.jax import QuantumLayer

    with oq.jax.enable_x64():
        x, w = Parameter("x"), Parameter("w")
        layer = QuantumLayer(
            Circuit(1).ry(x * w + w**2, 0),
            ("x",),
            ("w",),
            backend="numpy",
            gradient_method="adjoint",
        )
        plan = layer.adjoint_plan
        monkeypatch.setattr(
            Expression, "derivative", lambda *args: pytest.fail("Recompiled JAX bridge plan")
        )
        function = jax.jit(
            jax.value_and_grad(lambda weights, inputs: layer(weights, inputs).sum(), argnums=(0, 1))
        )
        for weight in (0.2, -0.4):
            weights, inputs = jnp.array([weight]), jnp.array([[0.3], [0.6]])
            value, (dw, dx) = function(weights, inputs)
            angle = inputs * weights + weights**2
            np.testing.assert_allclose(value, np.cos(angle).sum(), atol=1e-12)
            np.testing.assert_allclose(
                dw, (-np.sin(angle) * (inputs + 2 * weights)).sum(axis=0), atol=1e-12
            )
            np.testing.assert_allclose(dx, -np.sin(angle) * weights, atol=1e-12)
            assert layer.adjoint_plan is plan
