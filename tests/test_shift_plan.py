"""Prepared shift plans preserve ownership, bounded work and backend sampling semantics."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import openqml as oq


def workload():
    x, w, phase = (oq.Parameter(name) for name in ("x", "w", "phase"))
    circuit = oq.Circuit(2).ry(x * w, 0).crx(w.sin(), 0, 1).rz(x - w, 1).with_global_phase(phase)
    observables = (oq.PauliString("ZI"), oq.PauliString("IX"))
    return circuit, observables


class Counting:
    name = "prepared-shift-counting"

    def __init__(self, *, deterministic=True, lower=False):
        self.capabilities = oq.BackendCapabilities(
            native_expectations=True, deterministic_expectations=deterministic
        )
        self.preparations = 0
        self.sizes = []
        self.jobs = []
        self.lower = lower

    def prepare_gradient_circuit(self, circuit):
        self.preparations += 1
        return oq.decompose(circuit).with_global_phase(0) if self.lower else circuit

    def expectations_batch(self, circuits, observables, *, options):
        self.sizes.append(len(circuits))
        self.jobs.extend(circuits)
        values = oq.expectations_batch(circuits, observables)
        if not self.capabilities.deterministic_expectations:
            # A bounded stateful estimator makes accidental result sharing visible.
            start = sum(self.sizes) - len(circuits)
            values = values * 0.5 + 0.01 * np.sin(
                np.arange(start, start + len(circuits))[:, None] + 0.2
            )
        return values


@pytest.mark.parametrize("kind", ["jacobian", "jvp", "vjp"])
@pytest.mark.parametrize("columns", [False, True])
def test_coalesced_products_match_independent_finite_differences(kind, columns):
    circuit, obs = workload()
    backend = Counting()
    plan = oq.ShiftPlan(circuit, obs, ("x", "phase", "w"), backend)
    points = [dict(x=0.2, w=0.4, phase=0.1), dict(x=0.7, w=-0.3, phase=0.1)]
    rows = [points[i % 2] for i in range(12)]
    bindings = (
        {name: np.array([r[name] for r in rows]) for name in circuit.parameters}
        if columns
        else iter(rows)
    )
    reference = np.zeros((12, 2, 3))
    for index, name in enumerate(plan.parameters):
        plus = [{**row, name: row[name] + 1e-6} for row in rows]
        minus = [{**row, name: row[name] - 1e-6} for row in rows]
        reference[..., index] = (
            oq.expectations_batch(circuit, obs, plus) - oq.expectations_batch(circuit, obs, minus)
        ) / 2e-6
    if kind == "jacobian":
        actual, expected = plan.jacobian(bindings, chunk_size=3), reference
    elif kind == "jvp":
        directions = np.linspace(0.1, 0.7, 36).reshape(12, 3)
        actual = plan.jvp(bindings, directions, chunk_size=3)
        expected = np.einsum("bmp,bp->bm", reference, directions)
    else:
        cotangents = np.linspace(-0.4, 0.5, 24).reshape(12, 2)
        actual = plan.vjp(bindings, cotangents, chunk_size=3)
        expected = np.einsum("bmp,bm->bp", reference, cotangents)
    np.testing.assert_allclose(actual, expected, atol=3e-9, rtol=3e-9)
    assert backend.preparations == 1
    assert sum(backend.sizes) == 16  # Eight shifts per distinct point.
    assert max(backend.sizes) <= 3


def test_lowering_derivatives_and_parameter_selection_are_reused(monkeypatch):
    circuit, obs = workload()
    backend = Counting(lower=True)
    plan = oq.ShiftPlan(circuit, list(obs), ["phase", "w", "x"], backend)
    rows = [dict(x=0.3, w=0.4, phase=0.2)] * 4
    reference = oq.ShiftPlan(circuit, obs, ("x", "w")).jacobian(rows)
    with monkeypatch.context() as patch:
        patch.setattr(oq.Expression, "derivative", lambda *a: pytest.fail("Recompiled derivatives"))
        for scale in (0.2, -0.4):
            cotangents = np.full((4, 2), scale)
            directions = np.full((4, 2), -scale)
            np.testing.assert_allclose(
                plan.vjp(rows, cotangents, parameters=("x", "w")),
                np.einsum("bmp,bm->bp", reference, cotangents),
                atol=2e-12,
            )
            np.testing.assert_allclose(
                plan.jvp(rows, directions, parameters=("x", "w")),
                np.einsum("bmp,bp->bm", reference, directions),
                atol=2e-12,
            )
    assert backend.preparations == 1
    assert plan.circuit is circuit and "phase" not in plan.lowered_circuit.parameters
    assert plan.parameters == ("phase", "w", "x")
    assert not any(isinstance(value, np.ndarray) for value in vars(plan).values())
    with pytest.raises(FrozenInstanceError):
        plan.coalesce = False
    with pytest.raises(oq.ValidationError):
        plan.vjp(rows, np.ones((4, 2)), parameters=("missing",))


def test_reuse_windows_and_execution_chunks_have_independent_bounds():
    circuit, obs = workload()
    backend = Counting()
    plan = oq.ShiftPlan(circuit, obs, backend=backend, max_reuse_rows=3)
    rows = [dict(x=0.3, w=0.4, phase=0.2)] * 9
    directions = np.tile([0.0, 1.0, 0.2], (9, 1))
    plan.jvp(rows, directions, chunk_size=5)
    assert sum(backend.sizes) == 24 and max(backend.sizes) == 5
    assert backend.sizes[-1] == 4
    before = sum(backend.sizes)
    plan.jvp(rows, directions, chunk_size=5)
    assert sum(backend.sizes) - before == 24  # No execution results survive the call.


@pytest.mark.parametrize("delta", [-0.0, np.nextafter(0.0, 1.0)])
def test_binding_keys_do_not_quantize_or_merge_signed_zero(delta):
    circuit = oq.Circuit(1).ry(oq.Parameter("x"), 0)
    backend = Counting()
    plan = oq.ShiftPlan(circuit, [oq.PauliString("Z")], backend=backend)
    result = plan.jvp([{"x": 0.0}, {"x": delta}], np.ones((2, 1)))
    assert sum(backend.sizes) == 4
    np.testing.assert_allclose(result[:, 0], -np.sin([0.0, delta]), atol=1e-14)


def test_sampling_and_stateful_estimators_keep_independent_shift_requests():
    circuit, obs = workload()
    rows = [dict(x=0.3, w=0.4, phase=0.2)] * 6
    backend = Counting(deterministic=False)
    plan = oq.ShiftPlan(circuit, obs, backend=backend)
    result = plan.jvp(rows, np.tile([0.0, 1.0, 0.2], (6, 1)), chunk_size=3)
    assert not plan.coalesce and sum(backend.sizes) == 48
    assert not np.allclose(result, result[0])
    with pytest.raises(oq.ValidationError, match="deterministic_expectations"):
        oq.ShiftPlan(circuit, obs, backend=backend, coalesce=True)
    sampled = oq.ShiftPlan(circuit, obs, backend=oq.ShotBackend(shots=16))
    assert not sampled.coalesce


@pytest.mark.parametrize("kind", ["jacobian", "jvp", "vjp"])
def test_all_rows_are_checked_before_any_external_work(kind):
    circuit = oq.Circuit(1).ry(oq.Parameter("x"), 0).bit_flip(oq.Parameter("p"), 0)
    backend = Counting()
    plan = oq.ShiftPlan(circuit, [oq.PauliString("Z")], backend=backend, max_reuse_rows=1)
    rows = [dict(x=0.3, p=0.2), dict(x=0.1, p=1.1)]
    with pytest.raises(oq.ValidationError, match="probability"):
        if kind == "jacobian":
            plan.jacobian(rows, chunk_size=2)
        elif kind == "jvp":
            plan.jvp(rows, np.ones((2, 2)), chunk_size=2)
        else:
            plan.vjp(rows, np.ones((2, 1)), chunk_size=2)
    assert backend.sizes == []


def test_layer_cannot_mix_forward_and_backward_after_backend_replacement():
    torch = pytest.importorskip("torch")
    first, second = Counting(), Counting()
    first.name = second.name = "shift-layer-replacement"
    oq.register_backend(first)
    try:
        layer = oq.torch.QuantumLayer(
            oq.Circuit(1).ry(oq.Parameter("x") * oq.Parameter("w"), 0),
            ["x"],
            ["w"],
            backend=first.name,
            dtype=torch.float64,
        )
        inputs = torch.tensor([[0.3]], dtype=torch.float64, requires_grad=True)
        output = layer(inputs)
        before = sum(first.sizes)
        oq.register_backend(second, replace=True)
        with pytest.raises(oq.ValidationError, match="Registered backend changed"):
            output.sum().backward()
        with pytest.raises(oq.ValidationError, match="Registered backend changed"):
            layer(inputs)
        assert sum(first.sizes) == before and second.sizes == []
    finally:
        oq.unregister_backend(first.name)


def test_lowered_expression_domains_are_preflighted_too():
    class Lowered(Counting):
        def prepare_gradient_circuit(self, circuit):
            return oq.Circuit(1).ry(oq.Parameter("x").sqrt(), 0)

    backend = Lowered()
    plan = oq.ShiftPlan(
        oq.Circuit(1).ry(oq.Parameter("x"), 0), [oq.PauliString("Z")], backend=backend
    )
    with np.errstate(invalid="ignore"), pytest.raises(oq.ValidationError, match="expression"):
        plan.jvp([{"x": 0.3}, {"x": -0.3}], np.ones((2, 1)), chunk_size=2)
    assert backend.sizes == []


def test_empty_batches_zero_products_and_inactive_singular_columns():
    circuit = oq.Circuit(1).ry(oq.Parameter("x").sqrt() + oq.Parameter("w"), 0)
    backend = Counting()
    plan = oq.ShiftPlan(circuit, [oq.PauliString("Z")], backend=backend)
    rows = [dict(x=0.0, w=0.3)] * 4
    # The path moves only w; no derivative in the singular x direction is needed.
    result = plan.jvp(rows, np.tile([0.2, 0.0], (4, 1)))
    np.testing.assert_allclose(result[:, 0], -0.2 * np.sin(0.3), atol=2e-12)
    backend.sizes.clear()
    np.testing.assert_array_equal(plan.vjp(rows, np.zeros((4, 1))), np.zeros((4, 2)))
    assert backend.sizes == []
    assert plan.jvp([], np.empty((0, 2))).shape == (0, 1)
    assert plan.jacobian([], parameters=()).shape == (0, 1, 0)
    constant = oq.ShiftPlan(oq.Circuit(1).h(0), [oq.PauliString("Z")])
    np.testing.assert_array_equal(
        constant.jvp({}, np.empty((5, 0)), batch_size=5), np.zeros((5, 1))
    )


@pytest.mark.parametrize(
    "argument,value", [("max_reuse_rows", 0), ("max_reuse_rows", True), ("coalesce", "yes")]
)
def test_invalid_plan_options_are_rejected(argument, value):
    with pytest.raises(oq.ValidationError):
        oq.ShiftPlan(
            oq.Circuit(1).ry(oq.Parameter("x"), 0), [oq.PauliString("Z")], **{argument: value}
        )


def test_deterministic_capability_is_explicit_and_boolean():
    assert not oq.BackendCapabilities().deterministic_expectations
    assert oq.get_backend("numpy").capabilities.deterministic_expectations
    with pytest.raises(oq.ValidationError, match="booleans"):
        oq.BackendCapabilities(deterministic_expectations=1)


@pytest.mark.parametrize("framework", ["torch", "jax"])
def test_mapped_directions_share_quantum_jobs_through_standard_transforms(framework):
    lib = pytest.importorskip(framework)
    circuit, obs = workload()
    backend = Counting()
    if framework == "torch":
        layer = oq.torch.QuantumLayer(
            circuit, ["x"], ["w", "phase"], obs, backend=backend, dtype=lib.float64
        )
        w = lib.tensor([0.4, 0.1], dtype=lib.float64)
        x = lib.tensor([[0.7]], dtype=lib.float64)
        directions = lib.tensor(
            np.column_stack((np.linspace(0.1, 0.5, 32), np.zeros(32))), dtype=lib.float64
        )
        actual = lib.func.vmap(
            lambda d: lib.func.jvp(layer.functional, (w, x), (d, lib.zeros_like(x)))[1]
        )(directions)
        actual = actual.detach().numpy()
    else:
        import jax.numpy as jnp

        layer = oq.jax.QuantumLayer(circuit, ["x"], ["w", "phase"], obs, backend=backend)
        w, x = jnp.array([0.4, 0.1], dtype=jnp.float32), jnp.array([[0.7]], dtype=jnp.float32)
        directions = jnp.asarray(
            np.column_stack((np.linspace(0.1, 0.5, 32), np.zeros(32))), dtype=jnp.float32
        )
        actual = lib.jit(lib.vmap(lambda d: lib.jvp(layer, (w, x), (d, jnp.zeros_like(x)))[1]))(
            directions
        )
        actual = np.asarray(actual)
    row = dict(
        x=float(np.asarray(x)[0, 0]), w=float(np.asarray(w)[0]), phase=float(np.asarray(w)[1])
    )
    primal = circuit.bind(row)

    def angles(c):
        return tuple(p.constant for _, op in c.walk_operations() for p in op.params)

    shifts = sum(angles(job) != angles(primal) for job in backend.jobs)
    assert shifts == 8 and backend.preparations == 1
    jacobian = oq.ShiftPlan(circuit, obs, ("w", "phase")).jacobian([row])
    expected = np.einsum("bmp,dp->dbm", jacobian, np.asarray(directions))
    np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize("kind", ["jacobian", "jvp", "vjp"])
def test_noisy_measurement_feedback_occurrences_match_finite_differences(kind):
    x, w, p = (oq.Parameter(name) for name in ("x", "w", "p"))
    circuit = (
        oq.Circuit(2)
        .ry(x * w, 0)
        .rx(x.sin(), 1)
        .measure(0, 0)
        .if_else(oq.ClassicalCondition((0,), 1), oq.Circuit(2).ry(w * w, 1), oq.Circuit(2).rz(w, 1))
        .amplitude_damping(p, 1)
        .bit_flip(p / 2, 0)
    )
    obs = (oq.PauliString("IZ"), oq.PauliString("IX"))
    rows = [dict(x=0.3, w=0.6, p=0.2), dict(x=-0.2, w=0.4, p=0.3)] * 3
    plan = oq.ShiftPlan(circuit, obs, ("w", "p", "x"), max_reuse_rows=3)
    finite = np.zeros((6, 2, 3))
    for index, name in enumerate(plan.parameters):
        plus = [{**r, name: r[name] + 1e-6} for r in rows]
        minus = [{**r, name: r[name] - 1e-6} for r in rows]
        finite[..., index] = (
            oq.expectations_batch(circuit, obs, plus) - oq.expectations_batch(circuit, obs, minus)
        ) / 2e-6
    if kind == "jacobian":
        actual, expected = plan.jacobian(rows, chunk_size=3), finite
    elif kind == "jvp":
        directions = np.linspace(0.1, 0.6, 18).reshape(6, 3)
        actual = plan.jvp(rows, directions, chunk_size=3)
        expected = np.einsum("bmp,bp->bm", finite, directions)
    else:
        cotangents = np.linspace(-0.3, 0.4, 12).reshape(6, 2)
        actual = plan.vjp(rows, cotangents, chunk_size=3)
        expected = np.einsum("bmp,bm->bp", finite, cotangents)
    np.testing.assert_allclose(actual, expected, atol=4e-9, rtol=4e-9)


@pytest.mark.parametrize("backend", ["qiskit", "pennylane", "cirq"])
def test_sdk_shift_plan_reuse_and_imported_source_ownership(backend):
    pytest.importorskip(backend)
    circuit, obs = workload()
    rows = [dict(x=0.3, w=0.4, phase=0.2)] * 3
    if backend in {"qiskit", "cirq"}:
        exported = oq.convert(circuit, target=backend).value
        circuit = oq.convert(exported, source=backend, target="ir").value
    plan = oq.ShiftPlan(circuit, obs, ("w", "phase", "x"), backend=backend)
    assert plan.coalesce
    reference = oq.ShiftPlan(circuit, obs, plan.parameters, coalesce=False).jacobian(rows)
    directions = np.linspace(0.1, 0.7, 9).reshape(3, 3)
    cotangents = np.linspace(-0.4, 0.2, 6).reshape(3, 2)
    np.testing.assert_allclose(
        plan.jvp(rows, directions, chunk_size=3),
        np.einsum("bmp,bp->bm", reference, directions),
        atol=2e-11,
    )
    np.testing.assert_allclose(
        plan.vjp(rows, cotangents, chunk_size=3),
        np.einsum("bmp,bm->bp", reference, cotangents),
        atol=2e-11,
    )


@pytest.mark.parametrize("change", ["wire", "symbol"])
def test_invalid_backend_lowering_fails_during_plan_preparation(change):
    class Invalid(Counting):
        def prepare_gradient_circuit(self, circuit):
            return (
                oq.Circuit(2).ry(oq.Parameter("x"), 0)
                if change == "wire"
                else oq.Circuit(1).ry(oq.Parameter("new"), 0)
            )

    backend = Invalid()
    with pytest.raises(oq.ValidationError, match="preserve wires and parameter names"):
        oq.ShiftPlan(oq.Circuit(1).ry(oq.Parameter("x"), 0), [oq.PauliString("Z")], backend=backend)
    assert not backend.sizes


def test_registered_backend_replacement_requires_plan_recreation():
    first, second = Counting(), Counting()
    first.name = second.name = "shift-replacement"
    oq.register_backend(first)
    try:
        circuit = oq.Circuit(1).ry(oq.Parameter("x"), 0)
        observables = [oq.PauliString("Z")]
        named = oq.ShiftPlan(circuit, observables, backend=first.name)
        pinned = oq.ShiftPlan(circuit, observables, backend=first)
        oq.register_backend(second, replace=True)
        with pytest.raises(oq.ValidationError, match="Registered backend changed"):
            named.jvp([{"x": 0.3}], [[0.1]])
        assert first.sizes == second.sizes == []
        np.testing.assert_allclose(pinned.jvp([{"x": 0.3}], [[0.1]]), [[-0.1 * np.sin(0.3)]])
    finally:
        oq.unregister_backend(first.name)


def test_backend_cannot_enable_reuse_then_drop_its_determinism_contract():
    backend = Counting()
    plan = oq.ShiftPlan(
        oq.Circuit(1).ry(oq.Parameter("x"), 0), [oq.PauliString("Z")], backend=backend
    )
    backend.capabilities = oq.BackendCapabilities(native_expectations=True)
    with pytest.raises(oq.ValidationError, match="determinism changed"):
        plan.jvp([{"x": 0.3}], [[0.1]])
    assert backend.sizes == []
