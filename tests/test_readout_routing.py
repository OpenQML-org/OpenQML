"""Sparse readout routing must avoid cross-products and retain native derivatives."""

import math
from dataclasses import FrozenInstanceError
from itertools import product

import numpy as np
import pytest

from openqml import (
    Circuit,
    ExpectationPlan,
    Hamiltonian,
    LayerSpec,
    Parameter,
    PauliString,
    ResourceError,
    expectations,
)


def routed_spec(blocks=3, readouts=1, tied=False):
    rng = np.random.default_rng(18)
    matrix, _ = np.linalg.qr(rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8)))
    labels = ["".join(p) for p in product("IXYZ", repeat=3) if set(p) != {"I"}]
    inputs = ("x",) if tied else tuple(f"x{i}" for i in range(blocks))
    weights = ("w",) if tied else tuple(f"w{i}" for i in range(blocks))
    c, obs = Circuit(3 * blocks), []
    for i in range(blocks):
        angle = Parameter(inputs[0 if tied else i]) * Parameter(weights[0 if tied else i])
        c = c.ry(angle, 3 * i).rx(0.23, 3 * i + 2).unitary(matrix, 3 * i, 3 * i + 1, 3 * i + 2)
    for offset in range(readouts):
        local = [labels[(i + offset * blocks) % len(labels)] for i in range(blocks)]
        terms = tuple(
            PauliString("III" * i + p + "III" * (blocks - i - 1), 1 / blocks)
            for i, p in enumerate(local)
        )
        obs.extend((PauliString("".join(local)), Hamiltonian(terms)))
    return LayerSpec(c, inputs, weights, tuple(obs))


@pytest.mark.parametrize("cap", [1, 2, 3, 32])
@pytest.mark.parametrize("readouts", [1, 2])
def test_sparse_readouts_evaluate_only_requested_lane_pairs(cap, readouts, monkeypatch):
    import openqml.simulator as simulator

    spec = routed_spec(3, readouts)
    plan = ExpectationPlan(spec.circuit, spec.observables, max_dense_batch=cap)
    bindings = {
        n: np.linspace(-0.3, 0.4, 6).reshape(2, 3) + i / 20
        for i, n in enumerate(spec.input_parameters + spec.weight_parameters)
    }
    reference = expectations(spec.circuit, spec.observables, bindings)
    rows, calls, original = [], [], simulator._expectation_from_state
    apply = simulator._apply_matrix

    def record(state, observable, width, ops):
        rows.append(math.prod(state.shape[:-1]))
        return original(state, observable, width, ops)

    def gate(state, *args):
        calls.append(state.shape)
        return apply(state, *args)

    monkeypatch.setattr(simulator, "_expectation_from_state", record)
    monkeypatch.setattr(simulator, "_apply_matrix", gate)
    np.testing.assert_allclose(plan.evaluate(bindings), reference, atol=3e-14)
    assert sum(rows) == 3 * readouts * 6
    assert len(calls) == 3 * math.ceil(3 / cap)
    batch = plan.dense_batches[0]
    report = batch.to_dict()
    assert report["requested_readout_pairs"] == 3 * readouts
    assert report["unrouted_readout_pairs"] == 9 * readouts
    assert batch.estimated_work < len(batch.parameter_slots) * batch.estimated_work_per_lane
    for route in batch.readout_routes:
        assert len(route.lanes) == len(route.factor_indices) == 1
        assert len(route.factor_indices[0]) == 1
    with pytest.raises(FrozenInstanceError):
        batch.readout_routes[0].lanes = ()


def test_tied_lane_keeps_all_distinct_requested_readouts():
    spec = routed_spec(3, 2, True)
    plan = ExpectationPlan(spec.circuit, spec.observables)
    batch = plan.dense_batches[0]
    assert len(batch.parameter_slots) == 1 and len(batch.readout_routes) == 6
    assert all(route.lanes == (0,) for route in batch.readout_routes)
    values = dict(x=np.array([-0.3, 0.2]), w=0.6)
    np.testing.assert_allclose(
        plan.evaluate(values), expectations(spec.circuit, spec.observables, values), atol=3e-14
    )


def test_routes_can_share_lanes_and_skip_packs_without_losing_output_order():
    spec = routed_spec(3, 2)
    # Repeat a readout on another component so one route spans multiple lanes.
    first = spec.observables[0].paulis[:3]
    obs = (PauliString(first * 3), *spec.observables)
    plan = ExpectationPlan(spec.circuit, obs, max_dense_batch=2)
    assert any(len(route.lanes) == 3 for route in plan.dense_batches[0].readout_routes)
    values = {n: 0.2 + i / 20 for i, n in enumerate(spec.input_parameters + spec.weight_parameters)}
    np.testing.assert_allclose(
        plan.evaluate(values), expectations(spec.circuit, obs, values), atol=3e-14
    )
    empty = plan.evaluate({n: np.empty((2, 0)) for n in values})
    assert empty.shape == (2, 0, len(obs))


def test_binary_matrix_keys_are_compact_exact_endian_stable_and_do_not_mutate_inputs():
    from openqml._scheduling import _matrix_key

    matrix = np.eye(128, dtype=complex)
    matrix[0, 1] = complex(-0.0, -0.0)
    before = matrix.tobytes()
    key = _matrix_key(matrix)
    assert isinstance(key, bytes) and len(key) == 16 * matrix.size
    assert matrix.tobytes() == before
    assert key == _matrix_key(np.array(matrix, dtype=">c16", order="F"))
    assert key == _matrix_key(np.eye(128, dtype=complex))
    matrix[0, 0] = np.nextafter(1.0, 2.0)
    assert key != _matrix_key(matrix)


def test_digest_collisions_never_decide_template_identity(monkeypatch):
    import openqml._scheduling as scheduler

    monkeypatch.setattr(scheduler, "_template_id", lambda key: "forced-collision")
    c = (
        Circuit(4)
        .ry(Parameter("x"), 0)
        .unitary(np.eye(4), 0, 1)
        .ry(Parameter("x"), 2)
        .unitary(np.diag([1, 1, -1, 1]), 2, 3)
    )
    obs = (PauliString("XIXI"),)
    plan = ExpectationPlan(c, obs)
    assert len(plan.dense_batches) == 2
    np.testing.assert_allclose(
        plan.evaluate({"x": 0.3}), expectations(c, obs, {"x": 0.3}), atol=3e-14
    )


def test_prepared_state_initialization_follows_plan_preflight_without_revalidating(monkeypatch):
    import openqml.simulator as simulator

    spec = routed_spec()
    plan = ExpectationPlan(spec.circuit, spec.observables, max_memory_mb=0.01)
    bindings = {n: 0.3 for n in spec.input_parameters + spec.weight_parameters}
    reference = expectations(spec.circuit, spec.observables, bindings)
    monkeypatch.setattr(
        simulator, "_resource_check", lambda *a, **kw: pytest.fail("Repeated public validation")
    )
    np.testing.assert_allclose(plan.evaluate(bindings), reference, atol=3e-14)
    monkeypatch.setattr(
        simulator, "_initial_state", lambda *a: pytest.fail("Allocated before preflight")
    )
    with pytest.raises(ResourceError, match="max_memory_mb"):
        plan.evaluate({n: np.ones(20) for n in bindings})


def test_torch_routed_generated_weights_compile_hessian_jvp_and_checkpoints():
    torch = pytest.importorskip("torch")
    from openqml.interfaces.torch import QuantumLayer

    spec = routed_spec(3, 2)
    layer = QuantumLayer.from_spec(
        spec, execution_method="planned", dtype=torch.float64, batch_chunk_size=2
    )
    dense = QuantumLayer.from_spec(spec, dtype=torch.float64)
    w = torch.linspace(-0.2, 0.3, 6, dtype=torch.float64).reshape(2, 1, 3).requires_grad_()
    x = torch.linspace(-0.3, 0.2, 9, dtype=torch.float64).reshape(3, 3).requires_grad_()
    actual = torch.autograd.grad(layer.functional(w, x).square().sum(), (w, x), create_graph=True)
    expected = torch.autograd.grad(dense.functional(w, x).square().sum(), (w, x), create_graph=True)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=3e-12, rtol=3e-12)
    a = torch.autograd.grad(sum(v.square().sum() for v in actual), (w, x))
    b = torch.autograd.grad(sum(v.square().sum() for v in expected), (w, x))
    for first, second in zip(a, b, strict=True):
        torch.testing.assert_close(first, second, atol=3e-12, rtol=3e-12)
    tangents = (torch.ones_like(w), torch.ones_like(x))
    for a, b in zip(
        torch.func.jvp(layer.functional, (w, x), tangents),
        torch.func.jvp(dense.functional, (w, x), tangents),
        strict=True,
    ):
        torch.testing.assert_close(a, b, atol=3e-12, rtol=3e-12)
    compiled = torch.compile(layer.functional, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(compiled(w, x), dense.functional(w, x))
    assert layer.checkpoint().spec == spec


def test_jax_routed_jit_vmap_hessian_and_flax_optax_training():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from openqml.interfaces.jax import QuantumLayer

    spec = routed_spec(3, 2)
    layer = QuantumLayer.from_spec(spec, execution_method="planned")
    dense = QuantumLayer.from_spec(spec)
    w, x = (
        jnp.array([0.2, -0.3, 0.4], dtype=jnp.float32),
        jnp.linspace(-0.2, 0.3, 9, dtype=jnp.float32).reshape(3, 3),
    )
    np.testing.assert_allclose(jax.jit(layer)(w, x), dense(w, x), atol=2e-6)
    np.testing.assert_allclose(
        jax.jit(jax.hessian(lambda a: layer(a, x).sum()))(w),
        jax.hessian(lambda a: dense(a, x).sum())(w),
        atol=3e-6,
    )
    np.testing.assert_allclose(
        jax.jit(jax.vmap(layer, in_axes=(0, None)))(jnp.stack((w, w * 2)), x),
        dense(jnp.stack((w, w * 2))[:, None, :], x),
        atol=3e-6,
    )
    pytest.importorskip("flax")
    optax = pytest.importorskip("optax")
    from openqml.interfaces.flax import QuantumLayer as FlaxLayer

    module = FlaxLayer.from_spec(spec, execution_method="planned")
    params = module.init(jax.random.key(3), x)
    optimizer = optax.adam(0.01)
    state = optimizer.init(params)
    gradient = jax.jit(jax.grad(lambda p: jnp.square(module.apply(p, x)).sum()))(params)
    updates, _ = optimizer.update(gradient, state, params)
    updated = optax.apply_updates(params, updates)
    assert np.isfinite(module.apply(updated, x)).all()
