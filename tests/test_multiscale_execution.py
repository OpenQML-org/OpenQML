import numpy as np
import pytest

import openqml as oq


@pytest.mark.parametrize("n,k", [(1, 1), (3, 1), (5, 2), (6, 3)])
def test_multiscale_exact_readouts_match_dense(n, k):
    recipe = oq.MultiscalePlan(n, bond_qubits=k, depth=1)
    spec = recipe.to_spec()
    rng = np.random.default_rng(n + k)
    bindings = {name: rng.normal() for name in spec.input_parameters + spec.weight_parameters}
    planned = oq.TensorNetworkPlan(spec.circuit, spec.observables)
    np.testing.assert_allclose(
        planned.evaluate(bindings),
        oq.expectations(spec.circuit, spec.observables, bindings),
        atol=2e-12,
    )
    assert recipe.retained_wires and len(recipe.retained_wires) == min(k, n)


def test_unpooled_family_supports_full_schmidt_rank():
    recipe = oq.MultiscalePlan(4, bond_qubits=4, depth=3)
    spec = recipe.to_spec()
    rng = np.random.default_rng(191)
    bindings = {
        name: rng.uniform(-np.pi, np.pi) for name in spec.input_parameters + spec.weight_parameters
    }
    state = oq.statevector(spec.circuit, bindings)
    singular = np.linalg.svd(state.reshape(4, 4), compute_uv=False)
    assert singular[-1] > 1e-3
    assert recipe.to_dict()["unpooled_limit"]
    assert len(spec.observables) == 3


def _crossing_pairs():
    return oq.CircuitBuilder(4).h(0).h(1).cx(0, 2).cx(3, 1).cx(1, 3).build()


def test_adaptive_mps_refines_and_restores_wire_order():
    circuit = _crossing_pairs()
    obs = oq.z_observables(4) + (oq.PauliString("XIXI"), oq.PauliString("IZIZ"))
    report = oq.AdaptiveMPSBackend(
        initial_bond_dimension=1, max_bond_dimension=4, max_observable_error=1e-10
    ).evaluate(circuit, obs)
    assert report.attempted_bonds == (1, 2, 4)
    assert report.routed_operations > len(circuit.operations)
    assert report.truncation_free
    np.testing.assert_allclose(report.values, oq.expectations(circuit, obs), atol=1e-12)
    assert max(report.observable_error_bounds) <= 1e-10


def test_low_rank_certificate_contains_actual_error():
    circuit = _crossing_pairs()
    obs = oq.z_observables(4)
    report = oq.AdaptiveMPSBackend(
        initial_bond_dimension=1, max_bond_dimension=1, max_observable_error=2
    ).evaluate(circuit, obs)
    error = np.abs(report.values - oq.expectations(circuit, obs))
    assert np.all(error <= report.observable_error_bounds + 1e-12)
    assert not report.truncation_free
    with pytest.raises(oq.ResourceError, match="rank cap"):
        oq.AdaptiveMPSBackend(
            initial_bond_dimension=1, max_bond_dimension=1, max_observable_error=1e-10
        ).evaluate(circuit, obs)
    with pytest.raises(oq.ResourceError, match="total_work"):
        oq.AdaptiveMPSBackend(max_total_work=1).evaluate(circuit, obs)
    with pytest.raises(oq.ValidationError, match="nonlinear"):
        oq.AdaptiveMPSBackend().prepare_gradient_circuit(circuit)


def test_routed_ccx_and_barrier_match_dense():
    circuit = oq.CircuitBuilder(4).h(0).ry(0.4, 1).ccx(0, 1, 3).barrier().build()
    obs = oq.z_observables(4)
    report = oq.AdaptiveMPSBackend(max_bond_dimension=4, max_work=10**8).evaluate(circuit, obs)
    np.testing.assert_allclose(report.values, oq.expectations(circuit, obs), atol=1e-12)


def test_multiscale_resource_budgets_and_validation():
    with pytest.raises(oq.ResourceError):
        oq.MultiscalePlan(8, max_operations=8)
    with pytest.raises(oq.ValidationError):
        oq.MultiscalePlan(3, num_outputs=4)
    with pytest.raises(oq.ValidationError):
        oq.MultiscalePlan(3, bond_qubits=True)
    with pytest.raises(oq.ValidationError):
        oq.TensorNetworkPlan(oq.Circuit(1), oq.z_observables(1), rematerialize=1)


def test_torch_multiscale_rematerialized_gradients_and_hessian():
    torch = pytest.importorskip("torch")
    layer = oq.torch.QuantumLayer.multiscale(
        4,
        bond_qubits=2,
        depth=1,
        backend="torch",
        execution_method="tensor-network",
        tensor_options={"rematerialize": True},
        dtype=torch.float64,
    )
    rng = np.random.default_rng(44)
    x = torch.tensor(rng.normal(size=(2, 4)), requires_grad=True)
    weights = layer.weights.detach().clone().requires_grad_()
    output = layer.functional(weights, x)
    dense = oq.torch.QuantumLayer.from_spec(layer.to_spec(), backend="torch", dtype=torch.float64)
    expected = dense.functional(weights, x)
    np.testing.assert_allclose(output.detach(), expected.detach(), atol=2e-12)
    actual_grad = torch.autograd.grad(output.square().sum(), (weights, x), create_graph=True)
    expected_grad = torch.autograd.grad(expected.square().sum(), (weights, x), create_graph=True)
    for left, right in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(left, right, atol=2e-10, rtol=2e-9)
    actual_second = torch.autograd.grad(sum(g.square().sum() for g in actual_grad), (weights, x))
    expected_second = torch.autograd.grad(
        sum(g.square().sum() for g in expected_grad), (weights, x)
    )
    for left, right in zip(actual_second, expected_second, strict=True):
        torch.testing.assert_close(left, right, atol=2e-9, rtol=2e-8)
    policy = oq.torch.QuantumActorCritic(
        3, 2, num_qubits=4, quantum_layer=layer, dtype=torch.float64
    )
    assert policy(torch.ones(2, 3, dtype=torch.float64)).policy.shape == (2, 2)


def test_jax_and_flax_multiscale_rematerialization():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    layer = oq.jax.QuantumLayer.multiscale(
        3,
        bond_qubits=2,
        depth=1,
        backend="jax",
        execution_method="tensor-network",
        tensor_options={"rematerialize": True},
    )
    weights = layer.init(jax.random.PRNGKey(3))
    x = jnp.asarray([[0.1, 0.2, 0.3]])
    fn = jax.jit(lambda w: layer(w, x).sum())
    first = jax.grad(fn)(weights)
    second = jax.jvp(jax.grad(fn), (weights,), (jnp.ones_like(weights),))[1]
    assert jnp.isfinite(first).all() and jnp.isfinite(second).all()
    pytest.importorskip("flax")
    module = oq.flax.QuantumLayer.multiscale(
        3,
        bond_qubits=2,
        depth=1,
        backend="jax",
        execution_method="tensor-network",
        tensor_options={"rematerialize": True},
    )
    state = module.init(jax.random.PRNGKey(4), x)
    assert jax.jit(module.apply)(state, x).shape == (1, 3)
