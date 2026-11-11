import numpy as np
import pytest

import openqml as oq
from openqml._tensor_contraction import TensorNetworkPlan


@pytest.mark.parametrize("n,k", [(1, 1), (3, 1), (5, 2), (6, 3)])
def test_multiscale_exact_readouts_match_dense(n, k):
    recipe = oq.MultiscalePlan(n, bond_qubits=k, depth=1)
    spec = recipe.to_spec()
    rng = np.random.default_rng(n + k)
    bindings = {name: rng.normal() for name in spec.input_parameters + spec.weight_parameters}
    planned = TensorNetworkPlan(spec.circuit, spec.observables)
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


def test_multiscale_resource_budgets_and_validation():
    with pytest.raises(oq.ResourceError):
        oq.MultiscalePlan(8, max_operations=8)
    with pytest.raises(oq.ValidationError):
        oq.MultiscalePlan(3, num_outputs=4)
    with pytest.raises(oq.ValidationError):
        oq.MultiscalePlan(3, bond_qubits=True)
    with pytest.raises(oq.ValidationError):
        TensorNetworkPlan(oq.Circuit(1), oq.z_observables(1), rematerialize=1)


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
