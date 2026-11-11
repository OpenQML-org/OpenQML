"""Memory-admitted parameter-grid streaming and native derivative equivalence."""

import gc
import sys

import numpy as np
import pytest

import openqml as oq
from openqml._tensor_contraction import TensorNetworkPlan


@pytest.fixture(autouse=True)
def release_compilations():
    yield
    if "jax" in sys.modules:
        sys.modules["jax"].clear_caches()
    gc.collect()


def plans():
    c = oq.Circuit(3).ry(oq.Parameter("a"), 0).rxx(oq.Parameter("b"), 0, 1).cx(1, 2)
    obs = (oq.PauliString("ZZZ"), oq.PauliString("XII"))
    base = TensorNetworkPlan(c, obs)
    budget = (base._metadata_bytes + 10 * (base.working_bytes - base._metadata_bytes)) / 2**20
    return base, TensorNetworkPlan(c, obs, max_memory_mb=budget)


def test_broadcast_grid_chunks_and_tail_match_dense_batch_without_grid_materialization():
    base, limited = plans()
    binding = {"a": np.linspace(-0.4, 0.4, 7)[:, None], "b": np.linspace(-0.3, 0.3, 9)[None]}
    with pytest.raises(oq.ResourceError):
        limited.evaluate(binding)
    expected = base.evaluate(binding)
    actual, report = limited.evaluate(binding, batch_chunk_size=5, return_report=True)
    assert actual.shape == (7, 9, 2) and report["chunks"] >= 13
    np.testing.assert_allclose(actual, expected, atol=2e-14)
    chunks = list(limited.iter_evaluate(binding, batch_chunk_size=4))
    assert all(len(c) <= 4 for c in chunks) and len(chunks[-1]) == 3
    np.testing.assert_allclose(np.concatenate(chunks), expected.reshape(63, 2), atol=2e-14)


def test_constant_scalar_and_empty_batch_streams():
    p = TensorNetworkPlan(oq.Circuit(1), (oq.PauliString("I"),))
    value = p.evaluate(batch_shape=(3, 4), batch_chunk_size=5)
    np.testing.assert_array_equal(value, np.ones((3, 4, 1)))
    np.testing.assert_array_equal(p.evaluate(batch_chunk_size=2), [1])
    assert p.evaluate(batch_shape=(0, 3), batch_chunk_size=2).shape == (0, 3, 1)
    with pytest.raises(oq.ValidationError):
        p.evaluate(batch_chunk_size=0)


def test_torch_streaming_gradients_hessians_and_shared_scalar_weights():
    torch = pytest.importorskip("torch")
    base, limited = plans()
    a = torch.tensor([[-0.2], [0.3]], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([[0.1, 0.4, 0.5]], dtype=torch.float64, requires_grad=True)

    def evaluate(x, w):
        return limited.evaluate({"a": x, "b": w}, backend="torch", like=x, batch_chunk_size=2)

    torch.testing.assert_close(
        evaluate(a, b), base.evaluate({"a": a, "b": b}, backend="torch", like=a)
    )
    assert torch.autograd.gradcheck(evaluate, (a, b))
    assert torch.autograd.gradgradcheck(evaluate, (a, b))


def test_jax_streaming_jit_vmap_and_hessian():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    base, limited = plans()

    def objective(v, current, chunk):
        values = current.evaluate(
            {"a": v[:, None], "b": jnp.array([[0.1, 0.3, 0.4]])},
            backend="jax",
            like=v,
            batch_chunk_size=chunk,
        )
        return (values * values).sum()

    v = jnp.array([-0.2, 0.3])

    def scalar(x):
        return objective(x, limited, 2)

    def reference(x):
        return objective(x, base, None)

    np.testing.assert_allclose(jax.jit(scalar)(v), reference(v), atol=2e-6)
    np.testing.assert_allclose(jax.hessian(scalar)(v), jax.hessian(reference)(v), atol=5e-6)
