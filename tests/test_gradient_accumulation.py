import numpy as np
import pytest

import openqml as oq


def test_torch_nonfinite_rejection_and_optimizer_rng_resume(tmp_path):
    torch = pytest.importorskip("torch")
    torch.manual_seed(9)
    model = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Dropout(0.3), torch.nn.Linear(4, 1))
    trainer = oq.torch.HybridTrainer(
        model, torch.optim.Adam(model.parameters(), lr=0.01), microbatch_size=2
    )
    batch = torch.randn(5, 2)

    def objective(m, b):
        return m(b).square().mean()

    trainer.step(batch, objective)
    trainer.save(tmp_path / "training.pt")
    expected = trainer.step(batch, objective)
    weights = [p.detach().clone() for p in model.parameters()]
    trainer.load(tmp_path / "training.pt")
    resumed = trainer.step(batch, objective)
    assert resumed["loss"] == expected["loss"]
    for p, value in zip(model.parameters(), weights, strict=True):
        torch.testing.assert_close(p, value, atol=0, rtol=0)
    before = [p.detach().clone() for p in model.parameters()]
    result = trainer.step(batch, lambda m, b: m(b).mean() * float("nan"))
    assert result["skipped"] and result["updates"] == 2
    for p, value in zip(model.parameters(), before, strict=True):
        torch.testing.assert_close(p, value, atol=0, rtol=0)


def test_jax_scan_weighted_gradients_and_auxiliary_metrics():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    optax = pytest.importorskip("optax")
    params = {"w": jnp.asarray([0.2, -0.3])}
    batch = {"x": jnp.arange(14, dtype=jnp.float32).reshape(7, 2) / 10, "y": jnp.arange(7) / 7}

    def loss(p, b):
        return (b["x"] @ p["w"] - b["y"]) ** 2

    value, gradient = jax.value_and_grad(lambda p: loss(p, batch).mean())(params)
    for size in (1, 3, 20):
        actual, grad, count = jax.jit(
            lambda p, b, size=size: oq.jax.accumulate_gradients(loss, p, b, microbatch_size=size)
        )(params, batch)
        assert int(count) == 7
        np.testing.assert_allclose(actual, value, rtol=2e-6)
        np.testing.assert_allclose(grad["w"], gradient["w"], rtol=2e-6)
    optimizer = optax.sgd(0.1)

    def auxiliary(p, b):
        return loss(p, b), {"target": b["y"], "constant": 0.25}

    step = oq.jax.make_train_step(
        optimizer, auxiliary, microbatch_size=3, max_grad_norm=None, has_aux=True
    )
    updated, _, metrics = step(params, optimizer.init(params), batch)
    np.testing.assert_allclose(updated["w"], params["w"] - 0.1 * gradient["w"], rtol=1e-6)
    np.testing.assert_allclose(metrics["target"], batch["y"].mean(), rtol=1e-6)
    assert float(metrics["constant"]) == 0.25
    bad = oq.jax.make_train_step(optimizer, lambda p, b: loss(p, b) * jnp.nan)
    updated, _, metrics = bad(params, optimizer.init(params), batch)
    assert bool(metrics["skipped"])
    np.testing.assert_array_equal(updated["w"], params["w"])
