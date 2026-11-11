"""Public entry-point, native training and certified warm-start contracts."""

import gc
import importlib
import pickle
import subprocess
import sys

import numpy as np
import pytest

import openqml as oq


def test_bare_import_and_classical_finance_do_not_load_engines():
    source = """
import sys
import openqml as oq
assert not {"numpy", "psutil", "packaging", "torch", "jax", "flax"} & set(sys.modules)
assert len([m for m in sys.modules if m.startswith("openqml")]) <= 4
assert {"Model", "Circuit", "finance", "torch", "jax", "flax"} <= set(dir(oq))
r = oq.finance.allocate([[.01,-.02],[-.01,.03],[.02,.01]])
assert abs(sum(r.weights)-1)<1e-10
p = oq.finance.price([0,10,25], probabilities=[.2,.5,.3], discount=.98)
assert abs(p.value-12.25)<1e-12 and p.oracle_queries==0
engines={"torch","jax","flax","openqml.simulator","openqml.backends","openqml.amplitude",
         "openqml.sim.dense","openqml.runtime.backends","openqml.sim.amplitude"}
assert not engines & set(sys.modules)
"""
    subprocess.run([sys.executable, "-c", source], check=True)


def test_lazy_reexports_preserve_symbol_identity_and_old_result_pickle_path():
    from openqml._exports import EXPORTS

    for name, (module, symbol) in EXPORTS.items():
        assert getattr(oq, name) is getattr(importlib.import_module(module, "openqml"), symbol)
    assert oq.Circuit is oq.ir.Circuit
    assert pickle.loads(b"copenqml.finance.pricing\nPayoffEstimate\n.") is oq.finance.PayoffEstimate
    assert oq.finance.PayoffEstimate is oq.finance.pricing.PayoffEstimate


def test_missing_framework_is_deferred_until_fit_and_never_installed():
    source = """
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name.split(".")[0]=="torch":
            raise ModuleNotFoundError("test blocker",name="torch")
sys.meta_path.insert(0,Block())
import openqml as oq
model=oq.Model(2,1)
assert "torch" not in sys.modules
try:
    model.fit([[0.,1.]],[.2])
except oq.MissingDependencyError as e:
    assert "openqml[torch]" in str(e)
else:
    raise AssertionError("missing framework was not diagnosed")
"""
    subprocess.run([sys.executable, "-c", source], check=True)


def test_lazy_loader_preserves_unrelated_import_failures(monkeypatch):
    from openqml import _lazy

    def fail(*_):
        raise ModuleNotFoundError("broken internal module", name="internal_bug")

    monkeypatch.setattr(_lazy, "import_module", fail)
    with pytest.raises(ModuleNotFoundError, match="broken internal"):
        _lazy.resolve("torch", "openqml", {}, {"torch": ".interfaces.torch"}, {})


@pytest.fixture
def native():
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    pytest.importorskip("optax")
    try:
        yield torch, jax
    finally:
        jax.clear_caches()
        gc.collect()


@pytest.mark.parametrize("framework", ["torch", "jax", "flax"])
def test_model_fit_predict_learns_with_native_optimizer_and_reuses_state(framework, native):
    torch, jax = native
    if framework == "flax":
        pytest.importorskip("flax")
    rng = np.random.default_rng(7)
    x = rng.normal(size=(16, 2)).astype(np.float32) * 0.3
    y = 0.3 * x[:, 0] - 0.2 * x[:, 1] + 0.1
    model = oq.Model(
        hidden_features=4,
        depth=1,
        framework=framework,
        backend="torch" if framework == "torch" else "jax",
        lr=0.03,
    )
    assert model.fit(x, y, batch_size=16) is model
    first = model.history[0]["loss"]
    trainer = model.trainer
    model.fit(x, y, epochs=12, batch_size=16, microbatch_size=8)
    assert model.trainer is trainer and trainer.updates == 13
    assert model.history[-1]["loss"] < first * 0.65
    prediction = model.predict(x, batch_size=3)
    assert isinstance(prediction, np.ndarray) and prediction.shape == (16,)
    np.testing.assert_allclose(prediction, model.predict(x), rtol=1e-5, atol=1e-6)
    if framework == "torch":
        assert isinstance(model.module, torch.nn.Module)
        assert model.trainer.optimizer.state
    else:
        assert model.trainer.parameters
        if framework == "flax":
            np.testing.assert_allclose(
                model.module.apply({"params": trainer.parameters}, jax.numpy.asarray(x))[..., 0],
                prediction,
                atol=1e-6,
            )


@pytest.mark.parametrize("framework", ["torch", "jax", "flax"])
def test_model_classification_and_forecast_contracts(framework, native):
    if framework == "flax":
        pytest.importorskip("flax")
    backend = "torch" if framework == "torch" else "jax"
    x = np.array([[-0.3, 0], [0.3, 0], [0, -0.3], [0, 0.3]], np.float32)
    model = oq.Model(
        task="classification", framework=framework, backend=backend, hidden_features=4, depth=1
    )
    model.fit(x, [0, 1, 0, 1], epochs=2)
    p = model.predict_proba(x)
    np.testing.assert_allclose(p.sum(-1), 1, atol=1e-6)
    np.testing.assert_array_equal(model.predict(x), p.argmax(-1))
    assert np.asarray(model.predict(x[0])).shape == ()
    forecast = oq.Model(
        task="forecast", framework=framework, backend=backend, hidden_features=4, depth=1
    )
    inputs = np.broadcast_to(x[None], (2, 4, 2)).copy()
    target = np.full((2, 4, 2), 0.001, np.float32)
    forecast.fit(inputs, target)
    q = forecast.predict_quantiles(inputs)
    assert q.shape == (2, 4, 2, 5) and np.all(q[..., 1:] >= q[..., :-1])
    np.testing.assert_array_equal(forecast.predict(inputs), q[..., 2])


def test_model_validates_labels_shapes_and_prefit_predictions():
    with pytest.raises(oq.ValidationError):
        oq.Model(task="unknown")
    with pytest.raises(oq.ValidationError):
        oq.Model().predict([[0, 1]])
    with pytest.raises(oq.ValidationError):
        oq.Model(task="forecast").predict_quantiles([[0, 1]])
    with pytest.raises(oq.ValidationError):
        oq.Model().fit([[0, 1]], [float("nan")])
    with pytest.raises(oq.ValidationError):
        oq.Model(task="classification").fit([[0, 1]], [0.5])
    with pytest.raises(oq.ValidationError):
        oq.Model(task="classification").fit([[0, 1]], [-1])


def test_certified_warm_face_bypasses_primal_iterations_and_falls_back(monkeypatch):
    f = oq.finance
    rng = np.random.default_rng(45)
    risk = f.factor_risk(rng.normal(size=(6, 8)) * 0.01)
    previous = np.ones(8) / 8
    mean = rng.normal(size=8) * 1e-4
    plan = f.AllocationPlan(turnover=0.1)
    initial = plan.solve(mean, risk, previous=previous, caps=0.6)
    expected = plan.solve(mean + 0.00001, risk, previous=previous, caps=0.6)
    original, calls = f.AllocationPlan._identify, []

    def identify(self, *args):
        calls.append(1)
        return original(self, *args)

    monkeypatch.setattr(f.AllocationPlan, "_identify", identify)
    actual = plan.solve(
        mean + 0.00001, risk, previous=previous, caps=0.6, warm_start=initial.weights
    )
    assert not calls
    np.testing.assert_allclose(actual.weights, expected.weights, atol=1e-12)
    bad = np.zeros(8)
    bad[0] = 1
    recovered = plan.solve(mean + 0.00001, risk, previous=previous, caps=0.6, warm_start=bad)
    assert calls
    np.testing.assert_allclose(recovered.weights, expected.weights, atol=1e-12)


def test_warm_solve_native_values_and_gradients_match_cold(native):
    torch, jax = native
    jnp = jax.numpy
    risk = oq.finance.factor_risk(np.random.default_rng(3).normal(size=(6, 5)) * 0.01)
    plan = oq.finance.AllocationPlan(turnover=0.1)
    previous, mean = np.ones(5) / 5, np.arange(5) * 0.0001
    warm = plan.solve(mean, risk, previous=previous).weights
    gradients = []
    for seed in (None, warm):
        mu = torch.tensor(mean, requires_grad=True)
        w = plan.solve(mu, risk, previous=previous, warm_start=seed).weights
        gradients.append(torch.autograd.grad(w.square().sum(), mu)[0].numpy())
    np.testing.assert_allclose(*gradients, atol=1e-10)

    def objective(mu, seed):
        w = plan.solve(mu, risk, previous=previous, warm_start=seed).weights
        return jnp.sum(w * w)

    cold = jax.jit(jax.grad(lambda mu: objective(mu, None)))(jnp.asarray(mean))
    hot = jax.jit(jax.grad(lambda mu: objective(mu, jnp.asarray(warm))))(jnp.asarray(mean))
    np.testing.assert_allclose(cold, hot, atol=2e-6)


def test_native_financial_shortcuts_share_low_level_results(native):
    torch, jax = native
    values = np.array([[0.01, -0.02], [-0.01, 0.03], [0.02, 0.01]])
    direct = oq.finance.allocate(values)
    risk = oq.finance.factor_risk(values)
    expected = oq.finance.AllocationPlan().solve(risk.mean, risk)
    np.testing.assert_array_equal(direct.weights, expected.weights)
    for xp in (torch, jax.numpy):
        p = xp.asarray([0.2, 0.5, 0.3])
        result = oq.finance.price([0, 10, 25], probabilities=p, discount=0.98)
        assert float(result.value) == pytest.approx(12.25, abs=2e-6) and result.oracle_queries == 0
    with pytest.raises(oq.ValidationError):
        oq.finance.allocate(values, risk=risk)
    with pytest.raises(oq.ValidationError):
        oq.finance.allocate(values, plan=oq.finance.AllocationPlan(), ridge=0.1)
