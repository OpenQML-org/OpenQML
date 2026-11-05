"""Validation loss, full-state selection and checkpoint continuation contracts."""

import gc
import sys

import numpy as np
import pytest

import openqml as oq


@pytest.fixture(autouse=True)
def release_compilations():
    yield
    if "jax" in sys.modules:
        sys.modules["jax"].clear_caches()
    gc.collect()


def make_model(framework, **options):
    pytest.importorskip("torch" if framework == "torch" else "jax")
    if framework != "torch":
        pytest.importorskip("optax")
    if framework == "flax":
        pytest.importorskip("flax")
    return oq.Model(
        framework=framework,
        backend="torch" if framework == "torch" else "jax",
        hidden_features=4,
        depth=1,
        lr=0.02,
        seed=13,
        **options,
    )


@pytest.mark.parametrize("framework", ["torch", "jax", "flax"])
def test_selection_restores_optimizer_counters_rng_and_resumes_exactly(framework, tmp_path):
    x = np.random.default_rng(2).normal(size=(6, 2)).astype("float32") * 0.2
    y = x[:, 0] * 0.3 + 0.1
    first, selected = make_model(framework), make_model(framework)
    fit = dict(batch_size=4, microbatch_size=2, shuffle=True, validation_data=(x, y))
    first.fit(x, y, **fit)
    selected.fit(x, y, epochs=10, patience=2, min_delta=1e6, **fit)
    assert len(selected.history) == 3
    assert selected.best_epoch == 1 and selected.stopped_epoch == 3
    assert selected.trainer.updates == first.trainer.updates == 2
    assert selected.trainer.attempts == first.trainer.attempts
    assert selected._rng.bit_generator.state == first._rng.bit_generator.state
    np.testing.assert_allclose(selected.predict(x), first.predict(x), atol=1e-7)
    assert selected.evaluate(x, y, batch_size=2) == pytest.approx(
        selected.history[0]["val_loss"], abs=1e-7
    )
    assert selected.evaluate(x, y) == pytest.approx(
        np.mean((selected.predict(x) - y) ** 2), abs=1e-7
    )
    restored = oq.Model.load(selected.save(tmp_path / "selected.oqm"), backend=selected.backend)
    assert restored.history == selected.history
    assert restored.best_epoch == 1 and restored.stopped_epoch == 3
    for model in (first, selected, restored):
        model.fit(x, y, batch_size=4, microbatch_size=2, shuffle=True)
    np.testing.assert_allclose(selected.predict(x), first.predict(x), atol=1e-7)
    np.testing.assert_allclose(restored.predict(x), first.predict(x), atol=1e-7)


@pytest.mark.parametrize("framework", ["torch", "jax", "flax"])
@pytest.mark.parametrize("task", ["classification", "forecast"])
def test_task_loss_uses_native_evaluation_without_updating_state(framework, task):
    model = make_model(framework, task=task)
    x = np.full((4, 2), 0.1, dtype="float32")
    y = np.array([0, 1, 0, 1])
    if task == "forecast":
        x, y = x[:, None, :], np.full((4, 1, 1), 0.001, dtype="float32")
    model.fit(x, y, validation_data=(x, y))
    updates = model.trainer.updates
    expected = model.history[0]["val_loss"]
    assert model.evaluate(x, y, batch_size=3) == pytest.approx(expected, abs=1e-7)
    assert model.trainer.updates == updates
    if task == "classification":
        p = model.predict_proba(x)
        # Native log-softmax and log(softmax) have different rounding paths.
        eps = np.finfo(p.dtype).eps
        reference = -np.log(p[np.arange(4), y].astype(np.float64)).mean()
        np.testing.assert_allclose(expected, reference, rtol=4 * eps, atol=4 * eps)
    if framework == "torch":
        model.module.eval()
        model.evaluate(x, y)
        assert not model.module.training


@pytest.mark.parametrize(
    "options",
    [
        {"patience": 2},
        {"patience": 0},
        {"patience": True},
        {"min_delta": -1},
        {"restore_best": 1},
        {"validation_data": ()},
        {"validation_data": ([[0, 1]], [float("nan")])},
        {"validation_data": ([[0, 1, 2]], [1])},
        {"validation_data": ([[0, 1]], [1, 2])},
    ],
)
def test_selection_preflight_rejects_invalid_options_before_build(options):
    model = oq.Model()
    with pytest.raises(oq.ValidationError):
        model.fit([[0, 1], [1, 0]], [0.1, 0.2], **options)
    assert model.trainer is None


def test_validation_only_and_restore_best_false_keep_all_updates():
    x, y = [[0.1, 0.2], [0.2, -0.1]], [0.1, 0.2]
    for options in ({}, {"patience": 1, "min_delta": 1e6, "restore_best": False}):
        model = make_model("torch")
        model.fit(x, y, epochs=3, validation_data=(x, y), **options)
        assert model.trainer.updates == len(model.history)
        assert all("val_loss" in row for row in model.history)
        if options:
            assert model.stopped_epoch == 2


def test_evaluate_requires_fit_and_valid_class_labels():
    with pytest.raises(oq.ValidationError, match="fit"):
        oq.Model().evaluate([[0, 1]], [0])
    model = make_model("torch", task="classification")
    model.fit([[0, 1], [1, 0]], [0, 1])
    for labels in ([0.5], [2], [-1]):
        with pytest.raises(oq.ValidationError):
            model.evaluate([[0, 1]], labels)


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_validation_respects_spatial_readout_pooling(task):
    pytest.importorskip("torch")
    architecture = oq.HybridArchitecture.convolutional(
        1,
        2 if task == "classification" else 1,
        spatial_dims=1,
        hidden_features=4,
        depth=1,
    )
    model = oq.Model(architecture=architecture, task=task, backend="torch")
    x = np.full((3, 4, 1), 0.1, dtype="float32")
    y = [0, 1, 0] if task == "classification" else [0.1, 0.2, 0.1]
    model.fit(x, y, validation_data=(x, y))
    assert np.isfinite(model.evaluate(x, y))


def test_empty_target_axes_raise_validation_error_before_label_reduction():
    with pytest.raises(oq.ValidationError):
        oq.Model(task="classification").fit(np.zeros((2, 3)), np.zeros((2, 0)))
