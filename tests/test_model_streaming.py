import sys

import numpy as np
import pytest

import openqml as oq
from openqml.models import _data


def test_memmap_and_strided_validation_use_bounded_temporaries(tmp_path, monkeypatch):
    x = np.memmap(tmp_path / "inputs.dat", dtype="float64", mode="w+", shape=(2048, 2048))
    x[:] = 1
    original = np.isfinite
    observed = []

    def finite(value, *args, **kwargs):
        observed.append(value.size)
        assert value.size * (2 * value.dtype.itemsize + 2) <= 2**20
        return original(value, *args, **kwargs)

    monkeypatch.setattr(np, "isfinite", finite)
    with oq.compute_context(policy=oq.ComputePolicy(max_memory_mb=1)):
        for value in (x, x.T, x[::2, ::-3]):
            assert np.shares_memory(_data(value, "inputs"), x)
        x[-1, -1] = np.nan
        with pytest.raises(oq.ValidationError, match="finite"):
            _data(x, "inputs")
    assert len(observed) > 20


def test_floating_labels_check_late_fraction_without_full_array_floor(tmp_path, monkeypatch):
    labels = np.memmap(tmp_path / "labels.dat", dtype="float64", mode="w+", shape=(500_000,))
    labels[:] = 0
    labels[-1] = 0.5
    inputs = np.broadcast_to([0.1, 0.2], (len(labels), 2))
    floor = np.floor

    def bounded_floor(value):
        assert value.size * (2 * value.dtype.itemsize + 2) <= 2**20
        return floor(value)

    monkeypatch.setattr(np, "floor", bounded_floor)
    model = oq.Model(task="classification")
    with oq.compute_context(policy=oq.ComputePolicy(max_memory_mb=1)):
        with pytest.raises(oq.ValidationError, match="integers"):
            model._check_targets(inputs, labels, 2, features=2)


@pytest.mark.parametrize("framework", ["torch", "jax", "flax"])
@pytest.mark.parametrize("task", ["regression", "classification", "forecast"])
def test_streaming_predictions_match_eager_task_outputs_and_restore_state(framework, task):
    pytest.importorskip("torch" if framework == "torch" else "jax")
    if framework != "torch":
        pytest.importorskip("optax")
    if framework == "flax":
        pytest.importorskip("flax")
    x = np.random.default_rng(7).normal(size=(5, 2)).astype("float32") * 0.1
    y = x[:, 0] * 0.2
    if task == "classification":
        y = np.array([0, 1, 0, 1, 0])
    elif task == "forecast":
        x, y = x[:, None, :], y[:, None, None]
    model = oq.Model(
        framework=framework,
        backend="torch" if framework == "torch" else "jax",
        task=task,
        hidden_features=4,
        depth=1,
    ).fit(x, y, batch_size=3)
    kernel = oq.get_compute_kernel()
    updates = model.trainer.updates
    stream = model.iter_predict(x, batch_size=2)
    first = next(stream)
    assert oq.get_compute_kernel() is kernel
    if framework == "torch":
        assert model.module.training
        assert sys.modules["torch"].is_grad_enabled()
    np.testing.assert_allclose(np.concatenate([first, *stream]), model.predict(x, batch_size=2))
    if task == "classification":
        streamed = np.concatenate(list(model.iter_predict(x, batch_size=2, output="probabilities")))
        np.testing.assert_allclose(streamed, model.predict_proba(x, batch_size=2))
    elif task == "forecast":
        streamed = np.concatenate(list(model.iter_predict(x, batch_size=2, output="quantiles")))
        np.testing.assert_allclose(streamed, model.predict_quantiles(x, batch_size=2))
    assert model.trainer.updates == updates
    stream = model.iter_predict(x, batch_size=2)
    next(stream)
    stream.close()
    assert oq.get_compute_kernel() is kernel
    if framework == "torch":
        assert model.module.training
    if framework != "torch":
        sys.modules["jax"].clear_caches()


def test_stream_validates_only_consumed_batches_and_rejects_bad_output():
    pytest.importorskip("torch")
    model = oq.Model(hidden_features=4, depth=1, backend="torch").fit(
        [[0.1, 0.2], [0.2, 0.1]], [0.3, 0.2]
    )
    x = np.array([[0.1, 0.2], [np.nan, 0.3]])
    stream = model.iter_predict(x, batch_size=1)
    assert next(stream).shape == (1,)
    with pytest.raises(oq.ValidationError, match="finite"):
        next(stream)
    assert model.module.training
    with pytest.raises(oq.ValidationError, match="output"):
        next(model.iter_predict(x, output="probabilities"))
    assert next(model.iter_predict(x[0])).shape == (1,)
