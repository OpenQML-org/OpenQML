"""Portable high-level prediction and optimizer/shuffle continuation contracts."""

import gc
import json
import sys
import zipfile

import numpy as np
import pytest

import openqml as oq


@pytest.fixture(autouse=True)
def release_compilations():
    yield
    if "jax" in sys.modules:
        sys.modules["jax"].clear_caches()
    gc.collect()


def _framework(framework):
    pytest.importorskip("torch" if framework == "torch" else "jax")
    if framework != "torch":
        pytest.importorskip("optax")
    if framework == "flax":
        pytest.importorskip("flax")


@pytest.mark.parametrize("framework", ["torch", "jax", "flax"])
@pytest.mark.parametrize("task", ["regression", "classification", "forecast"])
def test_checkpoint_restores_predictions_optimizer_history_and_next_shuffle(
    framework, task, tmp_path
):
    _framework(framework)
    x = np.random.default_rng(23).normal(size=(8, 2)).astype("float32") * 0.2
    y = x[:, 0] * 0.2 + 0.1
    if task == "classification":
        y = np.arange(8) % 2
    if task == "forecast":
        x = np.broadcast_to(x[:, None], (8, 3, 2)).copy()
        y = np.broadcast_to(y[:, None, None], (8, 3, 1)).copy() * 0.01
    backend = "torch" if framework == "torch" else "jax"
    model = oq.Model(task=task, framework=framework, hidden_features=4, depth=1, backend=backend)
    model.fit(x, y, epochs=2, batch_size=4, microbatch_size=2, shuffle=True)
    expected = model.predict(x)
    path = tmp_path / "saved.oqm"
    assert model.save(path) == path
    assert list(tmp_path.iterdir()) == [path]
    restored = oq.Model.load(path, device="cpu", backend=backend)
    assert restored.trainer.updates == model.trainer.updates == 4
    assert restored.history == model.history
    assert restored.architecture.fingerprint == model.architecture.fingerprint
    assert restored._rng.bit_generator.state == model._rng.bit_generator.state
    np.testing.assert_allclose(restored.predict(x), expected, atol=1e-6)
    for current in (model, restored):
        current.fit(x, y, batch_size=4, microbatch_size=2, shuffle=True)
    assert restored.trainer.updates == model.trainer.updates == 6
    if framework == "torch":
        for name, value in model.module.values.items():
            np.testing.assert_array_equal(value.detach(), restored.module.values[name].detach())
    else:
        for name, value in model.trainer.parameters.items():
            np.testing.assert_allclose(value, restored.trainer.parameters[name], atol=1e-7)
    np.testing.assert_allclose(restored.predict(x), model.predict(x), atol=1e-6)


def test_custom_architecture_and_atomic_save_failure(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    architecture = oq.HybridArchitecture(2, 1, hidden_features=4, depth=1, norm_epsilon=2e-5)
    model = oq.Model(architecture=architecture, backend="torch")
    x = [[0.1, 0.2], [0.2, -0.1]]
    model.fit(x, [0.1, 0.2])
    path = model.save(tmp_path / "nested" / "model.oqm")
    before = path.read_bytes()

    def broken(*_):
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(model.trainer, "state_dict", broken)
    with pytest.raises(OSError, match="interrupted"):
        model.save(path)
    assert path.read_bytes() == before
    loaded = oq.Model.load(path, backend="torch")
    assert loaded.architecture.fingerprint == architecture.fingerprint
    np.testing.assert_array_equal(model.predict(x), loaded.predict(x))


def test_interrupted_checkpoint_writer_keeps_old_bytes_and_removes_partial_file(tmp_path):
    from openqml.qml._archive import atomic_write

    path = tmp_path / "training.state"
    path.write_bytes(b"complete old checkpoint")

    def interrupted(stream):
        stream.write(b"partial replacement")
        raise OSError("interrupted during serialization")

    with pytest.raises(OSError, match="during serialization"):
        atomic_write(path, interrupted)
    assert path.read_bytes() == b"complete old checkpoint"
    assert list(tmp_path.iterdir()) == [path]
    atomic_write(path, lambda stream: stream.write(b"complete replacement"))
    assert path.read_bytes() == b"complete replacement"


def test_checkpoint_rejects_unfitted_malformed_oversized_and_traversal_archives(tmp_path):
    with pytest.raises(oq.ValidationError, match="fit"):
        oq.Model().save(tmp_path / "none")
    for name, content in [("garbage", b"unrecognized"), ("empty", b"")]:
        path = tmp_path / name
        path.write_bytes(content)
        with pytest.raises(oq.ValidationError):
            oq.Model.load(path)
    path = tmp_path / "bad.oqm"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("../escaped", "bad")
        archive.writestr("model.json", "{}")
    with pytest.raises(oq.ValidationError):
        oq.Model.load(path)
    assert not (tmp_path.parent / "escaped").exists()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("model.json", json.dumps({"format": "unknown"}))
        archive.writestr("training", "x" * 10000)
    with pytest.raises(oq.ResourceError):
        oq.Model.load(path, max_bytes=100)
    with pytest.raises(oq.ValidationError, match="format"):
        oq.Model.load(path)


def test_torch_model_load_uses_tensor_records_without_torch_pickle_loader(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    model = oq.Model(hidden_features=4, depth=1, backend="torch")
    inputs = [[0.1, 0.2], [0.2, -0.1]]
    model.fit(inputs, [0.1, 0.2])
    path = model.save(tmp_path / "model.oqm")

    def forbidden(*args, **kwargs):
        pytest.fail("Model.load must not deserialize Torch pickle objects")

    monkeypatch.setattr(torch, "load", forbidden)
    restored = oq.Model.load(path, backend="torch")
    np.testing.assert_array_equal(model.predict(inputs), restored.predict(inputs))
