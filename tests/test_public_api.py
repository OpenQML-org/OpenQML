"""A lightweight core and optional integrations share one lazy namespace."""

import importlib
import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import pytest

import openqml as oq


def test_package_version_matches_the_installed_distribution():
    assert oq.__version__ == version("openqml")


def test_core_import_and_discovery_do_not_eagerly_load_integrations():
    code = """
import importlib.abc
import sys

names = {
    "torch", "jax", "flax", "qiskit", "pennylane", "cirq", "qiskit_aer",
    "openqasm3", "optax", "sympy"
}

class BlockFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in names:
            raise ModuleNotFoundError(f"Blocked integration: {fullname}")

sys.meta_path.insert(0, BlockFrameworks())
import openqml as oq

assert not names.intersection(sys.modules)
assert names.issubset(dir(oq))
assert not names.intersection(oq.__dict__)
core = {}
exec("from openqml import *", core)
assert core["Circuit"] is oq.Circuit
assert not names.intersection(core)
circuit = oq.Circuit(1).h(0)
assert abs(oq.expectation(circuit, oq.PauliString("Z"))) < 1e-12
for name in names:
    try:
        getattr(oq, name)
    except oq.MissingDependencyError as error:
        assert 'python -m pip install "openqml[' in str(error)
    else:
        raise AssertionError(f"Missing dependency did not fail: {name}")
    assert name not in oq.__dict__
try:
    oq.not_a_public_api
except AttributeError as error:
    assert "not_a_public_api" in str(error)
else:
    raise AssertionError("Unknown attribute did not fail")
"""
    source = str(Path(oq.__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join((source, os.environ.get("PYTHONPATH", "")))}
    subprocess.run(
        [sys.executable, "-c", code], env=env, check=True, capture_output=True, text=True
    )


@pytest.mark.parametrize("framework", ["torch", "jax", "flax"])
def test_root_namespace_reuses_the_canonical_framework_module(framework):
    pytest.importorskip(framework)
    namespace = getattr(oq, framework)
    original = importlib.import_module(f"openqml.interfaces.{framework}")
    assert namespace is original
    assert callable(namespace.QuantumLayer) and callable(namespace.HybridNetwork)
    retired = {
        "PPOTrainer",
        "QuantumActorCritic",
        "QuantumSurrogate",
        "TreeQuantumLayer",
        "TensorNetworkAdapter",
        "TensorTrainLinear",
    }
    assert retired.isdisjoint(vars(namespace))
    assert getattr(oq, framework) is namespace


def test_default_distribution_is_small_and_extras_cover_all_integrations():
    from importlib.metadata import requires

    from packaging.requirements import Requirement

    requirements = [Requirement(value) for value in requires("openqml")]
    base = {r.name: r for r in requirements if r.marker is None or r.marker.evaluate({"extra": ""})}
    expected = {
        "numpy": "numpy",
        "torch": "torch",
        "jax": "jax",
        "flax": "flax",
        "optax": "optax",
        "qiskit": "qiskit",
        "qiskit-aer": "qiskit_aer",
        "pennylane": "pennylane",
        "cirq-core": "cirq",
        "openqasm3": "openqasm3",
        "sympy": "sympy",
    }
    assert set(base) == {"numpy", "packaging", "psutil"}
    full = {
        r.name: r for r in requirements if r.marker is None or r.marker.evaluate({"extra": "all"})
    }
    assert expected.keys() <= full.keys()
    assert "qasm3-import" in full["qiskit"].extras
    assert "parser" in full["openqasm3"].extras
    assert all(r.name != "openqml" for r in requirements)


@pytest.mark.parametrize(
    "name", ["numpy", "optax", "qiskit", "pennylane", "cirq", "qiskit_aer", "openqasm3", "sympy"]
)
def test_sdk_namespaces_preserve_canonical_module_identity(name):
    native = pytest.importorskip(name)
    assert getattr(oq, name) is native


def test_framework_facades_preserve_native_apis_and_quantum_layers():
    for name in ("torch", "jax", "flax"):
        pytest.importorskip(name)
    assert oq.torch.nn is importlib.import_module("torch.nn")
    assert oq.torch.optim is importlib.import_module("torch.optim")
    assert oq.torch.randn is importlib.import_module("torch").randn
    assert oq.torch.real is importlib.import_module("torch").real
    assert oq.jax.grad is importlib.import_module("jax").grad
    assert oq.jax.numpy is importlib.import_module("jax.numpy")
    assert oq.flax.linen is importlib.import_module("flax.linen")
    assert "randn" in dir(oq.torch) and "grad" in dir(oq.jax) and "linen" in dir(oq.flax)
    for name in ("torch", "jax", "flax"):
        assert "QuantumLayer" in dir(getattr(oq, name))


def test_readme_quickstart_completes_an_optimizer_step():
    import re

    pytest.importorskip("torch")
    readme = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")
    namespace = {}
    for block in re.findall(r"```python\n(.*?)```", readme, re.S):
        exec(compile(block, "README.md", "exec"), namespace)
    assert namespace["oq"] is oq and namespace["correlation"] == pytest.approx(1)
    model = namespace["model"]
    assert tuple(model.predict_proba(namespace["x"]).shape) == (32, 2)
    assert model.trainer.updates == sum(row["updates"] for row in model.history) >= 1
    assert oq.numpy.isfinite(model.history[-1]["loss"])
    assert any(state["exp_avg"].count_nonzero() for state in model.trainer.optimizer.state.values())


def test_single_import_supports_jax_flax_and_optax_training():
    for name in ("jax", "flax", "optax"):
        pytest.importorskip(name)
    spec = oq.data_reuploading(2, depth=1)
    layer = oq.jax.QuantumLayer.from_spec(spec)
    inputs = oq.jax.numpy.asarray([[0.2, -0.3], [0.1, 0.4]])
    key = oq.jax.random.key(7)
    weights = layer.init(key)
    optimizer = oq.optax.adam(0.01)
    state = optimizer.init(weights)
    gradients = oq.jax.jit(oq.jax.grad(lambda w: oq.jax.numpy.square(layer(w, inputs)).mean()))(
        weights
    )
    updates, _ = optimizer.update(gradients, state, weights)
    updated = oq.optax.apply_updates(weights, updates)
    assert oq.numpy.isfinite(gradients).all() and not oq.numpy.array_equal(updated, weights)
    module = oq.flax.QuantumLayer.from_spec(spec)
    variables = module.init(key, inputs)
    assert module.apply(variables, inputs).shape == (2, 2)
