"""One default installation and one import expose all integrations lazily."""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

import openqml as oq


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
        assert "python -m pip install openqml" in str(error)
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
    assert namespace.QuantumLayer is original.QuantumLayer
    assert getattr(oq, framework) is namespace


def test_default_distribution_requires_every_supported_integration():
    from importlib.metadata import requires
    from importlib.util import find_spec

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
    assert expected.keys() <= base.keys()
    assert "qasm3-import" in base["qiskit"].extras
    assert "parser" in base["openqasm3"].extras
    assert all(find_spec(module) is not None for module in expected.values())
    assert all(r.name != "openqml" for r in requirements)


@pytest.mark.parametrize(
    "name", ["numpy", "optax", "qiskit", "pennylane", "cirq", "qiskit_aer", "openqasm3", "sympy"]
)
def test_sdk_namespaces_preserve_canonical_module_identity(name):
    native = importlib.import_module(name)
    assert getattr(oq, name) is native
    assert getattr(oq, name) is native


def test_framework_facades_preserve_native_apis_and_quantum_layers():
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


def test_readme_examples_need_only_the_openqml_import():
    import re

    text = (Path(__file__).parents[1] / "README.md").read_text()
    blocks = re.findall(r"```python\n(.*?)```", text, re.S)
    assert blocks and "openqml[" not in text
    namespace = {}
    for block in blocks:
        assert [line for line in block.splitlines() if line.startswith(("import ", "from "))] == [
            "import openqml as oq"
        ]
        exec(compile(block, "README.md", "exec"), namespace)
    model = namespace["model"]
    assert all(p.grad is not None and oq.torch.isfinite(p.grad).all() for p in model.parameters())


def test_single_import_supports_jax_flax_and_optax_training():
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


def test_single_import_supports_native_sdk_programs_and_conversion():
    program = oq.qiskit.QuantumCircuit(2)
    program.h(0)
    program.cx(0, 1)
    circuit = oq.convert(program, target="ir").value
    for target in ("qiskit", "cirq", "pennylane", "openqasm2", "openqasm3"):
        native = oq.convert(circuit, target=target).value
        restored = oq.convert(native, source=target, target="ir").value
        oq.numpy.testing.assert_allclose(
            oq.statevector(restored), oq.statevector(circuit), atol=1e-12
        )
