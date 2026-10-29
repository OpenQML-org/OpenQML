"""The single-import API keeps optional frameworks lazy and preserves their classes."""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

import openqml as oq


def test_core_import_and_discovery_do_not_require_optional_frameworks():
    code = """
import importlib.abc
import sys

names = {"torch", "jax", "flax"}

class BlockFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in names:
            raise ModuleNotFoundError(f"Blocked optional framework: {fullname}")

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
        assert f"openqml[{name}]" in str(error)
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
