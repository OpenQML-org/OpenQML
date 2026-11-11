"""Canonical package ownership and compatibility imports share one implementation."""

import importlib
import subprocess
import sys
from pathlib import Path

import openqml as oq


def test_namespaces_expose_the_same_public_objects():
    assert oq.ir.Circuit is oq.Circuit
    assert oq.ir.Parameter is oq.Parameter
    assert oq.ir.PauliString is oq.PauliString
    assert oq.sim.statevector is oq.statevector
    assert oq.sim.compile_circuit is oq.compile_circuit
    assert oq.autodiff.compile_compute is oq.compile_compute
    assert oq.qml.Model is oq.Model
    assert oq.interop.convert is oq.convert
    for group in ("ir", "sim", "compute", "runtime", "autodiff", "qml", "interop", "finance"):
        assert importlib.util.find_spec("openqml." + group).submodule_search_locations is not None
    root = Path(oq.__file__).parent
    assert not (root / "simulator.py").exists()
    assert not (root / "models.py").exists()
    assert not (root / "parameters.py").exists()


def test_parallel_legacy_and_canonical_imports_preserve_module_and_class_identity():
    code = """
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
import sys
import openqml
pairs = [
    ("openqml.simulator", "openqml.sim.dense"),
    ("openqml.models", "openqml.qml.models"),
    ("openqml.gradients", "openqml.autodiff.gradients"),
    ("openqml.adapters.qiskit", "openqml.interop.adapters.qiskit"),
]
assert not any(old in sys.modules for old, new in pairs)
with ThreadPoolExecutor(8) as pool:
    list(pool.map(import_module, [name for pair in pairs for name in pair] * 4))
for old, new in pairs:
    assert import_module(old) is import_module(new)
assert import_module("openqml.models").Model is openqml.Model
assert not {"torch", "jax", "qiskit", "cirq", "pennylane"}.intersection(sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=30)


def test_namespace_configuration_and_monkeypatches_have_one_owner(monkeypatch):
    import openqml.compute as compute
    from openqml.compute import kernel
    from openqml.ir import core

    original = core._validate_operation
    with monkeypatch.context() as patch:
        patch.setattr(oq.ir, "_validate_operation", lambda *args: original(*args))
        assert oq.ir._validate_operation is core._validate_operation
        oq.Circuit(1).x(0)
    assert core._validate_operation is original
    previous = compute._KERNEL
    with monkeypatch.context() as patch:
        replacement = oq.ComputeKernel()
        patch.setattr(kernel, "_KERNEL", replacement)
        assert compute._KERNEL is compute.get_compute_kernel() is replacement
    assert compute._KERNEL is previous
