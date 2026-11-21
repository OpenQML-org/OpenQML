import gc
import sys

import pytest


@pytest.fixture
def release_compilations():
    yield
    if "jax" in sys.modules:
        sys.modules["jax"].clear_caches()
    gc.collect()


@pytest.fixture
def fresh_kernel(monkeypatch):
    import openqml as oq

    monkeypatch.setattr(oq.compute, "_KERNEL", oq.ComputeKernel())


@pytest.fixture
def native_x64(release_compilations):
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield torch, jax, jax.numpy
    finally:
        jax.config.update("jax_enable_x64", previous)
