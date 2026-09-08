import tempfile

import pytest

import openqml


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path_factory):
    """Every test runs against a fresh cache and a fresh local store."""
    openqml.config.set_root_cache_directory(tmp_path_factory.mktemp("openqml"))
    openqml.config.use_local_server()
    openqml.config.avoid_duplicate_runs = True
    yield
