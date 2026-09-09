"""The top-level namespace is a promise: pin it.

Every name a user is told to import has to keep resolving, and it has to be the
same object as the one in the submodule it came from -- a re-export that drifts
is worse than no re-export at all.
"""

import importlib
import subprocess
import sys

import pytest

import openqml

SUBMODULES = [
    "backends", "catalog", "circuits", "datasets", "evaluations", "exceptions",
    "flows", "hamiltonians", "metrics", "models", "runs", "study", "tasks", "utils",
]


def test_everything_in_all_resolves():
    missing = [name for name in openqml.__all__ if not hasattr(openqml, name)]
    assert missing == []


def test_all_has_no_duplicates():
    assert len(openqml.__all__) == len(set(openqml.__all__))


def test_star_import_is_clean():
    namespace = {}
    exec("from openqml import *", namespace)  # noqa: S102 - that is the thing under test
    exported = {k for k in namespace if not k.startswith("__")}
    assert exported == set(openqml.__all__) - {"__version__"}


@pytest.mark.parametrize("name", SUBMODULES)
def test_submodules_are_importable_and_attached(name):
    module = importlib.import_module(f"openqml.{name}")
    assert getattr(openqml, name) is module


def test_reexports_are_the_same_objects():
    """A top-level name must be the submodule's object, not a copy of it."""
    for name in openqml.__all__:
        if name in SUBMODULES or name.startswith("__"):
            continue
        top = getattr(openqml, name)
        for source in SUBMODULES:
            module = getattr(openqml, source)
            if name in getattr(module, "__all__", []):
                assert getattr(module, name) is top, f"{name} drifted from openqml.{source}"


def test_every_built_in_model_is_importable_from_the_top():
    for name in openqml.list_models():
        assert getattr(openqml, name) is getattr(openqml.models, name)


def test_documented_quickstart_imports_work():
    from openqml import QuantumKernelClassifier, get_task, run_model_on_task  # noqa: F401

    run = run_model_on_task(QuantumKernelClassifier(feature_map="zz", n_qubits=2), get_task(1))
    assert run.summary()


def test_clear_state_cache_is_reachable():
    """The README documents it; it has to exist in both places."""
    assert openqml.clear_state_cache is openqml.models.clear_state_cache
    openqml.clear_state_cache()


def test_package_imports_without_optional_dependencies():
    """pandas and PennyLane are optional -- importing must not need either."""
    code = (
        "import sys;"
        "sys.modules['pandas'] = None;"
        "sys.modules['pennylane'] = None;"
        "import openqml;"
        "print(openqml.list_datasets(output_format='records')[0]['name'])"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "moons-2d"
