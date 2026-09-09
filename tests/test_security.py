"""A published flow and a downloaded dataset are data from strangers.

Reconstructing a flow imports code and a dataset writes to disk, so both are
places where a payload gets to decide what the library does. These pin the
boundary.
"""

import pathlib

import pytest

import openqml
from openqml.exceptions import OpenQMLError


# -- flows: reconstructing a model must not become "call anything" -----------

def test_a_flow_cannot_import_a_module_outside_the_allowlist():
    flow = openqml.OpenQMLFlow(name="evil", model_class="os.system",
                               parameters={"command": "id"})
    with pytest.raises(OpenQMLError, match="refusing to import"):
        flow.to_model()


def test_a_flow_cannot_call_a_plain_function_in_an_allowed_module():
    """The allowlist gates the module; the class check gates the callable.

    ``openqml.config.set_server`` passes the prefix test, so without this a
    flow could silently repoint the whole client at another server.
    """
    flow = openqml.OpenQMLFlow(name="gadget", model_class="openqml.config.set_server",
                               parameters={"url": "http://example.invalid/api"})
    with pytest.raises(OpenQMLError, match="refusing to call"):
        flow.to_model()
    assert openqml.config.server == "local"


def test_a_flow_cannot_smuggle_allowed_prefixes_through_its_parameters():
    """The trampoline: one flow re-entering flow_to_model to disable the check.

    ``openqml.flows.functions`` is inside the allowlist, so a flow naming
    ``flow_to_model`` with ``allowed_prefixes=None`` in its parameters used to
    rebuild a second flow with no allowlist at all -- arbitrary code execution
    from an ordinary-looking ``openqml.flow_to_model(id)``.
    """
    payload = openqml.OpenQMLFlow(
        name="payload", model_class="subprocess.Popen",
        parameters={"args": "true", "shell": True}).publish()
    trampoline = openqml.OpenQMLFlow(
        name="trampoline", model_class="openqml.flows.functions.flow_to_model",
        parameters={"flow": payload.id, "allowed_prefixes": None}).publish()

    with pytest.raises(OpenQMLError):
        openqml.flow_to_model(trampoline.id)


def test_the_documented_opt_in_still_reconstructs_a_flow():
    """``allowed_prefixes=None`` is the escape hatch the error message offers."""
    flow = openqml.OpenQMLFlow(name="ok", model_class="datetime.date",
                               parameters={"year": 2020, "month": 1, "day": 2})
    with pytest.raises(OpenQMLError, match="refusing to import"):
        flow.to_model()
    assert flow.to_model(allowed_prefixes=None).isoformat() == "2020-01-02"


def test_an_ordinary_flow_still_round_trips():
    model = openqml.QuantumKernelClassifier(feature_map="zz", n_qubits=2)
    rebuilt = openqml.flow_to_model(openqml.model_to_flow(model))
    assert type(rebuilt) is type(model)
    assert rebuilt.get_params() == model.get_params()


# -- datasets: an id from a payload must not choose where we write ------------

@pytest.mark.parametrize("bad", ["/tmp/openqml-escape", "../../../../tmp/openqml-escape",
                                 "..", ".", "a/b", ""])
def test_a_cache_path_component_cannot_be_a_path(bad):
    with pytest.raises(ValueError, match="invalid cache path component"):
        openqml.config.get_cache_directory("datasets", bad)


def test_a_dataset_id_cannot_escape_the_cache_directory(tmp_path):
    """``Path.joinpath`` treats an absolute part as a new root, so an id like
    ``/tmp/x`` used to discard the cache root entirely and write there."""
    escape = tmp_path / "escaped"
    dataset = openqml.OpenQMLDataset.from_dict({
        "id": str(escape), "name": "evil", "data_type": "tabular",
        "generator": {"function": "make_moons", "kwargs": {"n_samples": 8, "seed": 0}},
    })
    with pytest.raises(ValueError, match="invalid cache path component"):
        dataset.get_data()
    assert not escape.exists()


def test_a_normal_dataset_still_caches_under_the_cache_root():
    dataset = openqml.get_dataset("moons-2d")
    X, _, _ = dataset.get_data(target="class")
    assert X.shape == (200, 2)
    root = openqml.config.get_root_cache_directory().resolve()
    assert root in pathlib.Path(dataset._cache_path()).resolve().parents
