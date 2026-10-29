"""Dependency closure, ABI probes and footprint claims must reflect observed evidence."""

import json
from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace

import pytest

import openqml as oq
import openqml.environment as environment
from openqml.cli import main


def test_extra_markers_cycles_and_conflicts_are_resolved_without_dev_dependencies(monkeypatch):
    packages = {
        "openqml": SimpleNamespace(
            version="1", requires=["alpha[fast]>=1", "devtool; extra == 'dev'"], files=[]
        ),
        "alpha": SimpleNamespace(
            version="1.2", requires=["beta>=2; extra == 'fast'", "openqml>=1"], files=[]
        ),
        "beta": SimpleNamespace(version="1.5", requires=[], files=[]),
    }
    monkeypatch.setattr(environment.metadata, "distribution", lambda name: packages[name])
    monkeypatch.setattr(environment, "_INTEGRATIONS", {})
    report = oq.environment_report()
    assert set(report["dependencies"]) == {"openqml", "alpha", "beta"}
    assert report["dependencies"]["beta"]["status"] == "incompatible"
    assert (
        report["status"] == "error"
        and report["diagnostics"][0]["code"] == "DEPENDENCY_VERSION_CONFLICT"
    )


def test_missing_packages_are_diagnosed(monkeypatch):
    def missing(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(environment.metadata, "distribution", missing)
    monkeypatch.setattr(environment, "_INTEGRATIONS", {})
    report = oq.environment_report()
    assert report["status"] == "error" and report["dependencies"]["openqml"]["status"] == "missing"


def test_footprint_counts_dependency_files_once_and_excludes_caches(monkeypatch, tmp_path):
    file = tmp_path / "runtime.bin"
    file.write_bytes(b"1234567")
    package = SimpleNamespace(
        version="1", requires=[], files=[file.name, file.name], locate_file=lambda _: file
    )
    monkeypatch.setattr(environment.metadata, "distribution", lambda _: package)
    monkeypatch.setattr(environment, "_INTEGRATIONS", {})
    report = oq.environment_report(include_sizes=True)
    assert report["installed_footprint"]["bytes"] == 7
    assert report["installed_footprint"]["file_paths_counted"] == 1
    assert "pip_cache" in report["installed_footprint"]["excludes"]


def test_isolated_probe_contains_import_failure_and_checks_real_numpy_bridge(monkeypatch, tmp_path):
    module = tmp_path / "broken_sdk.py"
    module.write_text("import os\nos._exit(7)\n")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    result = environment._probe_import("broken_sdk", 10)
    assert result["status"] == "error" and result["returncode"] == 7
    if environment.metadata.packages_distributions().get("torch"):
        assert environment._probe_import("torch", 60)["checks"] == ["import", "numpy_array_bridge"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"check_imports": 1},
        {"include_sizes": "yes"},
        {"import_timeout": 0},
        {"import_timeout": float("nan")},
    ],
)
def test_environment_options_are_validated(kwargs):
    with pytest.raises(oq.ValidationError):
        oq.environment_report(**kwargs)


def test_observed_environment_and_report_cli_are_machine_readable(tmp_path):
    report = oq.environment_report()
    assert report["status"] == "ok" and report["dependency_count"] == 4
    assert report["scope"] == "observed_environment_not_universal_compatibility"
    target = tmp_path / "environment.json"
    assert main(["doctor", "--report", "--output", str(target)]) == 0
    torch = json.loads(target.read_text())["integrations"]["torch"]
    assert not torch["required"]
    assert "openqml[torch]" in torch["install_command"]
    assert main(["doctor", "--report", "--output", str(target)]) == 2


def test_missing_optional_integrations_are_healthy_and_selected_extras_are_required(monkeypatch):
    packages = {
        "openqml": SimpleNamespace(
            version="1", requires=["numpy", "torch; extra == 'torch'"], files=[]
        ),
        "numpy": SimpleNamespace(version="1", requires=[], files=[]),
    }

    def distribution(name):
        if name not in packages:
            raise PackageNotFoundError(name)
        return packages[name]

    monkeypatch.setattr(environment.metadata, "distribution", distribution)
    monkeypatch.setattr(environment, "_INTEGRATIONS", {"numpy": "numpy", "torch": "torch"})
    core = oq.environment_report()
    assert core["status"] == "ok"
    assert core["integrations"]["torch"]["status"] == "not_installed"
    assert not core["integrations"]["torch"]["required"]
    monkeypatch.setattr(
        environment, "_probe_import", lambda module, timeout: {"status": "ok", "checks": ["import"]}
    )
    assert oq.environment_report(check_imports=True)["runtime_probe_coverage"] == ["imports"]
    selected = oq.environment_report(extras=("torch",))
    assert selected["status"] == "error"
    assert selected["integrations"]["torch"]["required"]
    assert selected["selected_extras"] == ["torch"]


@pytest.mark.parametrize("extras", ["all", ["unknown"], [False]])
def test_invalid_extra_selections_are_rejected(extras):
    with pytest.raises(oq.ValidationError):
        oq.environment_report(extras=extras)
