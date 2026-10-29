"""Runtime dependency, isolated import and optional installed-footprint reports."""

import importlib.metadata as metadata
import os
import platform
import stat
import subprocess
import sys

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion

from ._dependencies import install_hint
from .errors import ValidationError

_INTEGRATIONS = {
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


def _probe_import(module, timeout):
    """A broken binary import cannot terminate or alter the caller's SDK state."""
    code = "import importlib; importlib.import_module(" + repr(module) + ")"
    checks = ["import"]
    if module == "torch":
        code += (
            "; import numpy as np; import torch; "
            "x=torch.tensor([1.0], requires_grad=True); x.square().sum().backward(); "
            "assert x.detach().cpu().numpy()[0] == 1; "
            "assert torch.from_numpy(np.ones(1)).shape == (1,)"
        )
        checks.append("numpy_array_bridge")
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        return {"status": "error", "reason": "import_timeout"}
    except OSError:
        return {"status": "error", "reason": "probe_launch_failed"}
    return (
        {"status": "ok", "checks": checks}
        if result.returncode == 0
        else {
            "status": "error",
            "reason": "runtime_probe_failed",
            "requested_checks": checks,
            "returncode": result.returncode,
            "detail": result.stderr[-2000:],
        }
    )


def environment_report(*, check_imports=False, include_sizes=False, import_timeout=60, extras=()):
    """Report core or explicitly selected extras; never install or update anything.

    Import probes run in separate processes. Optional byte counts cover logical
    files in the runtime dependency closure, deduplicated by resolved path;
    pip/build caches and unrelated installed packages are excluded. Metadata
    compatibility and successful imports are not a universal platform guarantee.
    """
    if type(check_imports) is not bool or type(include_sizes) is not bool:
        raise ValidationError("Report flags must be booleans.")
    if type(import_timeout) not in {int, float} or not 0 < import_timeout <= 600:
        raise ValidationError("import_timeout must be positive and at most 600 seconds.")
    if isinstance(extras, str) or not isinstance(extras, (tuple, list, set, frozenset)):
        raise ValidationError("extras must be a collection of installation extra names.")
    allowed = {
        "all",
        "torch",
        "jax",
        "flax",
        "qiskit",
        "aer",
        "qasm",
        "pennylane",
        "cirq",
        "symbolic",
        "dev",
    }
    if any(not isinstance(e, str) or e not in allowed for e in extras):
        raise ValidationError("Unknown installation extra in extras.")
    selected_extras = frozenset(extras)
    distributions, requested, dependencies, diagnostics = {}, {}, {}, []
    pending = [("openqml", selected_extras)]
    while pending:
        name, extras = pending.pop()
        key = canonicalize_name(name)
        previous = requested.get(key)
        if previous is not None and extras <= previous:
            continue
        requested[key] = extras | (previous or frozenset())
        if key not in distributions:
            try:
                distributions[key] = metadata.distribution(name)
            except metadata.PackageNotFoundError:
                distributions[key] = None
        distribution = distributions[key]
        if distribution is None:
            dependencies[key] = {"status": "missing"}
            diagnostics.append({"code": "DEPENDENCY_MISSING", "package": key, "severity": "error"})
            continue
        dependencies.setdefault(key, {"status": "installed", "version": distribution.version})
        for text in distribution.requires or ():
            try:
                requirement = Requirement(text)
            except InvalidRequirement:
                diagnostics.append(
                    {"code": "DEPENDENCY_METADATA_INVALID", "package": key, "severity": "error"}
                )
                continue
            if requirement.marker and not any(
                requirement.marker.evaluate({"extra": extra}) for extra in requested[key] | {""}
            ):
                continue
            child = canonicalize_name(requirement.name)
            if child not in distributions:
                try:
                    distributions[child] = metadata.distribution(requirement.name)
                except metadata.PackageNotFoundError:
                    distributions[child] = None
            installed = distributions[child]
            try:
                compatible = installed is None or requirement.specifier.contains(
                    installed.version, prereleases=True
                )
            except InvalidVersion:
                compatible = False
            if not compatible:
                dependencies[child] = {"status": "incompatible", "version": installed.version}
                diagnostics.append(
                    {
                        "code": "DEPENDENCY_VERSION_CONFLICT",
                        "package": child,
                        "required_by": key,
                        "specifier": str(requirement.specifier),
                        "severity": "error",
                    }
                )
            if requirement.url:
                diagnostics.append(
                    {
                        "code": "DIRECT_DEPENDENCY_URL_UNVERIFIED",
                        "package": child,
                        "severity": "warning",
                    }
                )
            pending.append((requirement.name, frozenset(requirement.extras)))
    integrations = {}
    for package, module in _INTEGRATIONS.items():
        entry = dependencies.get(package)
        required = entry is not None
        if entry is None:
            try:
                installed = metadata.distribution(package)
                entry = {"status": "installed", "version": installed.version}
            except metadata.PackageNotFoundError:
                entry = {"status": "not_installed"}
        integrations[module] = {
            **entry,
            "required": required,
            "install_command": install_hint(module),
        }
        if check_imports and entry["status"] == "installed":
            probe = _probe_import(module, import_timeout)
            integrations[module]["import"] = probe
            if probe["status"] != "ok":
                diagnostics.append(
                    {
                        "code": "INTEGRATION_RUNTIME_PROBE_FAILED",
                        "integration": module,
                        "severity": "error" if required else "warning",
                    }
                )
    footprint = None
    if include_sizes:
        seen, total, missing = set(), 0, 0
        for distribution in distributions.values():
            if distribution is None:
                continue
            for file in distribution.files or ():
                path = distribution.locate_file(file)
                key = os.path.normcase(str(path.resolve()))
                if key in seen:
                    continue
                seen.add(key)
                try:
                    info = path.stat()
                    if stat.S_ISREG(info.st_mode):
                        total += info.st_size
                except OSError:
                    missing += 1
        footprint = {
            "scope": "logical_runtime_dependency_files",
            "bytes": total,
            "file_paths_counted": len(seen),
            "missing_recorded_paths": missing,
            "excludes": ["pip_cache", "build_cache", "unrelated_distributions"],
        }
    probes = [entry["import"] for entry in integrations.values() if "import" in entry]
    probe_coverage = ["imports"] if probes else []
    if any("numpy_array_bridge" in p.get("checks", p.get("requested_checks", ())) for p in probes):
        probe_coverage.append("Torch_NumPy_bridge")
    return {
        "schema": "openqml.environment_report",
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "system": platform.system(),
        "machine": platform.machine(),
        "status": "error" if any(d["severity"] == "error" for d in diagnostics) else "ok",
        "scope": "observed_environment_not_universal_compatibility",
        "selected_extras": sorted(selected_extras),
        "isolated_import_checks": check_imports,
        "runtime_probe_coverage": probe_coverage,
        "accelerator_execution_checked": False,
        "dependency_count": len(dependencies),
        "dependencies": dict(sorted(dependencies.items())),
        "integrations": integrations,
        "installed_footprint": footprint,
        "diagnostics": diagnostics,
    }
