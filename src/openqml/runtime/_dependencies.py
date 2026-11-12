"""Lazy optional imports and installation advice; never runtime installation."""

from importlib import import_module

from ..errors import MissingDependencyError

INTEGRATION_EXTRAS = {
    "torch": "torch",
    "jax": "jax",
    "flax": "flax",
    "optax": "jax",
    "qiskit": "qiskit",
    "qiskit_aer": "aer",
    "pennylane": "pennylane",
    "cirq": "cirq",
    "openqasm3": "qasm",
    "sympy": "symbolic",
    "cotengra": "compute",
}


def install_hint(name):
    extra = INTEGRATION_EXTRAS.get(name, name)
    package = "openqml" if extra == "numpy" else f"openqml[{extra}]"
    return f'python -m pip install "{package}"'


def load_dependency(module, extra):
    try:
        return import_module(module)
    except ImportError as exc:
        missing = getattr(exc, "name", None)
        detail = (
            f" Dependency {missing!r} is missing."
            if missing and missing.split(".")[0] != module.split(".")[0]
            else f" Import failed: {exc}."
            if not isinstance(exc, ModuleNotFoundError)
            else ""
        )
        raise MissingDependencyError(
            f"Cannot load {module}.{detail} Run: {install_hint(extra)}."
        ) from exc
