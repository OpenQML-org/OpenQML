"""Shared PEP 562 resolution without importing optional frameworks."""

from importlib import import_module

_DEPENDENCIES = {
    "torch": {"torch"},
    "jax": {"jax", "jaxlib"},
    "flax": {"flax", "jax", "jaxlib", "optax"},
    "optax": {"optax", "jax", "jaxlib"},
    "numpy": {"numpy"},
    "qiskit": {"qiskit"},
    "qiskit_aer": {"qiskit_aer", "qiskit"},
    "pennylane": {"pennylane"},
    "cirq": {"cirq"},
    "openqasm3": {"openqasm3"},
    "sympy": {"sympy"},
}


def resolve(name, package, exports, modules, namespace):
    if name in exports:
        module, symbol = exports[name]
    elif name in modules:
        module, symbol = modules[name], None
    else:
        raise AttributeError(f"module {package!r} has no attribute {name!r}")
    try:
        loaded = import_module(module, package)
    except ModuleNotFoundError as exc:
        expected = _DEPENDENCIES.get(name, ())
        if not expected or (exc.name is not None and exc.name not in expected):
            raise
        from .errors import MissingDependencyError
        from .runtime._dependencies import install_hint

        raise MissingDependencyError(f"Cannot load {name}. Run: {install_hint(name)}.") from exc
    value = loaded if symbol is None else getattr(loaded, symbol)
    namespace[name] = value
    return value
