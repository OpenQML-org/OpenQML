"""Interop operations over the shared OpenQML representation and compute service."""

from .._lazy import resolve as _resolve

_EXPORTS = {
    "convert": (".conversion", "convert"),
    "diagnose": (".conversion", "diagnose"),
    "plan_conversion": (".conversion", "plan_conversion"),
    "compile_program": (".program", "compile_program"),
    "compile_verified": (".verification", "compile_verified"),
    "prove_equivalence": (".verification", "prove_equivalence"),
    "inspect_python": (".inspection", "inspect_python"),
    "register_adapter": (".frontends", "register_adapter"),
    "list_formats": (".frontends", "list_formats"),
    "QiskitEstimatorBackend": (".providers", "QiskitEstimatorBackend"),
    "QiskitSamplerBackend": (".providers", "QiskitSamplerBackend"),
}
_MODULES = {}
__all__ = tuple(_EXPORTS)


def __getattr__(name):
    return _resolve(name, __name__, _EXPORTS, _MODULES, globals())


def __dir__():
    return sorted(set(globals()) | _EXPORTS.keys())
