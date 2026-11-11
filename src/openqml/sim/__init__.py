"""Sim operations over the shared OpenQML representation and compute service."""

from .._lazy import resolve as _resolve

_EXPORTS = {
    "statevector": (".dense", "statevector"),
    "expectation": (".dense", "expectation"),
    "expectations": (".dense", "expectations"),
    "execute": (".dense", "execute"),
    "sample": (".dense", "sample"),
    "density_matrix": (".density", "density_matrix"),
    "execute_batch": (".batch", "execute_batch"),
    "expectations_batch": (".batch", "expectations_batch"),
    "compile_circuit": (".compiled", "compile_circuit"),
    "plan_execution": (".adaptive", "plan_execution"),
    "ReadoutMitigation": (".mitigation", "ReadoutMitigation"),
    "zero_noise_extrapolate": (".mitigation", "zero_noise_extrapolate"),
}
_MODULES = {}
__all__ = tuple(_EXPORTS)


def __getattr__(name):
    return _resolve(name, __name__, _EXPORTS, _MODULES, globals())


def __dir__():
    return sorted(set(globals()) | _EXPORTS.keys())
