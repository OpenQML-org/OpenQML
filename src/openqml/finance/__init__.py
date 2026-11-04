"""Native financial models, risk, allocation, accounting and coherent estimation."""

from .._lazy import resolve as _resolve
from ._exports import EXPORTS as _EXPORTS
from ._exports import MODULES as _MODULES
from ._exports import PUBLIC as _PUBLIC

__all__ = _PUBLIC


def __getattr__(name):
    return _resolve(name, __name__, _EXPORTS, _MODULES, globals())


def __dir__():
    return sorted(set(globals()) | _EXPORTS.keys() | _MODULES.keys())
