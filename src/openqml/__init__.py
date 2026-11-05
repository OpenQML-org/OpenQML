"""Quantum-classical learning, finance and circuit tools from one lazy namespace."""

from ._exports import EXPORTS as _EXPORTS
from ._exports import MODULES as _MODULES
from ._exports import PUBLIC as _PUBLIC
from ._lazy import resolve as _resolve

__version__ = "0.16.0"
__all__ = _PUBLIC


def __getattr__(name):
    return _resolve(name, __name__, _EXPORTS, _MODULES, globals())


def __dir__():
    return sorted(set(globals()) | _EXPORTS.keys() | _MODULES.keys())
