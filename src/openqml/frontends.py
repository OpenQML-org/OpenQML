"""Explicit extension registry for additional circuit ecosystems."""

import re
from dataclasses import dataclass
from importlib import import_module
from threading import RLock

from .errors import ValidationError, conversion_error
from .ir import Circuit

BUILTIN_FORMATS = ("ir", "json", "qiskit", "pennylane", "cirq", "openqasm2", "openqasm3")
_REGISTRY = {}
_LOCK = RLock()


@dataclass(frozen=True)
class FrameworkAdapter:
    """An importer/exporter plus optional detector and representability diagnostics.

    importer(value, **source_options) -> Circuit
    exporter(circuit) -> native object
    detector(value) -> bool
    diagnose(circuit) -> iterable[Diagnostic]
    binder(native, bindings) -> native (optional; defaults to IR bind/export)
    runner(native, *, shots, seed, **options) -> native result (optional)
    """

    name: str
    importer: object
    exporter: object
    detector: object = None
    diagnose: object = None
    binder: object = None
    runner: object = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not re.fullmatch(r"[a-z][a-z0-9_-]*", self.name):
            raise ValidationError("An adapter needs a lowercase format name.")
        if not callable(self.importer) or not callable(self.exporter):
            raise ValidationError("An adapter needs importer and exporter callables.")
        if self.detector is not None and not callable(self.detector):
            raise ValidationError("Adapter detector must be callable.")
        if self.diagnose is not None and not callable(self.diagnose):
            raise ValidationError("Adapter diagnose must be callable.")
        if self.binder is not None and not callable(self.binder):
            raise ValidationError("Adapter binder must be callable.")
        if self.runner is not None and not callable(self.runner):
            raise ValidationError("Adapter runner must be callable.")


def register_adapter(adapter, *, replace=False):
    """Register only an explicit instance; package discovery never executes third-party code."""
    if not isinstance(adapter, FrameworkAdapter):
        raise ValidationError("Expected a FrameworkAdapter instance.")
    with _LOCK:
        if adapter.name in BUILTIN_FORMATS or adapter.name in {"qasm2", "qasm3"}:
            raise ValidationError("Built-in format names are protected.")
        if adapter.name in _REGISTRY and not replace:
            raise ValidationError(f"Adapter {adapter.name!r} is already registered.")
        _REGISTRY[adapter.name] = adapter


def unregister_adapter(name):
    with _LOCK:
        if name not in _REGISTRY:
            raise ValidationError("Only a registered custom adapter can be removed.")
        return _REGISTRY.pop(name)


def list_formats():
    with _LOCK:
        return BUILTIN_FORMATS + tuple(sorted(_REGISTRY))


def get_adapter(name):
    with _LOCK:
        if name in _REGISTRY:
            return _REGISTRY[name]
    if name in {"qiskit", "cirq", "pennylane"}:
        module = import_module(f"openqml.adapters.{name}")
        return FrameworkAdapter(
            name, getattr(module, f"from_{name}"), getattr(module, f"to_{name}")
        )
    raise conversion_error("UNKNOWN_FORMAT", f"No native circuit adapter registered for {name!r}.")


def detect_custom(value):
    with _LOCK:
        adapters = tuple(_REGISTRY.values())
    matches = [adapter.name for adapter in adapters if adapter.detector and adapter.detector(value)]
    if len(matches) > 1:
        raise conversion_error(
            "AMBIGUOUS_FRAMEWORK", f"Multiple adapters recognize this object: {matches}."
        )
    return matches[0] if matches else None


def import_native(value, framework, **options):
    circuit = get_adapter(framework).importer(value, **options)
    if not isinstance(circuit, Circuit):
        raise conversion_error("ADAPTER_CONTRACT", "An adapter importer must return a Circuit.")
    return circuit
