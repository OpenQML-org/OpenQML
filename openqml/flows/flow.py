"""Flows: a portable description of the model that produced a run."""

from __future__ import annotations

import importlib
import inspect
from typing import Any, Dict, Optional

from .._store import get_store
from ..entities import OpenQMLEntity
from ..exceptions import OpenQMLError
from ..utils import get_params, jsonify, stable_hash

__all__ = ["OpenQMLFlow", "ALLOWED_MODULE_PREFIXES"]

#: Reconstructing a flow imports a module by name. Downloaded flows are data
#: from strangers, so only these prefixes are imported unless you opt in.
ALLOWED_MODULE_PREFIXES = ("openqml.", "sklearn.")


class OpenQMLFlow(OpenQMLEntity):
    """Model class + parameters + version, in a form somebody else can re-run."""

    _entity_name = "flows"
    _repr_fields = ("id", "name", "external_version", "model_class", "parameters", "tags")

    def __init__(self, name: str, model_class: str, parameters: Optional[dict] = None,
                 external_version: str = "", description: str = "", dependencies: str = "",
                 components: Optional[dict] = None, tags=None, id: Optional[int] = None,
                 uploaded_at: Optional[str] = None):
        self.id = id
        self.name = name
        self.model_class = model_class
        self.parameters = dict(parameters or {})
        self.external_version = external_version
        self.description = description
        self.dependencies = dependencies
        self.components = dict(components or {})
        self.tags = list(tags or [])
        self.uploaded_at = uploaded_at

    @property
    def flow_hash(self) -> str:
        """Identity of the *code*, ignoring hyper-parameter values."""
        return stable_hash({"class": self.model_class, "version": self.external_version})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "model_class": self.model_class,
            "parameters": jsonify(self.parameters),
            "external_version": self.external_version,
            "description": self.description,
            "dependencies": self.dependencies,
            "components": jsonify(self.components),
            "tags": self.tags,
            "flow_hash": self.flow_hash,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "OpenQMLFlow":
        known = {"id", "name", "model_class", "parameters", "external_version", "description",
                 "dependencies", "components", "tags", "uploaded_at"}
        return cls(**{k: v for k, v in payload.items() if k in known})

    def to_model(self, allowed_prefixes=ALLOWED_MODULE_PREFIXES, **overrides):
        """Instantiate the model this flow describes.

        A flow is data from a stranger, and reconstructing one imports code, so
        two rules hold whatever the payload says. The module has to sit under
        ``allowed_prefixes``, and the name has to resolve to a *class* -- a
        module-level function would turn "rebuild this model" into "call this
        function with attacker-chosen keyword arguments", which is a different
        and much larger permission.
        """
        module_name, _, class_name = self.model_class.rpartition(".")
        if allowed_prefixes is not None and not module_name.startswith(tuple(allowed_prefixes)):
            raise OpenQMLError(
                f"refusing to import {module_name!r}: reconstructing a flow imports code, so "
                f"only {allowed_prefixes} are allowed by default. Pass allowed_prefixes=None "
                f"if you trust this flow."
            )
        if "allowed_prefixes" in self.parameters:
            raise OpenQMLError(
                "refusing to rebuild a flow whose parameters carry 'allowed_prefixes': a "
                "stored parameter must not be able to widen what this flow may import"
            )
        try:
            module = importlib.import_module(module_name)
            factory = getattr(module, class_name)
        except (ImportError, AttributeError) as error:
            raise OpenQMLError(f"cannot import {self.model_class}: {error}") from None
        if not inspect.isclass(factory):
            raise OpenQMLError(
                f"refusing to call {self.model_class}: a flow names the model's class, and "
                f"{class_name!r} is a {type(factory).__name__}. Calling it would let a "
                f"published flow invoke any function in an allowed module."
            )
        parameters = dict(self.parameters)
        parameters.update(overrides)
        return factory(**parameters)
