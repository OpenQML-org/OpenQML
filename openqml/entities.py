"""Common base class for the four publishable entity types."""

from __future__ import annotations

from typing import Any, Dict, Optional

from . import config
from ._store import get_store
from .exceptions import ObjectNotPublishedError

__all__ = ["OpenQMLEntity"]


class OpenQMLEntity:
    """Shared behaviour: identity, serialisation, publishing, pretty printing."""

    _entity_name = "entities"
    _repr_fields: tuple = ()

    id: Optional[int] = None

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - overridden
        raise NotImplementedError

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]):  # pragma: no cover - overridden
        raise NotImplementedError

    # -- server round trip ------------------------------------------------
    def publish(self):
        """Upload the entity and return it with its assigned ``id``."""
        response = get_store().post(self._entity_name, self.to_dict())
        self.id = int(response["id"])
        return self

    @property
    def url(self) -> Optional[str]:
        """Where this entity lives on the configured server, if there is one."""
        if self.id is None or config.is_local():
            return None
        return f"{config.server.rstrip('/')}/{self._entity_name}/{self.id}"

    def require_id(self) -> int:
        if self.id is None:
            raise ObjectNotPublishedError(
                f"this {type(self).__name__} has no id yet -- call .publish() first"
            )
        return int(self.id)

    # -- presentation -----------------------------------------------------
    def _repr_rows(self):
        for field in self._repr_fields:
            value = getattr(self, field, None)
            if value is None or value == [] or value == {}:
                continue
            yield field.replace("_", " ").title(), value

    def __repr__(self) -> str:
        header = f"OpenQML {type(self).__name__.replace('OpenQML', '')}"
        rows = list(self._repr_rows())
        width = max([len(name) for name, _ in rows] + [len(header)])
        lines = [header, "=" * (width + 2)]
        for name, value in rows:
            text = ", ".join(map(str, value)) if isinstance(value, (list, tuple)) else str(value)
            if len(text) > 68:
                text = text[:65] + "..."
            lines.append(f"{name.ljust(width)}: {text}")
        return "\n".join(lines)
