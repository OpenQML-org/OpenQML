"""Benchmark suites: a named, citable set of tasks."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .._store import get_store
from ..entities import OpenQMLEntity
from ..tasks import get_task, get_tasks
from ..utils import as_table

__all__ = ["OpenQMLSuite", "get_suite", "list_suites", "create_suite", "delete_suite"]


class OpenQMLSuite(OpenQMLEntity):
    """A fixed list of tasks, so "we benchmarked on X" means something."""

    _entity_name = "suites"
    _repr_fields = ("id", "name", "alias", "task_ids", "tags")

    def __init__(self, name: str, task_ids: Sequence[int], description: str = "",
                 alias: str = "", tags=None, id: Optional[int] = None,
                 uploaded_at: Optional[str] = None):
        self.id = id
        self.name = name
        self.alias = alias
        self.task_ids = [int(t) for t in task_ids]
        self.description = description
        self.tags = list(tags or [])
        self.uploaded_at = uploaded_at

    @property
    def tasks(self) -> List:
        """The task objects themselves."""
        return get_tasks(self.task_ids)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "alias": self.alias,
            "task_ids": self.task_ids, "description": self.description, "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "OpenQMLSuite":
        known = {"id", "name", "alias", "task_ids", "description", "tags", "uploaded_at"}
        return cls(**{k: v for k, v in payload.items() if k in known})

    def run(self, model, publish: bool = False, **kwargs) -> List:
        """Run one model over every task in the suite."""
        from ..runs import run_models_on_tasks

        return run_models_on_tasks([model], self.tasks, publish=publish, **kwargs)

    def compare(self, models, **kwargs):
        """Model x task table over the whole suite."""
        from ..evaluations import compare

        return compare(models, self.tasks, **kwargs)


def get_suite(identifier) -> OpenQMLSuite:
    """Fetch a suite by id, name or alias."""
    return OpenQMLSuite.from_dict(get_store().get("suites", identifier))


def list_suites(output_format: str = "dataframe"):
    records = [{
        "id": payload.get("id"),
        "name": payload.get("name"),
        "alias": payload.get("alias"),
        "n_tasks": len(payload.get("task_ids", [])),
        "tags": ",".join(payload.get("tags", [])),
    } for payload in get_store().list("suites")]
    return as_table(records, output_format, index="id")


def create_suite(name: str, task_ids: Sequence[int], description: str = "", alias: str = "",
                 tags: Optional[Sequence[str]] = None) -> OpenQMLSuite:
    """Define a suite. Call ``.publish()`` to register it."""
    return OpenQMLSuite(name=name, task_ids=task_ids, description=description, alias=alias,
                        tags=tags)


def delete_suite(identifier) -> None:
    get_store().delete("suites", identifier)
