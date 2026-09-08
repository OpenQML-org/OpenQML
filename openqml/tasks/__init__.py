"""Tasks: a dataset plus the protocol that makes two results comparable."""

from .functions import create_task, delete_task, get_task, get_tasks, list_tasks
from .split import Split, make_split
from .task import (
    TASK_TYPES,
    GroundStateEstimationTask,
    OpenQMLTask,
    StateClassificationTask,
    SupervisedClassificationTask,
    SupervisedRegressionTask,
    task_from_dict,
)

__all__ = [
    "OpenQMLTask", "SupervisedClassificationTask", "SupervisedRegressionTask",
    "StateClassificationTask", "GroundStateEstimationTask", "TASK_TYPES",
    "list_tasks", "get_task", "get_tasks", "create_task", "delete_task",
    "Split", "make_split", "task_from_dict",
]
