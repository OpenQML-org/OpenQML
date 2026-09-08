"""Datasets: quantum-aware metadata on top of ordinary arrays."""

from . import generators
from .dataset import OpenQMLDataset
from .functions import (
    create_dataset,
    delete_dataset,
    get_dataset,
    get_datasets,
    list_datasets,
)

__all__ = [
    "OpenQMLDataset", "list_datasets", "get_dataset", "get_datasets",
    "create_dataset", "delete_dataset", "generators",
]
