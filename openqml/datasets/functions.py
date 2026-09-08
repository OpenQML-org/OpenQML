"""Dataset-level API: list, get, create, delete."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .._store import get_store
from ..utils import as_table
from .dataset import OpenQMLDataset

__all__ = ["list_datasets", "get_dataset", "get_datasets", "create_dataset", "delete_dataset"]


def list_datasets(data_type: Optional[str] = None, tag: Optional[str] = None,
                  output_format: str = "dataframe"):
    """List datasets, optionally filtered by ``data_type`` or ``tag``."""
    records = []
    for payload in get_store().list("datasets"):
        if data_type and payload.get("data_type") != data_type:
            continue
        if tag and tag not in payload.get("tags", []):
            continue
        records.append({
            "id": payload.get("id"),
            "name": payload.get("name"),
            "data_type": payload.get("data_type"),
            "n_samples": payload.get("n_samples"),
            "n_features": payload.get("n_features"),
            "n_classes": payload.get("n_classes"),
            "qubits": payload.get("qubits"),
            "suggested_encoding": payload.get("suggested_encoding"),
            "tags": ",".join(payload.get("tags", [])),
        })
    return as_table(records, output_format, index="id")


def get_dataset(identifier, download_data: bool = True) -> OpenQMLDataset:
    """Fetch one dataset by numeric id or by name."""
    dataset = OpenQMLDataset.from_dict(get_store().get("datasets", identifier))
    if download_data:
        dataset._materialise()
    return dataset


def get_datasets(identifiers: Sequence, download_data: bool = True) -> List[OpenQMLDataset]:
    return [get_dataset(i, download_data) for i in identifiers]


def create_dataset(name: str, description: str, X, y=None, attribute_names=None,
                   data_type: str = "tabular", default_target_attribute: str = "class",
                   qubits: Optional[int] = None, suggested_encoding: str = "angle",
                   licence: str = "CC-BY-4.0", creator: str = "", citation: str = "",
                   tags: Optional[Sequence[str]] = None,
                   meta: Optional[Dict[str, Any]] = None) -> OpenQMLDataset:
    """Wrap in-memory arrays as a dataset. Call ``.publish()`` to register it."""
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be two-dimensional (n_samples, n_features)")
    arrays = {"X": X}
    n_classes = None
    if y is not None:
        y = np.asarray(y)
        if len(y) != len(X):
            raise ValueError("X and y must have the same number of rows")
        arrays["y"] = y
        if y.dtype.kind in "iub" or data_type in ("bitstring",):
            n_classes = int(len(np.unique(y)))
    return OpenQMLDataset(
        name=name,
        description=description,
        data_type=data_type,
        default_target_attribute=default_target_attribute,
        attribute_names=list(attribute_names or [f"x{i}" for i in range(X.shape[1])]),
        qubits=qubits,
        suggested_encoding=suggested_encoding,
        licence=licence,
        creator=creator,
        citation=citation,
        tags=list(tags or []),
        n_samples=int(X.shape[0]),
        n_features=int(X.shape[1]),
        n_classes=n_classes,
        meta=meta,
        arrays=arrays,
    )


def delete_dataset(identifier) -> None:
    """Remove a dataset (local store only unless the server allows it)."""
    get_store().delete("datasets", identifier)
