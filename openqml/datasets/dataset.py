"""The dataset entity."""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .. import config
from .._store import get_store
from ..entities import OpenQMLEntity
from ..exceptions import OpenQMLCacheError
from ..hamiltonians import exact_ground_state, terms_from_json, terms_to_json
from ..utils import stable_hash
from .generators import get_generator

__all__ = ["OpenQMLDataset"]

DATA_TYPES = ("tabular", "bitstring", "quantum_state", "hamiltonian")


class OpenQMLDataset(OpenQMLEntity):
    """Features plus the quantum metadata a QML experiment actually needs.

    ``data_type`` decides how the rows should be read:

    ``tabular`` / ``bitstring``
        real-valued rows, to be encoded by a feature map.
    ``quantum_state``
        rows are already normalised statevectors of ``qubits`` qubits.
    ``hamiltonian``
        rows are Pauli sums; ``get_hamiltonians()`` returns them together with
        exact reference energies.
    """

    _entity_name = "datasets"
    _repr_fields = ("id", "name", "version", "data_type", "n_samples", "n_features",
                    "n_classes", "qubits", "suggested_encoding", "licence", "tags")

    def __init__(self, name: str, description: str = "", data_type: str = "tabular",
                 default_target_attribute: str = "class", attribute_names=None,
                 qubits: Optional[int] = None, suggested_encoding: str = "angle",
                 licence: str = "CC-BY-4.0", creator: str = "", citation: str = "",
                 tags=None, version: int = 1, id: Optional[int] = None,
                 generator: Optional[dict] = None, data_file: Optional[str] = None,
                 data_b64: Optional[str] = None, n_samples: Optional[int] = None,
                 n_features: Optional[int] = None, n_classes: Optional[int] = None,
                 hamiltonians: Optional[list] = None, meta: Optional[dict] = None,
                 arrays: Optional[dict] = None, uploaded_at: Optional[str] = None):
        if data_type not in DATA_TYPES:
            raise ValueError(f"data_type must be one of {DATA_TYPES}")
        self.id = id
        self.name = name
        self.description = description
        self.data_type = data_type
        self.default_target_attribute = default_target_attribute
        self.attribute_names = list(attribute_names or [])
        self.qubits = qubits
        self.suggested_encoding = suggested_encoding
        self.licence = licence
        self.creator = creator
        self.citation = citation
        self.tags = list(tags or [])
        self.version = version
        self.generator = generator
        self.data_file = data_file
        self.data_b64 = data_b64
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes
        self.hamiltonians = hamiltonians
        self.meta = dict(meta or {})
        self.uploaded_at = uploaded_at
        self._arrays = arrays

    # -- loading ----------------------------------------------------------
    def _cache_path(self):
        return config.get_cache_directory("datasets", self.id or f"tmp-{stable_hash(self.name)}") / "data.npz"

    def _materialise(self) -> Dict[str, np.ndarray]:
        if self._arrays is not None:
            return self._arrays
        cache = self._cache_path()
        if cache.exists():
            with np.load(cache, allow_pickle=False) as handle:
                self._arrays = {key: handle[key] for key in handle.files}
            return self._arrays
        if self.data_file:
            with np.load(self.data_file, allow_pickle=False) as handle:
                arrays = {key: handle[key] for key in handle.files}
        elif self.data_b64:
            buffer = io.BytesIO(base64.b64decode(self.data_b64))
            with np.load(buffer, allow_pickle=False) as handle:
                arrays = {key: handle[key] for key in handle.files}
        elif self.generator:
            arrays = self._run_generator()
        else:
            raise OpenQMLCacheError(f"dataset {self.name!r} carries no data source")
        np.savez_compressed(cache, **arrays)
        self._arrays = arrays
        return arrays

    def _run_generator(self) -> Dict[str, np.ndarray]:
        function = get_generator(self.generator["function"])
        result = function(**self.generator.get("kwargs", {}))
        if self.data_type == "hamiltonian":
            separations = np.array([record["separation"] for record in result], dtype=float)
            energies = np.array([exact_ground_state(record["terms"], self.qubits)[0]
                                 for record in result], dtype=float)
            self.hamiltonians = [
                {"separation": record["separation"], "terms": terms_to_json(record["terms"])}
                for record in result
            ]
            return {"X": separations.reshape(-1, 1), "y": energies}
        X, y, names = result[0], result[1], result[2]
        if not self.attribute_names:
            self.attribute_names = list(names)
        return {"X": np.asarray(X), "y": np.asarray(y)}

    def get_data(self, target: Optional[str] = None,
                 dataset_format: str = "array") -> Tuple[Any, Any, List[str]]:
        """Return ``(X, y, attribute_names)``.

        ``target=None`` returns the full feature matrix and ``y=None`` -- the
        same convention as ``openml.datasets.OpenQMLDataset.get_data``, minus
        the categorical indicator, which quantum encodings do not use.
        """
        arrays = self._materialise()
        X, y = arrays["X"], arrays.get("y")
        names = list(self.attribute_names) or [f"x{i}" for i in range(X.shape[1])]
        if dataset_format == "dataframe":
            try:
                import pandas as pd

                X = pd.DataFrame(X, columns=names)
                y = None if y is None else pd.Series(y, name=target or "y")
            except ImportError:  # pragma: no cover
                pass
        if target is None:
            return X, None, names
        return X, y, names

    def get_statevectors(self) -> np.ndarray:
        """Rows as normalised complex statevectors (any data type)."""
        X = np.asarray(self._materialise()["X"], dtype=complex)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms

    def get_hamiltonians(self) -> List[dict]:
        """``[{'separation': r, 'terms': [...], 'reference_energy': e}, ...]``."""
        if self.data_type != "hamiltonian":
            raise ValueError(f"dataset {self.name!r} is not a Hamiltonian dataset")
        arrays = self._materialise()
        records = []
        for i, entry in enumerate(self.hamiltonians or []):
            terms = terms_from_json(entry["terms"])
            records.append({
                "separation": entry.get("separation"),
                "terms": terms,
                "reference_energy": float(arrays["y"][i]),
            })
        return records

    @property
    def qubits_required(self) -> int:
        """Qubits needed for the suggested encoding."""
        if self.qubits:
            return int(self.qubits)
        features = self.n_features or self._materialise()["X"].shape[1]
        if self.suggested_encoding == "amplitude":
            return int(np.ceil(np.log2(max(features, 2))))
        return int(features)

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "data_type": self.data_type,
            "default_target_attribute": self.default_target_attribute,
            "attribute_names": self.attribute_names,
            "qubits": self.qubits,
            "suggested_encoding": self.suggested_encoding,
            "licence": self.licence,
            "creator": self.creator,
            "citation": self.citation,
            "tags": self.tags,
            "version": self.version,
            "generator": self.generator,
            "data_file": self.data_file,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "hamiltonians": self.hamiltonians,
            "meta": self.meta,
        }
        return {k: v for k, v in payload.items() if v is not None}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "OpenQMLDataset":
        known = {
            "id", "name", "description", "data_type", "default_target_attribute",
            "attribute_names", "qubits", "suggested_encoding", "licence", "creator",
            "citation", "tags", "version", "generator", "data_file", "data_b64",
            "n_samples", "n_features", "n_classes", "hamiltonians", "meta", "uploaded_at",
        }
        return cls(**{k: v for k, v in payload.items() if k in known})

    def publish(self) -> "OpenQMLDataset":
        """Store the arrays, then register the metadata."""
        if self._arrays is not None and not self.generator:
            folder = config.get_cache_directory("local_store", "data")
            digest = stable_hash({k: v.tolist() for k, v in self._arrays.items()})
            path = folder / f"{digest}.npz"
            if not path.exists():
                np.savez_compressed(path, **self._arrays)
            if config.is_local():
                self.data_file = str(path)
            else:
                self.data_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        payload = self.to_dict()
        if self.data_b64:
            payload["data_b64"] = self.data_b64
        response = get_store().post(self._entity_name, payload)
        self.id = int(response["id"])
        return self

    def summary(self) -> str:
        return (f"{self.name} (id={self.id}, {self.data_type}): "
                f"{self.n_samples} samples x {self.n_features} features, "
                f"{self.n_classes or '-'} classes, {self.qubits_required} qubits "
                f"via {self.suggested_encoding} encoding")
