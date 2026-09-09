"""Small helpers shared across the package."""

from __future__ import annotations

import copy
import hashlib
import json
import numbers
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np

__all__ = ["clone", "get_params", "jsonify", "stable_hash", "as_table", "check_random_state"]


def get_params(estimator) -> Dict[str, Any]:
    """Return an estimator's constructor parameters (scikit-learn convention)."""
    if hasattr(estimator, "get_params"):
        return dict(estimator.get_params())
    return {
        k: v for k, v in vars(estimator).items() if not k.startswith("_") and not k.endswith("_")
    }


def clone(estimator):
    """Return an unfitted copy of ``estimator``.

    Uses ``get_params``/``__class__`` when available -- which covers both the
    built-in models and any scikit-learn estimator -- and falls back to a deep
    copy for objects that do not follow the convention.
    """
    if hasattr(estimator, "get_params"):
        params = {k: clone(v) if hasattr(v, "get_params") else copy.deepcopy(v)
                  for k, v in estimator.get_params().items()}
        return estimator.__class__(**params)
    return copy.deepcopy(estimator)


def jsonify(obj):
    """Convert numpy/complex containers into something ``json.dumps`` accepts."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, numbers.Integral):
        return int(obj)
    if isinstance(obj, numbers.Real):
        return float(obj)
    if isinstance(obj, complex):
        return {"__complex__": True, "real": obj.real, "imag": obj.imag}
    if isinstance(obj, np.ndarray):
        return jsonify(obj.tolist())
    if isinstance(obj, np.generic):
        return jsonify(obj.item())
    if isinstance(obj, Mapping):
        return {str(k): jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (set, frozenset)):
        # a set iterates in hash order, which is randomised per process for
        # strings -- and json.dumps(sort_keys=True) does not reorder a list, so
        # leaving it unsorted makes stable_hash unstable across runs.
        return sorted((jsonify(v) for v in obj), key=repr)
    if isinstance(obj, (list, tuple)):
        return [jsonify(v) for v in obj]
    if hasattr(obj, "to_dict"):
        return jsonify(obj.to_dict())
    return str(obj)


def stable_hash(obj) -> str:
    """Deterministic short hash of any jsonifiable object."""
    payload = json.dumps(jsonify(obj), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def check_random_state(seed):
    """Return a ``numpy.random.Generator`` for ``None`` / int / Generator."""
    if seed is None or isinstance(seed, (int, np.integer)):
        return np.random.default_rng(seed)
    if isinstance(seed, np.random.Generator):
        return seed
    if isinstance(seed, np.random.RandomState):
        return np.random.default_rng(seed.randint(0, 2**31 - 1))
    raise ValueError(f"cannot interpret {seed!r} as a random state")


def as_table(records: List[dict], output_format: str = "dataframe", index: str | None = None):
    """Return ``records`` as a pandas DataFrame, or as a dict keyed by ``index``.

    ``output_format="dataframe"`` degrades to ``"dict"`` when pandas is absent,
    so the library never hard-depends on pandas.
    """
    if output_format not in ("dataframe", "dict", "records"):
        raise ValueError("output_format must be 'dataframe', 'dict' or 'records'")
    if output_format == "records":
        return records
    if output_format == "dict":
        if index is None:
            return {i: r for i, r in enumerate(records)}
        return {r[index]: r for r in records}
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - depends on the environment
        return as_table(records, "dict", index)
    frame = pd.DataFrame(records)
    if index is not None and index in frame.columns and len(frame):
        frame = frame.set_index(index, drop=False)
    return frame
