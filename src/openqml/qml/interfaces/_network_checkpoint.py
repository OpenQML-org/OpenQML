"""Atomic, pickle-free JAX architecture/optimizer checkpoints."""

import json
import os
import tempfile
from pathlib import Path

import jax
import numpy as np

from ...errors import ValidationError
from ...ir import integer
from ...ir._serialization import load_json
from .._archive import check_npz_payload
from ..training import DepthSchedule


def _layout(tree):
    pairs, definition = jax.tree_util.tree_flatten_with_path(tree)
    records = [
        {
            "path": jax.tree_util.keystr(path),
            "shape": list(np.shape(value)),
            "dtype": str(value.dtype) if hasattr(value, "dtype") else str(np.asarray(value).dtype),
        }
        for path, value in pairs
    ]
    return records, definition, [value for _, value in pairs]


def save_network(trainer, path):
    layout, definition, leaves = _layout((trainer.parameters, trainer.optimizer_state))
    metadata = {
        "format": "openqml.network.training.v1",
        "architecture": trainer.model.architecture.fingerprint,
        "architecture_spec": trainer.model.architecture.to_dict(),
        "active_depth": trainer.model.active_depth,
        "depth_schedule": trainer.depth_schedule.stages,
        "updates": trainer.updates,
        "attempts": trainer.attempts,
        "layout": layout,
        "tree": str(definition),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(handle, "wb") as stream:
            np.savez_compressed(
                stream,
                metadata=np.asarray(json.dumps(metadata)),
                **{f"leaf_{i}": np.asarray(value) for i, value in enumerate(leaves)},
            )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def load_network(trainer, path, *, max_bytes=256 * 2**20):
    maximum = integer(max_bytes, "max_bytes", 1)
    check_npz_payload(path, maximum)
    expected, definition, templates = _layout((trainer.parameters, trainer.optimizer_state))
    with np.load(path, allow_pickle=False) as data:
        metadata = load_json(str(data["metadata"]), maximum, label="Network checkpoint")
        if (
            metadata.get("format") != "openqml.network.training.v1"
            or metadata.get("architecture") != trainer.model.architecture.fingerprint
            or metadata.get("layout") != expected
            or metadata.get("tree") != str(definition)
            or set(data.files) != {"metadata", *(f"leaf_{i}" for i in range(len(templates)))}
        ):
            raise ValidationError("Network checkpoint architecture/optimizer layout differs.")
        schedule = DepthSchedule(metadata["depth_schedule"])
        trainer.model.architecture.active(schedule.stages[-1][1])
        depth = trainer.model.architecture.active(metadata["active_depth"])
        updates = integer(metadata["updates"], "updates", 0)
        attempts = integer(metadata["attempts"], "attempts", updates)
        leaves = []
        for i, (record, template) in enumerate(zip(expected, templates, strict=True)):
            value = data[f"leaf_{i}"]
            if list(value.shape) != record["shape"] or str(value.dtype) != record["dtype"]:
                raise ValidationError("Network checkpoint tensor shape/dtype differs.")
            leaves.append(jax.device_put(value, getattr(template, "sharding", None)))
    parameters, optimizer_state = jax.tree_util.tree_unflatten(definition, leaves)
    trainer.parameters, trainer.optimizer_state = parameters, optimizer_state
    trainer.depth_schedule, trainer.model = schedule, trainer.model.with_depth(depth)
    trainer.updates, trainer.attempts = updates, attempts
    trainer._compiled, trainer._compiled_model = None, None
    return trainer
