"""One atomic model archive over the existing native training checkpoints."""

import json
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from .architecture import HybridArchitecture
from .errors import ResourceError, ValidationError
from .ir import integer

_CONFIG = (
    "input_features",
    "output_features",
    "task",
    "framework",
    "hidden_features",
    "depth",
    "num_qubits",
    "quantum_depth",
    "lr",
    "seed",
)


def _tensor_tree(value, arrays):
    """Encode Torch training state as data; never pickle arbitrary objects."""
    import torch

    if isinstance(value, torch.Tensor):
        index = len(arrays)
        arrays.append(value.detach().cpu().numpy())
        return {"tensor": index}
    if value is None or type(value) in {bool, int, float, str}:
        return {"scalar": value}
    if isinstance(value, dict):
        return {
            "dict": [[_tensor_tree(k, arrays), _tensor_tree(v, arrays)] for k, v in value.items()]
        }
    if isinstance(value, (tuple, list)):
        return {
            "tuple" if isinstance(value, tuple) else "list": [
                _tensor_tree(v, arrays) for v in value
            ]
        }
    raise ValidationError(
        "Model training state contains an unsupported object; use native checkpoints."
    )


def _restore_tree(tree, arrays):
    import torch

    if not isinstance(tree, dict) or len(tree) != 1:
        raise ValidationError("Malformed training-state tree.")
    kind, value = next(iter(tree.items()))
    if kind == "tensor":
        index = integer(value, "tensor index")
        return torch.from_numpy(arrays[f"array_{index}"])
    if kind == "scalar" and (value is None or type(value) in {bool, int, float, str}):
        return value
    if kind in {"tuple", "list"} and isinstance(value, list):
        result = [_restore_tree(v, arrays) for v in value]
        return tuple(result) if kind == "tuple" else result
    if kind == "dict" and isinstance(value, list):
        return {_restore_tree(k, arrays): _restore_tree(v, arrays) for k, v in value}
    raise ValidationError("Unsupported training-state tree node.")


def save_model(model, path):
    if model.trainer is None:
        raise ValidationError("Call fit before save.")
    if model.framework == "torch" and (
        model.module.tensor_mesh is not None or model.trainer.distributed
    ):
        raise ValidationError("Use native checkpoints for distributed training meshes.")
    micro = (
        model.trainer.microbatch_size
        if model.framework == "torch"
        else model.trainer.options.get("microbatch_size", 32)
    )
    metadata = {
        "format": "openqml.model.v1",
        "config": {name: getattr(model, name) for name in _CONFIG},
        "architecture": model.architecture.to_dict(),
        "microbatch_size": micro,
        "history": model.history,
        "shuffle_rng": model._rng.bit_generator.state,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent, prefix=path.name + ".") as folder:
        folder = Path(folder)
        if model.framework == "torch":
            arrays = []
            metadata["state_tree"] = _tensor_tree(model.trainer.state_dict(), arrays)
            with (folder / "training").open("wb") as stream:
                np.savez_compressed(stream, **{f"array_{i}": v for i, v in enumerate(arrays)})
        else:
            model.trainer.save(folder / "training")
        try:
            manifest = json.dumps(metadata, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValidationError("Model checkpoint metadata must be finite JSON.") from exc
        with zipfile.ZipFile(folder / "archive", "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("model.json", manifest)
            archive.write(folder / "training", "training")
        os.replace(folder / "archive", path)
    return path


def load_model(cls, path, *, device, backend, max_bytes):
    maximum = integer(max_bytes, "max_bytes", 1)
    try:
        with zipfile.ZipFile(path) as archive:
            info = archive.infolist()
            if len(info) != 2 or {v.filename for v in info} != {"model.json", "training"}:
                raise ValidationError("Model checkpoint must contain metadata and training state.")
            if sum(v.file_size for v in info) > maximum:
                raise ResourceError("Model checkpoint exceeds max_bytes.")
            metadata = json.loads(archive.read("model.json"))
            if not isinstance(metadata, dict) or metadata.get("format") != "openqml.model.v1":
                raise ValidationError("Unsupported model checkpoint format.")
            config = metadata["config"]
            if not isinstance(config, dict) or set(config) != set(_CONFIG):
                raise ValidationError("Malformed model checkpoint configuration.")
            specification = metadata["architecture"]
            if integer(specification["depth"], "depth", 1) > maximum // 64:
                raise ResourceError("Model architecture exceeds the checkpoint size budget.")
            architecture = HybridArchitecture.from_dict(specification)
            if 4 * sum(np.prod(p.shape, dtype=object) for p in architecture.parameters) > maximum:
                raise ResourceError("Model parameters exceed the checkpoint size budget.")
            micro = integer(metadata["microbatch_size"], "microbatch_size", 1)
            rng = np.random.default_rng()
            rng.bit_generator.state = metadata["shuffle_rng"]
            history = metadata["history"]
            if not isinstance(history, list) or any(
                not isinstance(row, dict)
                or set(row) != {"loss", "updates"}
                or not np.isfinite(float(row["loss"]))
                or integer(row["updates"], "updates") < 0
                for row in history
            ):
                raise ValidationError("Malformed model training history.")
            arguments = dict(config, device=device, backend=backend)
            if config["task"] != "forecast":
                arguments["architecture"] = architecture
            model = cls(**arguments)
            model._build(config["input_features"], config["output_features"], micro)
            if model.architecture.fingerprint != architecture.fingerprint:
                raise ValidationError("Model configuration and saved architecture differ.")
            # Read members by name; never extract arbitrary paths from an archive.
            with tempfile.TemporaryDirectory(prefix="openqml-load-") as folder:
                payload = Path(folder) / "training"
                payload.write_bytes(archive.read("training"))
                with zipfile.ZipFile(payload) as inner:
                    if sum(v.file_size for v in inner.infolist()) > maximum:
                        raise ResourceError("Expanded training tensors exceed max_bytes.")
                if model.framework == "torch":
                    with np.load(payload, allow_pickle=False) as arrays:
                        state = _restore_tree(metadata["state_tree"], arrays)
                    model.trainer.load_state_dict(state)
                else:
                    model.trainer.load(payload)
            model._rng, model.history = rng, history
            return model
    except (zipfile.BadZipFile, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, (ValidationError, ResourceError)):
            raise
        raise ValidationError("Malformed model checkpoint.") from exc
