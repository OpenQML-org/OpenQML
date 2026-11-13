"""Tensor archive validation and shared atomic checkpoint writes."""

import io
import math
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from ..errors import ResourceError, ValidationError


def atomic_write(path, writer):
    """Replace a checkpoint only after its complete writer succeeds."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(handle, "wb") as stream:
            writer(stream)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def check_npz_payload(payload, maximum):
    from ..ir.core import integer

    maximum = integer(maximum, "max_bytes", 1)
    try:
        _check_npz_payload(payload, maximum)
    except (ValidationError, ResourceError):
        raise
    except (zipfile.BadZipFile, EOFError, ValueError, RuntimeError, OverflowError) as exc:
        raise ValidationError("Malformed training tensor archive.") from exc


def _check_npz_payload(payload, maximum):
    source = io.BytesIO(payload) if isinstance(payload, (bytes, bytearray)) else payload
    with zipfile.ZipFile(source) as archive:
        members = archive.infolist()
        if len({item.filename for item in members}) != len(members):
            raise ValidationError("Training tensor archive contains duplicate members.")
        if sum(item.file_size for item in members) > maximum:
            raise ResourceError("Expanded training tensors exceed max_bytes.")
        for item in members:
            if not item.filename.endswith(".npy"):
                raise ValidationError("Training tensor members must use the NPY format.")
            with archive.open(item) as stream:
                version = np.lib.format.read_magic(stream)
                if version not in {(1, 0), (2, 0)}:
                    raise ValidationError("Unsupported training tensor format version.")
                length_size = 2 if version == (1, 0) else 4
                encoded = stream.read(length_size)
                if len(encoded) != length_size:
                    raise ValidationError("Truncated training tensor header.")
                length = int.from_bytes(encoded, "little")
                if length > min(16384, maximum):
                    raise ResourceError("Training tensor header exceeds its size budget.")
                header = stream.read(length)
                if len(header) != length:
                    raise ValidationError("Truncated training tensor header.")
                reader = (
                    np.lib.format.read_array_header_1_0
                    if version == (1, 0)
                    else np.lib.format.read_array_header_2_0
                )
                shape, _, dtype = reader(io.BytesIO(encoded + header), max_header_size=16384)
                if dtype.hasobject:
                    raise ValidationError("Training tensors must be pickle-free numeric arrays.")
                required = math.prod(shape) * dtype.itemsize
                if required > maximum or any(v > np.iinfo(np.intp).max for v in shape):
                    raise ResourceError("Declared tensor shape exceeds max_bytes.")
                if required != item.file_size - (8 + length_size + length):
                    raise ValidationError("Tensor shape and stored payload size differ.")
