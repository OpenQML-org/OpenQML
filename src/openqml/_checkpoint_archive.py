"""Validate tensor archive declarations before allocation or native loading."""

import io
import math
import zipfile

import numpy as np

from .errors import ResourceError, ValidationError


def check_npz_payload(payload, maximum):
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
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
                shape, _, dtype = reader(
                    io.BytesIO(encoded + header), max_header_size=16384
                )
                if dtype.hasobject:
                    raise ValidationError("Training tensors must be pickle-free numeric arrays.")
                required = math.prod(shape) * dtype.itemsize
                if required > maximum:
                    raise ResourceError("Declared tensor shape exceeds max_bytes.")
                if required != item.file_size - (8 + length_size + length):
                    raise ValidationError("Tensor shape and stored payload size differ.")
