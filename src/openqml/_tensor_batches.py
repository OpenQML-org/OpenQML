"""Bounded, row-major gathering of broadcast parameter batches on native devices."""

import math

import numpy as np

from .errors import ResourceError


def chunk_capacity(plan, values, shape, output_bytes=0):
    available = plan.max_memory_mb * 2**20 - plan._metadata_bytes - plan._reuse_bytes - output_bytes
    quantum = max(1, plan.working_bytes - plan._metadata_bytes - plan._reuse_bytes)
    indexing = 8 * (2 * len(values) + len(shape) + len(plan.observables) + 2)
    capacity = int(available) // (quantum + indexing)
    if capacity < 1:
        raise ResourceError("Tensor chunks and retained outputs exceed max_memory_mb.")
    return capacity


def gather_rows(values, shape, start, stop, ops):
    result, index_cache = {}, {}
    for name, value in values.items():
        if value.ndim == 0:
            result[name] = value
            continue
        host = isinstance(value, np.ndarray)
        if host not in index_cache:
            if ops.backend == "jax" and not host and math.prod(shape) > 2**31 - 1:
                import jax

                if not jax.config.x64_enabled:
                    raise ResourceError("Enable JAX x64 for batch indices above the int32 range.")
            index_cache[host] = (
                np.arange(start, stop, dtype=np.int64)
                if host
                else ops.xp.arange(start, stop, device=ops.device)
                if ops.backend == "torch"
                else ops.xp.arange(start, stop)
            )
        indices = index_cache[host]
        offset = len(shape) - value.ndim
        coordinates = tuple(
            0 if extent == 1 else (indices // math.prod(shape[offset + i + 1 :])) % extent
            for i, extent in enumerate(value.shape)
        )
        selected = value[coordinates]
        if host and ops.backend != "numpy":
            selected = ops.array(selected, real_value=True)
        result[name] = ops.xp.broadcast_to(selected, (stop - start,))
    return result
