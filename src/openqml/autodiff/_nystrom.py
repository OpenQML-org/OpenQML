"""A damped low-rank metric preconditioner in native parameter coordinates."""

import math

from ..compute._norms import stop
from ..errors import ValidationError


def preconditioner(matvec, gradient, ops, rank, damping, max_memory_mb, reserved_bytes=0):
    """Use orthonormal cosine probes and a regularized rank-space Woodbury solve."""
    shapes = tuple(v.shape for v in gradient.values())
    counts = tuple(math.prod(shape) for shape in shapes)
    size = sum(counts)
    if rank > size:
        raise ValidationError("preconditioner_rank exceeds the number of coordinates.")
    ops.kernel.admit(
        reserved_bytes + 8 * (7 * size * rank + 10 * rank * rank + 6 * size),
        max_memory_mb,
        itemsize=1,
    )
    xp = ops.xp

    def flat(leaves):
        parts = tuple(leaves[k].reshape(-1) for k in gradient)
        return (
            parts[0]
            if len(parts) == 1
            else (xp.cat(parts) if ops.backend == "torch" else xp.concatenate(parts))
        )

    def unpack(vector):
        offset, leaves = 0, {}
        for key, count, shape in zip(gradient, counts, shapes, strict=True):
            leaves[key] = vector[offset : offset + count].reshape(shape)
            offset += count
        return leaves

    device = {"device": ops.device} if ops.backend == "torch" else {}
    # Construct fixed probes on the vector's device, independently of RNG state.
    rows = ops.array(xp.arange(size, **device), real_value=True)
    cols = ops.array(xp.arange(rank, **device), real_value=True)
    probes = xp.cos(((rows + 0.5) * (math.pi / size))[:, None] * cols[None, :])
    probes = probes * xp.where(
        cols == 0,
        ops.array(math.sqrt(1 / size), real_value=True),
        ops.array(math.sqrt(2 / size), real_value=True),
    )
    columns = tuple(flat(matvec(unpack(probes[:, j]))) for j in range(rank))
    y = xp.stack(columns, dim=1) if ops.backend == "torch" else xp.stack(columns, axis=1)
    scale = stop(xp.maximum(xp.abs(y).max(), ops.array(damping, real_value=True)), ops.backend)
    y = y / scale
    w = ops.matmul(probes.T, y)
    w = (w + w.T) * 0.5
    eye = ops.array(xp.eye(rank, **device), real_value=True)
    # The shift protects null and dependent sketch directions. The outer
    # linear system and its freshly recomputed residual still use the full metric.
    gram = ops.matmul(y.T, y)
    relative_damping = damping / scale
    ridge = (16 * rank * xp.finfo(y.dtype).eps) * (
        1 + xp.abs(gram).max() + relative_damping * xp.abs(w).max()
    )
    regularizer = relative_damping * w + ridge * eye
    core = gram + regularizer
    rhs = (
        xp.cat((y.T, regularizer), dim=1)
        if ops.backend == "torch"
        else xp.concatenate((y.T, regularizer), axis=1)
    )
    factors = ops.solve(core, rhs)
    gain, coarse = factors[:, :size], factors[:, size:]

    def apply(leaves):
        v = flat(leaves)
        a = ops.matmul(gain, v)
        remainder = v - ops.matmul(y, a)
        # For H=Y core^-1 Y^T, (I-H)^2 + Y core^-1 R core^-1 Y^T = I-H.
        # This form restores the small damped component directly inside range(Y).
        return unpack(
            remainder + ops.matmul(y, ops.matmul(coarse, a) - ops.matmul(gain, remainder))
        )

    return apply
