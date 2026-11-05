"""Scaled native reductions for finite norms across wide floating-point ranges."""


def stop(value, backend):
    if backend == "torch":
        return value.detach()
    if backend == "jax":
        import jax

        return jax.lax.stop_gradient(value)
    return value


def vector_norm(value, xp, backend, *, axis=None, keepdims=False):
    """L2 norm with scaled sum-of-squares and a zero subgradient at the origin."""
    magnitude = xp.abs(value)
    if str(magnitude.dtype).removeprefix("torch.") in {"float16", "bfloat16"}:
        magnitude = magnitude.float() if backend == "torch" else magnitude.astype("float32")
    axes = tuple(range(value.ndim)) if axis is None else (axis,) if isinstance(axis, int) else axis
    if not axes:
        return xp.where(magnitude == 0, 0, magnitude)
    if any(value.shape[a] == 0 for a in axes):
        return (
            magnitude.sum(dim=axes, keepdim=keepdims)
            if backend == "torch"
            else magnitude.sum(axis=axes, keepdims=keepdims)
        )
    scale = (
        magnitude.amax(dim=axes, keepdim=True)
        if backend == "torch"
        else xp.max(magnitude, axis=axes, keepdims=True)
    )
    scale = stop(scale, backend)
    safe = xp.where((scale > 0) & xp.isfinite(scale), scale, 1)
    unit = magnitude / safe
    squares = (
        (unit * unit).sum(dim=axes, keepdim=True)
        if backend == "torch"
        else (unit * unit).sum(axis=axes, keepdims=True)
    )
    root = xp.sqrt(xp.where(squares == 0, 1, squares))
    result = xp.where(squares == 0, 0, scale * root)
    if keepdims:
        return result
    return result.squeeze(dim=axes) if backend == "torch" else xp.squeeze(result, axis=axes)


def global_norm(values, xp, backend):
    """Reduce native tensor leaves without concatenating their storage."""
    values = tuple(values)
    if not values:
        return xp.asarray(0.0)
    if backend == "torch" and all(
        value.device.type == "cpu"
        and value.dtype in {xp.float16, xp.bfloat16, xp.float32, xp.complex64}
        for value in values
    ):
        # Float64 accumulation spans every feasible float32 sum of squares.
        # Use native CPU reductions to avoid a sequence of eager scaling kernels.
        norms = tuple(
            xp.linalg.vector_norm(value, dtype=xp.complex128 if value.is_complex() else xp.float64)
            for value in values
        )
        return xp.linalg.vector_norm(xp.stack(norms))
    norms = tuple(vector_norm(value, xp, backend) for value in values)
    stacked = xp.stack(norms)
    return vector_norm(stacked, xp, backend)
