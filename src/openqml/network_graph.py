"""Sparse quantum edge messages with permutation-equivariant aggregation."""

import numpy as np

from .errors import ValidationError


def quantum_messages(ctx):
    x, c, xp, data = ctx.inputs, ctx.spec.config, ctx.xp, ctx.context
    edge = data.get("edge_index")
    if x.ndim != 3 or edge is None:
        raise ValidationError("Graph operators require nodes[B,N,F] and edge_index[B,E,2].")
    if "int" not in str(edge.dtype) or "uint" in str(edge.dtype):
        raise ValidationError("edge_index must use a signed integer dtype.")
    batch, nodes, width = x.shape
    edge = ctx.array(edge, integer=True)
    count = edge.shape[1]
    edge_mask = data.get("edge_mask")
    if edge_mask is not None and tuple(edge_mask.shape) != (batch, count):
        raise ValidationError("edge_mask must have shape [B,E].")
    enabled = ctx.array(
        np.ones((batch, count), dtype=bool) if edge_mask is None else edge_mask, boolean=True
    )
    valid = (edge >= 0) & (edge < nodes)
    valid = valid.all(dim=-1) if ctx.backend == "torch" else xp.all(valid, axis=-1)
    bad = enabled & ~valid
    traced = False
    if ctx.backend == "jax":
        import jax

        traced = isinstance(bad, jax.core.Tracer)
    if not traced and bool(bad.any()):
        raise ValidationError("An active graph edge has an out-of-range node index.")
    safe = edge.clamp(0, nodes - 1) if ctx.backend == "torch" else xp.clip(edge, 0, nodes - 1)
    row = ctx.array(np.arange(batch)[:, None], integer=True)
    node_mask = data.get("node_mask", data.get("mask"))
    node_mask = ctx.array(
        np.ones((batch, nodes), dtype=bool) if node_mask is None else node_mask, boolean=True
    )
    enabled = enabled & valid & node_mask[row, safe[..., 0]] & node_mask[row, safe[..., 1]]
    features, positions = data.get("edge_features"), data.get("positions")
    if c["edge_features"]:
        if features is None or tuple(features.shape) != (batch, count, c["edge_features"]):
            raise ValidationError("Declared graph edge feature width differs from edge_features.")
    elif features is not None and features.shape[-1]:
        raise ValidationError("Declare edge_features in the graph operator specification.")
    if c["geometric"] and (
        positions is None or positions.ndim != 3 or tuple(positions.shape[:2]) != (batch, nodes)
    ):
        raise ValidationError("Geometric graph messages require positions[B,N,D].")
    messages, destinations, degrees = [], [], []
    for start in range(0, count, c["chunk_size"]):
        stop = min(count, start + c["chunk_size"])
        source, destination = safe[:, start:stop, 0], safe[:, start:stop, 1]
        values = [x[row, destination], x[row, source]]
        if c["edge_features"]:
            values.append(features[:, start:stop])
        if c["geometric"]:
            relative = positions[row, source] - positions[row, destination]
            relative = xp.where(enabled[:, start:stop, None], relative, 0)
            values.append(ctx.sum(relative * relative, -1, True))
        active = enabled[:, start:stop, None]
        ctx.check_workspace(batch * (stop - start) * (2 * width + c["edge_features"] + 1))
        inputs = xp.where(active, ctx.cat(values), 0)
        messages.append(ctx.project("decode", ctx.kernel(inputs)) * active)
        destinations.append(destination)
        degrees.append(active)
    if not count:
        return x * 0
    messages, destinations = ctx.cat(messages, axis=1), ctx.cat(destinations, axis=1)
    degrees = ctx.cat(degrees, axis=1)
    if ctx.backend == "torch":
        messages = messages.to(x.dtype)
        result = xp.zeros_like(x).scatter_add(
            1, destinations[..., None].expand(-1, -1, width), messages
        )
        normalizer = xp.zeros_like(x[..., :1]).scatter_add(
            1, destinations[..., None], degrees.to(x.dtype)
        )
    else:
        result, normalizer = xp.zeros_like(x), xp.zeros_like(x[..., :1])
        if ctx.backend == "jax":
            result = result.at[row, destinations].add(messages)
            normalizer = normalizer.at[row, destinations].add(degrees.astype(x.dtype))
        else:
            np.add.at(result, (row, destinations), messages)
            np.add.at(normalizer, (row, destinations), degrees.astype(x.dtype))
    if c["aggregation"] == "mean":
        normalizer = (
            normalizer.clamp_min(1) if ctx.backend == "torch" else xp.maximum(normalizer, 1)
        )
        result = result / normalizer
    if traced:
        # A compiled graph cannot raise a Python exception. Invalid active edges
        # poison only their graph; the integrated finite-gradient check rejects
        # the update. Masked -1 padding is valid and contributes nothing.
        invalid = xp.any(bad, axis=-1)
        result = xp.where(invalid[:, None, None], xp.nan, result)
    return result
