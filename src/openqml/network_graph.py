"""Sparse quantum edge messages with permutation-equivariant aggregation."""

import numpy as np

from .errors import ValidationError


def lift_graph_pairs(ctx):
    """Lift nodes and sparse edges to ordered-pair states, without node IDs.

    Pair axes transform together under relabeling. Edge multiplicities and
    edge features are summed; diagonal indicators distinguish equality. Dense
    storage is explicit and admitted before constructing quadratic arrays.
    """
    x, c, xp, data = ctx.inputs, ctx.spec.config, ctx.xp, ctx.context
    edge = data.get("edge_index")
    if x.ndim != 3 or edge is None:
        raise ValidationError("Pair graphs require nodes[B,N,F] and edge_index[B,E,2].")
    batch, nodes, width = x.shape
    feature_width = 2 * width + 2 + c["edge_features"] + int(c["geometric"])
    ctx.check_workspace(
        batch * nodes * nodes * (feature_width + 3 * ctx.architecture.hidden_features)
    )
    if "int" not in str(edge.dtype) or "uint" in str(edge.dtype):
        raise ValidationError("edge_index must use a signed integer dtype.")
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
    mask = data.get("node_mask", data.get("mask"))
    mask = ctx.array(np.ones((batch, nodes), dtype=bool) if mask is None else mask, boolean=True)
    enabled = enabled & valid & mask[row, safe[..., 0]] & mask[row, safe[..., 1]]
    values = [enabled[..., None] * ctx.array(1)]
    features = data.get("edge_features")
    if c["edge_features"]:
        if features is None or tuple(features.shape) != (batch, count, c["edge_features"]):
            raise ValidationError("Declared graph edge feature width differs from edge_features.")
        values.append(xp.where(enabled[..., None], ctx.array(features), 0))
    elif features is not None and features.shape[-1]:
        raise ValidationError("Declare edge_features in the pair operator specification.")
    edge_values = ctx.cat(values)
    pair_index = safe[..., 0] * nodes + safe[..., 1]
    storage = ctx.zeros((batch, nodes * nodes, edge_values.shape[-1]))
    if ctx.backend == "torch":
        storage = storage.scatter_add(1, pair_index[..., None].expand_as(edge_values), edge_values)
    elif ctx.backend == "jax":
        storage = storage.at[row, pair_index].add(edge_values)
    else:
        np.add.at(storage, (row, pair_index), edge_values)
    storage = storage.reshape((batch, nodes, nodes, -1))
    pair_mask = mask[:, :, None] & mask[:, None, :]
    a = xp.broadcast_to(x[:, :, None, :], (batch, nodes, nodes, width))
    b = xp.broadcast_to(x[:, None, :, :], (batch, nodes, nodes, width))
    indices = ctx.array(np.arange(nodes), integer=True)
    diagonal = xp.broadcast_to(
        (indices[:, None] == indices[None, :])[None, ..., None] * ctx.array(1),
        (batch, nodes, nodes, 1),
    )
    values = [a, b, storage, diagonal]
    if c["geometric"]:
        positions = data.get("positions")
        if positions is None or positions.ndim != 3 or tuple(positions.shape[:2]) != (batch, nodes):
            raise ValidationError("Geometric pair graphs require positions[B,N,D].")
        ctx.check_workspace(batch * nodes * nodes * (feature_width + positions.shape[-1]))
        positions = xp.where(mask[..., None], ctx.array(positions), 0)
        relative = positions[:, :, None] - positions[:, None, :]
        values.append(ctx.sum(relative * relative, -1, True))
    result = xp.where(pair_mask[..., None], ctx.cat(values), 0)
    if traced:
        invalid = xp.any(bad, axis=-1)
        result = xp.where(invalid[:, None, None, None], xp.nan, result)
        pair_mask = pair_mask | invalid[:, None, None]
    return result, {"mask": pair_mask}


def quantum_pairs(ctx):
    """Channelwise matrix products aggregate through an intermediate node k.

    Quantum work is quadratic (one circuit per pair), not cubic (one circuit
    per triple). The classical contraction still has O(N^3 H) arithmetic.
    This factorized finite-width model does not inherit a full WL theorem.
    """
    x, c, xp = ctx.inputs, ctx.spec.config, ctx.xp
    if x.ndim != 4 or x.shape[1] != x.shape[2]:
        raise ValidationError("Pair blocks require [B,N,N,H] hidden states.")
    batch, nodes, _, hidden = x.shape
    ctx.check_workspace(
        batch
        * nodes
        * nodes
        * (5 * hidden + len(ctx.architecture.cells[ctx.index].to_spec().observables))
    )
    flat = x.reshape((batch, nodes * nodes, hidden))
    features = ctx.cat(
        [
            ctx.kernel(flat[:, i : i + c["chunk_size"]])
            for i in range(0, nodes * nodes, c["chunk_size"])
        ],
        axis=1,
    )
    features = features.reshape((batch, nodes, nodes, -1))
    joined = ctx.cat((x, features))
    mask = ctx.array(ctx.context["mask"], boolean=True)
    left = xp.tanh(ctx.project("pair_left", joined)) * mask[..., None]
    right = xp.tanh(ctx.project("pair_right", joined)) * mask[..., None]
    # The same pair features feed both factors: no triple-index circuit batch.
    contraction = xp.einsum("bikc,bkjc->bijc", left, right)
    count = ctx.sum(mask, -1, True)
    count = count.clamp_min(1) if ctx.backend == "torch" else xp.maximum(count, 1)
    contraction = contraction / xp.sqrt(count[..., None])
    return (ctx.project("decode", features) + ctx.project("pair_contract", contraction)) * mask[
        ..., None
    ]


def invariant_pair_readout(value, context, backend):
    from .simulator import _TensorOps

    xp = _TensorOps(backend, value).xp
    mask = context["mask"]
    if backend == "torch":
        mask = xp.as_tensor(mask, device=value.device, dtype=xp.bool)
        all_pairs = value.sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp_min(1)[..., None]
        ids = xp.arange(value.shape[1], device=value.device)
        diagonal = (
            value[:, ids, ids].sum(dim=1) / mask[:, ids, ids].sum(dim=1).clamp_min(1)[..., None]
        )
        return xp.cat((all_pairs, diagonal), dim=-1)
    mask = xp.asarray(mask)
    all_pairs = xp.sum(value, axis=(1, 2)) / xp.maximum(xp.sum(mask, axis=(1, 2)), 1)[..., None]
    ids = xp.arange(value.shape[1])
    diagonal = (
        xp.sum(value[:, ids, ids], axis=1)
        / xp.maximum(xp.sum(mask[:, ids, ids], axis=1), 1)[..., None]
    )
    return xp.concatenate((all_pairs, diagonal), axis=-1)


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
