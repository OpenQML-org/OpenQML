"""OpenQML CUDA statevector kernels. Import only after selecting a supported device."""

from math import prod

import torch
import triton
import triton.language as tl


@triton.jit
def _apply_gate(
    state,
    gate,
    output,
    GROUPS: tl.constexpr,
    MASKS: tl.constexpr,
    OFFSETS: tl.constexpr,
    BLOCK: tl.constexpr,
    BATCH_SHAPE: tl.constexpr,
    STATE_STRIDES: tl.constexpr,
    MATRIX_STRIDES: tl.constexpr,
    STATE_LAST: tl.constexpr,
    GATE_ROW: tl.constexpr,
    GATE_COL: tl.constexpr,
    STATE_CONJ: tl.constexpr,
    GATE_CONJ: tl.constexpr,
):
    # A lane owns every amplitude of one local subspace. Insert zero bits at
    # target positions, then OR in the local basis bits in circuit wire order.
    tile = tl.program_id(0).to(tl.int64)
    tiles = tl.cdiv(GROUPS, BLOCK)
    batch = tile // tiles
    group = (tile % tiles) * BLOCK + tl.arange(0, BLOCK)
    active = group < GROUPS
    base = group
    for bit in tl.static_range(len(MASKS)):
        low = MASKS[bit] - 1
        base = ((base & ~low) << 1) | (base & low)
    dimension: tl.constexpr = len(OFFSETS)
    start = batch * (GROUPS * dimension)
    input_start, gate_start = batch * 0, batch * 0
    remainder = batch
    for axis in tl.static_range(len(BATCH_SHAPE)):
        digit = remainder % BATCH_SHAPE[axis]
        input_start += digit * STATE_STRIDES[axis]
        gate_start += digit * MATRIX_STRIDES[axis]
        remainder = remainder // BATCH_SHAPE[axis]
    matrix = gate + 2 * gate_start
    # Read each input amplitude once and reuse it across local output rows.
    re = ()
    im = ()
    for col in tl.static_range(dimension):
        index = 2 * (input_start + (base | OFFSETS[col]) * STATE_LAST)
        re += (tl.load(state + index, active, 0.0),)
        value = tl.load(state + index + 1, active, 0.0)
        im += (-value if STATE_CONJ else value,)
    for row in tl.static_range(dimension):
        yr = tl.full((BLOCK,), 0, state.dtype.element_ty)
        yi = tl.full((BLOCK,), 0, state.dtype.element_ty)
        for col in tl.static_range(dimension):
            element = matrix + 2 * (row * GATE_ROW + col * GATE_COL)
            mr, mi = tl.load(element), tl.load(element + 1)
            if GATE_CONJ:
                mi = -mi
            yr += mr * re[col] - mi * im[col]
            yi += mr * im[col] + mi * re[col]
        index = 2 * (start + (base | OFFSETS[row]))
        tl.store(output + index, yr, active)
        tl.store(output + index + 1, yi, active)


def launch_gate(state, matrix, qubits, width):
    from contextlib import nullcontext

    dimension = 1 << len(qubits)
    groups = state.shape[-1] // dimension
    masks = tuple(1 << (width - 1 - q) for q in qubits)
    offsets = tuple(
        sum(mask for j, mask in enumerate(reversed(masks)) if (local >> j) & 1)
        for local in range(dimension)
    )
    output = torch.empty_like(state, memory_format=torch.contiguous_format)
    if state.numel() == 0:
        return output
    state = state.resolve_neg()
    matrix = matrix.resolve_neg().expand(state.shape[:-1] + (dimension, dimension))
    block = 64
    grid = (state.numel() // state.shape[-1] * triton.cdiv(groups, block),)
    # The interpreter accepts host tensors. Production dispatch supplies CUDA tensors.
    with torch.cuda.device(state.device) if state.is_cuda else nullcontext():
        _apply_gate[grid](
            torch.view_as_real(state.conj() if state.is_conj() else state),
            torch.view_as_real(matrix.conj() if matrix.is_conj() else matrix),
            torch.view_as_real(output),
            groups,
            tuple(sorted(masks)),
            offsets,
            block,
            tuple(reversed(state.shape[:-1])),
            state.stride()[:-1][::-1],
            matrix.stride()[:-2][::-1],
            state.stride(-1),
            matrix.stride(-2),
            matrix.stride(-1),
            state.is_conj(),
            matrix.is_conj(),
            num_warps=2,
        )
    return output


@triton.jit
def _gate_cotangent(
    state,
    grad,
    partial,
    GROUPS: tl.constexpr,
    COUNT: tl.constexpr,
    PARTS: tl.constexpr,
    DIM: tl.constexpr,
    MASKS: tl.constexpr,
    WIRE_BITS: tl.constexpr,
    SHAPE: tl.constexpr,
    BROADCAST: tl.constexpr,
    XS: tl.constexpr,
    GS: tl.constexpr,
    X_LAST: tl.constexpr,
    G_LAST: tl.constexpr,
    X_CONJ: tl.constexpr,
    G_CONJ: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    element = pid // PARTS
    row, col = (element // DIM) % DIM, element % DIM
    x_local, g_local = col * 0, row * 0
    for i in tl.static_range(len(WIRE_BITS)):
        x_local += ((col >> i) & 1) * WIRE_BITS[i]
        g_local += ((row >> i) & 1) * WIRE_BITS[i]
    out_batch = element // (DIM * DIM)
    x_base, g_base = x_local * X_LAST, g_local * G_LAST
    for i in tl.static_range(len(SHAPE)):
        if not BROADCAST[i]:
            x_base += (out_batch % SHAPE[i]) * XS[i]
            g_base += (out_batch % SHAPE[i]) * GS[i]
            out_batch = out_batch // SHAPE[i]
    re = tl.full((BLOCK,), 0, state.dtype.element_ty)
    im = tl.full((BLOCK,), 0, state.dtype.element_ty)
    for tile in range(tl.cdiv(COUNT, PARTS * BLOCK)):
        index = ((pid % PARTS) + tile * PARTS) * BLOCK + tl.arange(0, BLOCK)
        group = index % GROUPS
        for i in tl.static_range(len(MASKS)):
            low = MASKS[i] - 1
            group = ((group & ~low) << 1) | (group & low)
        a, b = x_base + group * X_LAST, g_base + group * G_LAST
        replica = index // GROUPS
        for i in tl.static_range(len(SHAPE)):
            if BROADCAST[i]:
                a += (replica % SHAPE[i]) * XS[i]
                b += (replica % SHAPE[i]) * GS[i]
                replica = replica // SHAPE[i]
        xr = tl.load(state + 2 * a, index < COUNT, 0.0)
        xi = tl.load(state + 2 * a + 1, index < COUNT, 0.0)
        gr = tl.load(grad + 2 * b, index < COUNT, 0.0)
        gi = tl.load(grad + 2 * b + 1, index < COUNT, 0.0)
        if X_CONJ:
            xi = -xi
        if G_CONJ:
            gi = -gi
        re += gr * xr + gi * xi
        im += gi * xr - gr * xi
    tl.store(partial + 2 * pid, tl.sum(re, axis=0))
    tl.store(partial + 2 * pid + 1, tl.sum(im, axis=0))


def launch_gate_cotangent(state, grad, wires, width, matrix_shape):
    """Reduce spectator amplitudes and broadcast parameter batches directly into dU."""
    from contextlib import nullcontext

    dim = 1 << len(wires)
    batch = tuple(state.shape[:-1])
    target = (1,) * (len(batch) + 2 - len(matrix_shape)) + tuple(matrix_shape[:-2])
    if state.numel() == 0:
        return state.new_zeros(matrix_shape)
    broadcast = tuple(m == 1 and n != 1 for m, n in zip(target, batch, strict=True))
    groups = state.shape[-1] // dim
    count = groups * prod(n for n, shared in zip(batch, broadcast, strict=True) if shared)
    block = 128
    parts = min(32, triton.cdiv(count, block))
    from ..compute.kernel import get_compute_kernel

    get_compute_kernel()._workspace((state, grad), tuple(matrix_shape) + (parts,), buffers=2)
    partial = state.new_empty(tuple(matrix_shape) + (parts,))
    bits = tuple(1 << (width - 1 - q) for q in reversed(wires))
    state, grad = state.resolve_neg(), grad.resolve_neg()
    with torch.cuda.device(state.device) if state.is_cuda else nullcontext():
        _gate_cotangent[(partial.numel(),)](
            torch.view_as_real(state.conj() if state.is_conj() else state),
            torch.view_as_real(grad.conj() if grad.is_conj() else grad),
            torch.view_as_real(partial),
            groups,
            count,
            parts,
            dim,
            tuple(sorted(bits)),
            bits,
            batch[::-1],
            broadcast[::-1],
            state.stride()[:-1][::-1],
            grad.stride()[:-1][::-1],
            state.stride(-1),
            grad.stride(-1),
            state.is_conj(),
            grad.is_conj(),
            block,
            num_warps=4,
        )
    return partial.squeeze(-1) if parts == 1 else partial.sum(dim=-1)


@triton.jit
def _contract(
    left,
    right,
    output,
    count,
    BATCH: tl.constexpr,
    LS: tl.constexpr,
    RS: tl.constexpr,
    FREE_L: tl.constexpr,
    FREE_R: tl.constexpr,
    SUM_L: tl.constexpr,
    SUM_R: tl.constexpr,
    CONJ_L: tl.constexpr,
    CONJ_R: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    bits = index
    a = tl.full((BLOCK,), 0, tl.int64)
    b = tl.full((BLOCK,), 0, tl.int64)
    for axis in tl.static_range(len(FREE_R)):
        b += (bits & 1) * FREE_R[axis]
        bits = bits >> 1
    for axis in tl.static_range(len(FREE_L)):
        a += (bits & 1) * FREE_L[axis]
        bits = bits >> 1
    for axis in tl.static_range(len(BATCH)):
        digit = bits % BATCH[axis]
        a += digit * LS[axis]
        b += digit * RS[axis]
        bits = bits // BATCH[axis]
    yr = tl.full((BLOCK,), 0, left.dtype.element_ty)
    yi = tl.full((BLOCK,), 0, left.dtype.element_ty)
    for k in tl.static_range(len(SUM_L)):
        lhs = left + 2 * (a + SUM_L[k])
        rhs = right + 2 * (b + SUM_R[k])
        ar, ai = tl.load(lhs, index < count, 0.0), tl.load(lhs + 1, index < count, 0.0)
        br, bi = tl.load(rhs, index < count, 0.0), tl.load(rhs + 1, index < count, 0.0)
        if CONJ_L:
            ai = -ai
        if CONJ_R:
            bi = -bi
        yr += ar * br - ai * bi
        yi += ar * bi + ai * br
    tl.store(output + 2 * index, yr, index < count)
    tl.store(output + 2 * index + 1, yi, index < count)


def launch_contraction(plan, left, right):
    from contextlib import nullcontext

    def strides(value, rank, axes):
        n = value.ndim - rank
        expanded = value.expand(batch + value.shape[n:])
        return tuple(reversed(expanded.stride()[: len(batch)])), tuple(
            value.stride(n + axis) for axis in axes
        )

    def offsets(stride):
        return tuple(
            sum(s for bit, s in enumerate(reversed(stride)) if (i >> bit) & 1)
            for i in range(plan.k)
        )

    batch = torch.broadcast_shapes(
        left.shape[: left.ndim - plan.left_rank], right.shape[: right.ndim - plan.right_rank]
    )
    left = left.resolve_neg()
    right = right.resolve_neg()
    ls, la = strides(left, plan.left_rank, plan.left_axes)
    rs, ra = strides(right, plan.right_rank, plan.right_axes)
    contracted = plan.k.bit_length() - 1
    free_l = plan.left_rank - contracted
    result = left.new_empty(batch + (2,) * plan.output_rank)
    count, block = result.numel(), 128
    if count:
        with torch.cuda.device(left.device) if left.is_cuda else nullcontext():
            _contract[(triton.cdiv(count, block),)](
                torch.view_as_real(left.conj() if left.is_conj() else left),
                torch.view_as_real(right.conj() if right.is_conj() else right),
                torch.view_as_real(result),
                count,
                tuple(reversed(batch)),
                ls,
                rs,
                tuple(reversed(la[:free_l])),
                tuple(reversed(ra[contracted:])),
                offsets(la[free_l:]),
                offsets(ra[:contracted]),
                left.is_conj(),
                right.is_conj(),
                block,
            )
    return result


@triton.jit
def _pauli_pair(
    state,
    weight,
    output,
    PAIRS: tl.constexpr,
    FLIP: tl.constexpr,
    SIGN: tl.constexpr,
    PIVOT: tl.constexpr,
    PHASE: tl.constexpr,
    SHAPE: tl.constexpr,
    XS: tl.constexpr,
    WS: tl.constexpr,
    LAST: tl.constexpr,
    CONJ: tl.constexpr,
    BACKWARD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    tiles = tl.cdiv(PAIRS, BLOCK)
    row = pid // tiles
    pair = (pid % tiles) * BLOCK + tl.arange(0, BLOCK)
    a = ((pair & -PIVOT) << 1) | (pair & (PIVOT - 1))
    b = a ^ FLIP
    start, scale_offset = row * 0, row * 0
    batch = row
    for axis in tl.static_range(len(SHAPE)):
        digit = batch % SHAPE[axis]
        start += digit * XS[axis]
        if BACKWARD:
            scale_offset += digit * WS[axis]
        batch = batch // SHAPE[axis]
    ar = tl.load(state + 2 * (start + a * LAST), pair < PAIRS, 0.0)
    ai = tl.load(state + 2 * (start + a * LAST) + 1, pair < PAIRS, 0.0)
    br = tl.load(state + 2 * (start + b * LAST), pair < PAIRS, 0.0)
    bi = tl.load(state + 2 * (start + b * LAST) + 1, pair < PAIRS, 0.0)
    if CONJ:
        ai, bi = -ai, -bi
    bits = a & SIGN
    for n in tl.static_range(4):
        if SIGN >> (32 >> n):
            bits = bits ^ (bits >> (32 >> n))
    odd = (0x6996 >> (bits & 15)) & 1
    factor = tl.where(odd != 0, -2.0, 2.0)
    if PHASE >= 2:
        factor = -factor
    if BACKWARD:
        factor = factor * tl.load(weight + scale_offset)
        # grad <x|P|x> = 2 P x. The two matrix elements are conjugates.
        if PHASE & 1:
            dar, dai, dbr, dbi = bi, -br, -ai, ar
        else:
            dar, dai, dbr, dbi = br, bi, ar, ai
        dst = output + 2 * row * (2 * PAIRS)
        tl.store(dst + 2 * a, factor * dar, pair < PAIRS)
        tl.store(dst + 2 * a + 1, factor * dai, pair < PAIRS)
        tl.store(dst + 2 * b, factor * dbr, pair < PAIRS)
        tl.store(dst + 2 * b + 1, factor * dbi, pair < PAIRS)
    else:
        if PHASE & 1:
            component = ar * bi - ai * br
        else:
            component = ar * br + ai * bi
        tl.store(output + pid, tl.sum(factor * component, axis=0))


def launch_pauli_pair(state, word, weight=None):
    """Reduce disjoint amplitude pairs or apply their exact real-loss pullback."""
    from contextlib import nullcontext

    from ..compute.kernel import get_compute_kernel
    from ..ir.observables import _masks

    flip, sign = _masks(word[::-1])
    pairs, block = state.shape[-1] // 2, 256
    tiles = triton.cdiv(pairs, block)
    batch = tuple(state.shape[:-1])
    if weight is None:
        shape = batch + (tiles,)
        get_compute_kernel()._workspace((state,), shape, buffers=2)
        output = state.real.new_empty(shape)
        raw_output = output
    else:
        weight = weight.resolve_neg()
        get_compute_kernel()._workspace((state, weight), state.shape)
        output = torch.empty_like(state, memory_format=torch.contiguous_format)
        raw_output = torch.view_as_real(output)
    if state.numel():
        state = state.resolve_neg()
        data = torch.view_as_real(state.conj() if state.is_conj() else state)
        with torch.cuda.device(state.device) if state.is_cuda else nullcontext():
            _pauli_pair[(prod(batch) * tiles,)](
                data,
                data if weight is None else weight,
                raw_output,
                pairs,
                flip,
                sign,
                1 << (flip.bit_length() - 1),
                word.count("Y") % 4,
                batch[::-1],
                state.stride()[:-1][::-1],
                () if weight is None else weight.stride()[::-1],
                state.stride(-1),
                state.is_conj(),
                weight is not None,
                block,
            )
    if weight is not None:
        return output
    return output.squeeze(-1) if tiles == 1 else output.sum(dim=-1)
