"""Exact Pauli contractions over one evolved open-boundary MPS."""

import numpy as np

from ..compute.kernel import get_compute_kernel

# (bra index, ket index, Pauli matrix element)
_ELEMENTS = {
    "I": ((0, 0, 1), (1, 1, 1)),
    "X": ((0, 1, 1), (1, 0, 1)),
    "Y": ((0, 1, -1j), (1, 0, 1j)),
    "Z": ((0, 0, 1), (1, 1, -1)),
}


def _left(env, site, pauli, kernel):
    result = np.zeros((site.shape[2], site.shape[2]), dtype=complex)
    for s, t, coefficient in _ELEMENTS[pauli]:
        product = kernel.matmul(env, site[:, t, :])
        result += coefficient * kernel.matmul(site[:, s, :].conj().T, product)
    return result


def read_words(sites, order, words, workspace):
    """Share exact identity environments; retain only right boundaries actually used."""
    width = len(sites)
    unit = np.ones((1, 1), dtype=complex)
    kernel = get_compute_kernel()
    values = {}
    straight = len(words) * width
    groups, boundaries = {}, set()
    shared_steps = straight
    # The direct path retains one environment and no width-sized strings per term.
    if len(words) > 1 and 512 * len(words) <= workspace:
        inner = 0
        for word in words:
            first = next(i for i, q in enumerate(order) if word[q] != "I")
            stop = next(i + 1 for i in range(width - 1, first - 1, -1) if word[order[i]] != "I")
            groups.setdefault(first, []).append((word, stop))
            boundaries.add(stop)
            inner += stop - first
        shared_steps = max(groups) + width - min(boundaries) + inner
        storage = 512 * len(words) + sum(16 * sites[b - 1].shape[2] ** 2 for b in boundaries)
        if storage > workspace:
            shared_steps = straight
    if shared_steps >= straight:
        for word in words:
            env = unit
            for site, wire in zip(sites, order, strict=True):
                env = _left(env, site, word[wire], kernel)
            values[word] = float(env[0, 0].real)
        return values, straight

    right, env = {width: unit}, unit
    for i in range(width - 1, min(boundaries) - 1, -1):
        a, b = sites[i][:, 0, :], sites[i][:, 1, :]
        env = kernel.matmul(kernel.matmul(a, env), a.conj().T) + kernel.matmul(
            kernel.matmul(b, env), b.conj().T
        )
        if i in boundaries:
            right[i] = env
    left = unit
    last_start = max(groups)
    for i in range(last_start + 1):
        for word, stop in groups.get(i, ()):
            env = left
            for j in range(i, stop):
                env = _left(env, sites[j], word[order[j]], kernel)
            values[word] = float(kernel.einsum("ab,ba->", env, right[stop]).real)
        if i < last_start:
            left = _left(left, sites[i], "I", kernel)
    return values, shared_steps
