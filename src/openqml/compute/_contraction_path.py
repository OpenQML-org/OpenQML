"""Deterministic path portfolios and bounded exact contraction-tree surgery."""


def _tree(labels, steps):
    children = {len(labels) + k: (s[0], s[1]) for k, s in enumerate(steps)}
    return children, len(labels) + len(steps) - 1


def _emit(labels, children, root, merge):
    """Emit a depth-first binary path without recursive Python calls."""
    identities, indices, steps = {}, {}, []
    pending = [(root, False)]
    while pending:
        node, ready = pending.pop()
        if node < len(labels):
            identities[node], indices[node] = node, labels[node]
        elif ready:
            a, b = children[node]
            la, lb = indices.pop(a), indices.pop(b)
            shared, out = merge(la, lb)
            steps.append((identities.pop(a), identities.pop(b), la, lb, shared, out))
            identities[node], indices[node] = len(labels) + len(steps) - 1, out
        else:
            a, b = children[node]
            pending.extend(((node, True), (b, False), (a, False)))
    return tuple(steps)


def depth_first(labels, steps, merge):
    """Minimize estimated peak over all depth-first child orders of this tree."""
    if not steps:
        return steps
    size = [1 << len(row) for row in labels]
    delta = [0] * len(labels)
    peak = [0] * len(labels)
    children = {}
    for k, (a, b, _, _, _, out) in enumerate(steps, len(labels)):
        output = 1 << len(out)
        # Raw leaves are live on entry. After a subtree completes the live
        # volume has changed by delta, which can be negative.
        join = delta[a] + delta[b] + 2 * (size[a] + size[b]) + 3 * output
        ab = max(peak[a], delta[a] + peak[b], join)
        ba = max(peak[b], delta[b] + peak[a], join)
        children[k] = (b, a) if ba < ab else (a, b)
        peak.append(min(ab, ba))
        delta.append(delta[a] + delta[b] + output - size[a] - size[b])
        size.append(output)
    root = len(size) - 1
    return _emit(labels, children, root, merge)


def portfolio(labels, charge, greedy, merge, profile, target_peak, trials):
    """Choose a feasible low-FLOP tree, then refine bounded three-way subtrees.

    Rotations preserve every index and tensor. Every candidate is profiled in
    its actual depth-first execution order, including simultaneous live inputs.
    """

    def score(path):
        charge(len(labels) + len(path))
        peak, flops, _ = profile(labels, path)
        return (False, flops, peak) if peak <= target_peak else (True, peak, flops)

    candidates = [
        depth_first(labels, greedy(labels, charge, objective=objective), merge)
        for objective in ("size", "flops", "balanced")
    ]
    current_score, current = min(
        ((score(path), path) for path in candidates), key=lambda entry: entry[0]
    )
    children, root = _tree(labels, current)
    # Visit expensive joins first. trials bounds candidate subtrees, while
    # charge accounts for the full path emitted/profiled for every proposal.
    ranked = sorted(
        range(len(current)),
        key=lambda k: len(current[k][2]) + len(current[k][3]) - len(current[k][4]),
        reverse=True,
    )
    for k in ranked[:trials]:
        parent = len(labels) + k
        for side in (0, 1):
            pair = children[parent]
            child, other = pair[side], pair[1 - side]
            if child < len(labels):
                continue
            first, second = children[child]
            for keep, remaining in ((first, second), (second, first)):
                old_parent, old_child = children[parent], children[child]
                children[child], children[parent] = (keep, other), (child, remaining)
                proposed = depth_first(labels, _emit(labels, children, root, merge), merge)
                proposed_score = score(proposed)
                if proposed_score < current_score:
                    current, current_score = proposed, proposed_score
                    break
                children[parent], children[child] = old_parent, old_child
            else:
                continue
            # Accepted rotations change this join's children; continue with
            # the next ranked join rather than reusing its old sibling pair.
            break
    return current


def slice_index(labels, steps, candidates, target_peak, trials, charge):
    """Rank separators by their effect on the whole live schedule and both slices."""
    charge(sum(map(len, candidates)))
    sizes = {}
    for row in candidates:
        for digit in row:
            sizes[digit] = sizes.get(digit, 0) + (1 << len(row))
    if not sizes:
        return None
    shortlist = sorted(sizes, key=lambda digit: (-sizes[digit], digit))[:trials]
    charge(len(shortlist) * (len(labels) + len(steps)))

    def score(digit):
        live = [1 << (len(row) - (digit in row)) for row in labels]
        total = peak = sum(live)
        work = 0
        for a, b, la, lb, shared, out in steps:
            size = 1 << (len(out) - (digit in out))
            peak = max(peak, total + 2 * (live[a] + live[b]) + 3 * size)
            work += 8 << (len(la) + len(lb) - len(shared) - int(digit in la or digit in lb))
            total += size - live[a] - live[b]
            live.append(size)
        return (False, 2 * work, peak) if peak <= target_peak else (True, peak, 2 * work)

    return min(shortlist, key=score)
