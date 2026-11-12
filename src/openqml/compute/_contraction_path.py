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
    if not steps:
        return steps
    children, root = _tree(labels, steps)
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
    current = min(candidates, key=score)
    current_score = score(current)
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
                proposed = _emit(labels, children, root, merge)
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
