"""Shape-specialized GEMMs and demand-driven, slice-dependent subtree reuse."""

from dataclasses import dataclass

from .kernel import get_compute_kernel


@dataclass(frozen=True)
class ContractionKernel:
    left_axes: tuple
    right_axes: tuple
    left_rank: int
    right_rank: int
    m: int
    k: int
    n: int
    output_rank: int

    @classmethod
    def compile(cls, la, lb, shared):
        la, lb, shared = tuple(la), tuple(lb), tuple(shared)
        names = {label: i for i, label in enumerate(dict.fromkeys(la + lb))}
        signature = tuple(tuple(names[x] for x in row) for row in (la, lb, shared))
        return get_compute_kernel().plan(
            "contraction", signature, lambda: cls._compile(la, lb, shared)
        )

    @classmethod
    def _compile(cls, la, lb, shared):
        a, b = tuple(x for x in la if x not in shared), tuple(x for x in lb if x not in shared)
        return cls(
            tuple(la.index(x) for x in a + shared),
            tuple(lb.index(x) for x in shared + b),
            len(la),
            len(lb),
            1 << len(a),
            1 << len(shared),
            1 << len(b),
            len(a) + len(b),
        )

    def apply(self, left, right, ops):
        na, nb = left.ndim - self.left_rank, right.ndim - self.right_rank
        pa = tuple(range(na)) + tuple(na + i for i in self.left_axes)
        pb = tuple(range(nb)) + tuple(nb + i for i in self.right_axes)
        left = ops.transpose(left, pa).reshape(left.shape[:na] + (self.m, self.k))
        right = ops.transpose(right, pb).reshape(right.shape[:nb] + (self.k, self.n))
        value = ops.matmul(left, right)
        return value.reshape(value.shape[:-2] + (2,) * self.output_rank)


class ContractionProgram:
    """One immutable topology; only numerical values live in a forward-local cache."""

    def __init__(self, graph, namespace, intern):
        desc, original, _, steps, sliced, _, _, _ = graph
        slots = {index: i for i, index in enumerate(sliced)}
        keys, masks, selectors = [], [], []

        def identity(signature):
            if signature not in intern:
                intern[signature] = len(intern)
            return intern[signature]

        for (kind, op, bra), row in zip(desc, original, strict=True):
            value = (op.name, op.params, op.matrix) if kind == "gate" else op
            origin = (kind, value, bra if kind == "gate" else False, len(row))
            selector = tuple(slots.get(i, -1) for i in row)
            dependent = tuple(i for i in selector if i >= 0)
            signature = ("sliced", namespace, origin, selector) if dependent else ("input", origin)
            keys.append(identity(signature))
            masks.append(sum(1 << i for i in set(dependent)))
            selectors.append(selector)
        kernels = []
        for a, b, la, lb, shared, _ in steps:
            kernel = ContractionKernel.compile(la, lb, shared)
            kernels.append(kernel)
            keys.append(identity(("contract", keys[a], keys[b], kernel)))
            masks.append(masks[a] | masks[b])
        self.keys, self.masks = tuple(keys), tuple(masks)
        self.selectors, self.kernels = tuple(selectors), tuple(kernels)
        self.steps, self.input_count = steps, len(desc)

    def run(self, raw, assignment, cache, ops, statistics=None):
        root = self.input_count + len(self.steps) - 1
        pending, values = [(root, False)], {}
        while pending:
            node, ready = pending.pop()
            leaf = node < self.input_count
            key = ("tree", self.keys[node], assignment & self.masks[node])
            cached = None if leaf else cache.get(key)
            if cached is not None:
                values[node] = cached
                if statistics is not None:
                    statistics["subtree_cache_hits"] += 1
                continue
            if leaf:
                value = raw[node]
                selector = self.selectors[node]
                prefix = value.ndim - len(selector)
                values[node] = value[
                    (slice(None),) * prefix
                    + tuple(
                        (assignment >> bit) & 1 if bit >= 0 else slice(None) for bit in selector
                    )
                ]
            elif ready:
                k = node - self.input_count
                a, b = self.steps[k][:2]
                value = self.kernels[k].apply(values.pop(a), values.pop(b), ops)
                values[node] = value
                cache.put(key, value)
                if statistics is not None:
                    statistics["contractions_executed"] += 1
            else:
                a, b = self.steps[node - self.input_count][:2]
                pending.extend(((node, True), (b, False), (a, False)))
        return values[root]

    def run_flat(self, raw, assignment, ops, statistics=None):
        """Execute precompiled slots without per-node dictionaries or tree walks."""
        values = [None] * (self.input_count + len(self.steps))
        for node, (value, selector) in enumerate(zip(raw, self.selectors, strict=True)):
            if self.masks[node]:
                prefix = value.ndim - len(selector)
                value = value[
                    (slice(None),) * prefix
                    + tuple(
                        (assignment >> bit) & 1 if bit >= 0 else slice(None) for bit in selector
                    )
                ]
            values[node] = value
        for k, ((a, b, *_), kernel) in enumerate(zip(self.steps, self.kernels, strict=True)):
            value = kernel.apply(values[a], values[b], ops)
            values[a] = values[b] = None
            values[self.input_count + k] = value
        if statistics is not None:
            statistics["contractions_executed"] += len(self.steps)
        return values[-1]
