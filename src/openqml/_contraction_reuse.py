"""Bounded per-evaluation common subexpressions for exact contraction trees."""

import math
from collections import OrderedDict


class ContractionReuse:
    def __init__(self, budget):
        # Half for symbolic interning, half for live tensor references. Neither
        # bindings nor tensors survive a forward; autodiff adds shared adjoints.
        self.slots = budget // 1024
        self.capacity = budget // 2
        self.nodes, self.values, self.bytes = {}, OrderedDict(), 0

    def node(self, key):
        if key in self.nodes:
            return self.nodes[key]
        if len(self.nodes) >= self.slots:
            return None
        index = len(self.nodes)
        self.nodes[key] = index
        return index

    def get(self, key):
        if key is None or key not in self.values:
            return None
        value, _ = self.values[key]
        self.values.move_to_end(key)
        return value

    def put(self, key, value):
        if key is None or key in self.values:
            return
        itemsize = value.element_size() if hasattr(value, "element_size") else value.dtype.itemsize
        size = math.prod(value.shape) * itemsize + 256
        if size > self.capacity:
            return
        while self.values and self.bytes + size > self.capacity:
            _, (_, old_size) = self.values.popitem(last=False)
            self.bytes -= old_size
        self.values[key] = (value, size)
        self.bytes += size
