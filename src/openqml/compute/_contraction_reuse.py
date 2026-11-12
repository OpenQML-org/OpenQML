"""Bounded per-evaluation common subexpressions for exact contraction trees."""

import math
from collections import OrderedDict


class ContractionReuse:
    def __init__(self, budget):
        # Symbolic identities belong to the compiled plan. This budget covers
        # numerical references and their keys only, with explicit entry overhead.
        self.capacity = budget
        self.values, self.bytes = OrderedDict(), 0
        self.peak_bytes = 0

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
        self.peak_bytes = max(self.peak_bytes, self.bytes)
