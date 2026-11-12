"""Bounded single-flight caches; independent keys compile concurrently."""

from collections import OrderedDict
from concurrent.futures import Future
from threading import RLock, get_ident

from ..errors import ValidationError


class SingleFlightCache:
    def __init__(self, entries, max_bytes=None, size=None):
        self.capacity, self.max_bytes, self.size = entries, max_bytes, size
        self.values, self.pending, self.lock = OrderedDict(), {}, RLock()
        self.bytes = self.hits = self.misses = self.evictions = self.epoch = 0

    def get(self, key, build):
        with self.lock:
            if key in self.values:
                self.hits += 1
                self.values.move_to_end(key)
                return self.values[key][0]
            epoch = self.epoch
            flight = epoch, key
            waiting = self.pending.get(flight)
            if waiting is None:
                future = Future()
                self.pending[flight] = future, get_ident()
                self.misses += 1
            else:
                future, owner = waiting
                if owner == get_ident():
                    raise ValidationError("Recursive compilation of the same cache key.")
                self.hits += 1
        if waiting is not None:
            return future.result()
        try:
            value = build()
            size = 0 if self.size is None else self.size(key, value)
        except BaseException as exc:
            with self.lock:
                self.pending.pop(flight, None)
                future.set_exception(exc)
            raise
        with self.lock:
            if (
                epoch == self.epoch
                and self.capacity
                and (self.max_bytes is None or size <= self.max_bytes)
            ):
                while self.values and (
                    len(self.values) >= self.capacity
                    or (self.max_bytes is not None and self.bytes + size > self.max_bytes)
                ):
                    _, (_, removed) = self.values.popitem(last=False)
                    self.bytes -= removed
                    self.evictions += 1
                self.values[key] = value, size
                self.bytes += size
            self.pending.pop(flight, None)
            future.set_result(value)
        return value

    def info(self):
        with self.lock:
            return {
                "entries": len(self.values),
                "bytes": self.bytes,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "max_entries": self.capacity,
                "max_bytes": self.max_bytes,
            }

    def clear(self):
        with self.lock:
            self.epoch += 1
            self.values.clear()
            self.bytes = self.hits = self.misses = self.evictions = 0
