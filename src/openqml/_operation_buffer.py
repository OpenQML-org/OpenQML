"""Persistent operation chunks with a lazily materialized tuple view."""

from dataclasses import dataclass, field
from itertools import chain

_CHUNK = 32


@dataclass(frozen=True, slots=True, eq=False)
class _Prefix:
    previous: object
    items: tuple


@dataclass(frozen=True, slots=True, eq=False)
class OperationBuffer:
    prefix: _Prefix | None
    tail: tuple
    size: int
    _flat: tuple | None = field(default=None, repr=False)

    @classmethod
    def from_tuple(cls, operations):
        return cls(None, operations, len(operations), operations)

    def append(self, operation):
        if len(self.tail) < _CHUNK:
            return type(self)(self.prefix, self.tail + (operation,), self.size + 1)
        return type(self)(_Prefix(self.prefix, self.tail), (operation,), self.size + 1)

    def extend(self, operations):
        if not operations:
            return self
        if len(self.tail) + len(operations) <= _CHUNK:
            return type(self)(self.prefix, self.tail + operations, self.size + len(operations))
        prefix = _Prefix(self.prefix, self.tail) if self.tail else self.prefix
        return type(self)(prefix, operations, self.size + len(operations))

    def as_tuple(self):
        value = self._flat
        if value is None:
            chunks, prefix = [self.tail], self.prefix
            while prefix is not None:
                chunks.append(prefix.items)
                prefix = prefix.previous
            value = tuple(chain.from_iterable(reversed(chunks)))
            # Concurrent first readers may compute the same immutable value.
            object.__setattr__(self, "_flat", value)
        return value


class OperationView:
    """Keep Circuit's public tuple field and dataclass defaults intact."""

    def __get__(self, instance, owner=None):
        return () if instance is None else instance._buffer.as_tuple()
