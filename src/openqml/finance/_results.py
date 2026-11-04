"""Shared financial results without loading an execution engine."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PayoffEstimate:
    value: object
    intervals: tuple
    confidence: float
    amplitude: object

    @property
    def interval(self):
        return self.intervals[0][0], self.intervals[-1][1]

    @property
    def oracle_queries(self):
        return 0 if self.amplitude is None else self.amplitude.oracle_queries
