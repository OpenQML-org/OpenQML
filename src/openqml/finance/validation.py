"""Chronological, label-interval-aware financial validation splits."""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from ..errors import ValidationError
from ..ir.core import integer


class FinancialFold(NamedTuple):
    train: np.ndarray
    test: np.ndarray
    purged: np.ndarray


@dataclass(frozen=True)
class PurgedWalkForward:
    train_size: int
    test_size: int
    step_size: int | None = None
    gap: int = 0
    expanding: bool = True

    def __post_init__(self):
        for name in ("train_size", "test_size"):
            object.__setattr__(self, name, integer(getattr(self, name), name, 1))
        object.__setattr__(
            self,
            "step_size",
            self.test_size if self.step_size is None else integer(self.step_size, "step_size", 1),
        )
        object.__setattr__(self, "gap", integer(self.gap, "gap", 0))
        if type(self.expanding) is not bool:
            raise ValidationError("expanding must be boolean.")

    def split(self, times, label_end):
        """Purge every training label touching a test decision timestamp.

        gap adds a pre-test gap in observations. Fit features, atom selection
        and calibration on training rows; assess distribution drift separately.
        """
        times, ends = np.asarray(times), np.asarray(label_end)
        if times.ndim != 1 or ends.shape != times.shape or not len(times):
            raise ValidationError("times and label_end must be matching nonempty vectors.")
        if times.dtype.kind not in "Mifu" or ends.dtype.kind != times.dtype.kind:
            raise ValidationError("Use matching numeric or NumPy datetime timestamp types.")
        if times.dtype.kind == "M":
            valid = ~np.isnat(times) & ~np.isnat(ends)
        else:
            valid = np.isfinite(times) & np.isfinite(ends)
        if not np.all(valid) or np.any(times[1:] <= times[:-1]) or np.any(ends < times):
            raise ValidationError(
                "Decision times must increase strictly; labels must end "
                "at or after their decisions."
            )
        count = 0
        for start in range(
            self.train_size + self.gap, len(times) - self.test_size + 1, self.step_size
        ):
            stop = start - self.gap
            first = 0 if self.expanding else max(0, stop - self.train_size)
            candidates = np.arange(first, stop)
            keep = ends[candidates] < times[start]
            training = candidates[keep]
            if not len(training):
                raise ValidationError("Label purging removed the entire training fold.")
            count += 1
            yield FinancialFold(
                training, np.arange(start, start + self.test_size), candidates[~keep]
            )
        if not count:
            raise ValidationError("Insufficient observations for a complete train/test fold.")
