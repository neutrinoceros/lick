__all__ = ["Interval"]

from dataclasses import dataclass
from math import isfinite

import numpy as np

from lick._typing import FArray1D

f64 = np.float64


@dataclass(kw_only=True, slots=True, frozen=True)
class Interval:
    min: float
    max: float

    def __post_init__(self):
        if not (isfinite(self.min) and isfinite(self.max)):
            msg = "max and min must both be finite"
        elif self.max <= self.min:
            # 0-width intervals are not allowed so we can guarantee that
            # as_evenly_spaced_array always returns unique values
            msg = "max must be greater than min"
        else:
            return
        raise ValueError(f"{msg}. Got min={self.min}, max={self.max}")

    def with_overrides(
        self,
        *,
        min: float | None = None,
        max: float | None = None,
    ) -> "Interval":
        return Interval(
            min=float(min) if min is not None else self.min,
            max=float(max) if max is not None else self.max,
        )

    @property
    def span(self) -> float:
        return self.max - self.min

    def as_evenly_spaced_array(self, size: int) -> FArray1D[f64]:
        # always compute linspace in double precision to minimize round-off errors
        # in following computations. Memory overhead is negligible and downcasting is
        # trivial to do outside of the function if required.
        # TODO: consider exposing a dtype kwarg
        # (requires solid integration tests with single precision)
        return np.linspace(f64(self.min), f64(self.max), size)
