__all__ = [
    "Grid",
    "Interpolator",
    "Interval",
    "Mesh",
    "Method",
]

from dataclasses import dataclass
from math import isfinite
from typing import Generic, Literal, TypeAlias, final

import numpy as np
from interpn import interpn

from lick._typing import F, FArray1D, FArray2D

Method: TypeAlias = Literal["nearest", "linear", "cubic"]


@final
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

    def as_evenly_spaced_array(self, size: int, *, dtype: F) -> FArray1D[F]:
        return np.linspace(self.min, self.max, size, dtype=dtype)


@final
@dataclass(slots=True, frozen=True)
class Monotonic(Generic[F]):
    base: FArray1D[F]

    def __post_init__(self):
        sorted_base = np.sort(self.base)
        if np.all(self.base == sorted_base) or np.all(self.base[::-1] == sorted_base):
            return
        raise ValueError(
            "Expected a monotonic base array (either increasing or decreasing order)"
        )

    def is_decreasing(self) -> bool:
        return bool(self.base[-1] < self.base[0])

    def as_increasing_array(self) -> FArray1D[F]:
        if self.is_decreasing():
            return self.base[::-1]
        else:
            return self.base


@final
@dataclass(kw_only=True, slots=True, frozen=True)
class Grid(Generic[F]):
    x: FArray1D[F]
    y: FArray1D[F]

    def __post_init__(self):
        if self.y.dtype == self.x.dtype and self.y.ndim == self.x.ndim == 1:
            return

        raise TypeError(
            "x and y must be 1D arrays with the same data type. "
            f"Got x.ndim={self.x.ndim}, x.dtype={self.x.dtype!s}, "
            f"y.ndim={self.y.ndim}, y.dtype={self.y.dtype!s}"
        )

    @classmethod
    def from_intervals(
        cls,
        *,
        x: Interval,
        y: Interval,
        small_dim_npoints: int,
        dtype: F,
    ) -> "Grid[F]":
        s = small_dim_npoints
        if s < 2:
            raise ValueError(f"Received {small_dim_npoints=}, expected at least 2")
        if (xy_ratio := x.span / y.span) >= 1:
            size_x = int(s * xy_ratio)
            size_y = s
        else:
            size_x = s
            size_y = int(s / xy_ratio)

        return Grid(
            x=x.as_evenly_spaced_array(size_x, dtype=dtype),
            y=y.as_evenly_spaced_array(size_y, dtype=dtype),
        )

    @classmethod
    def from_unsanitized_arrays(cls, x: FArray1D[F], y: FArray1D[F]) -> "Grid[F]":
        return Grid(
            x=Monotonic(x).as_increasing_array(),
            y=Monotonic(y).as_increasing_array(),
        )

    @property
    def dtype(self) -> np.dtype[F]:
        return self.x.dtype


@final
@dataclass(kw_only=True, slots=True, frozen=True)
class Mesh(Generic[F]):
    x: FArray2D[F]
    y: FArray2D[F]

    def __post_init__(self):
        if (
            self.y.dtype == self.x.dtype
            and self.y.ndim == self.x.ndim == 2
            and self.y.shape == self.x.shape
        ):
            return

        raise TypeError(
            "x and y must be 2D arrays with the same data type and shape. "
            f"Got x.shape={self.x.shape}, x.dtype={self.x.dtype!s}, "
            f"y.shape={self.y.shape}, y.dtype={self.y.dtype!s}"
        )

    @classmethod
    def from_grid(cls, grid: Grid[F], *, indexing: Literal["xy", "ij"]) -> "Mesh[F]":
        x, y = np.meshgrid(grid.x, grid.y, indexing=indexing)
        return Mesh(x=x, y=y)

    @property
    def dtype(self) -> np.dtype[F]:
        return self.x.dtype

    @property
    def shape(self) -> tuple[int, int]:
        return self.x.shape

    def astype(self, dtype: F, /) -> "Mesh[F]":
        return Mesh(x=self.x.astype(dtype), y=self.y.astype(dtype))


@final
@dataclass(kw_only=True, slots=True, frozen=True)
class Interpolator(Generic[F]):
    grid: Grid[F]
    target_mesh: Mesh[F]

    def __call__(
        self,
        vals: FArray2D[F],
        /,
        *,
        method: Method,
    ) -> FArray2D[F]:
        if any(o.dtype != self.grid.dtype for o in (vals, self.target_mesh)):
            raise TypeError(
                f"Expected all inputs to match this interpolator's grid data type ({self.grid.dtype}). "
                f"Received {vals.dtype=!s}, target_mesh.dtype={self.target_mesh.dtype!s}"
            )

        # https://github.com/la-niche/lick/issues/246
        # if vals.shape != target_mesh.shape:
        #    raise TypeError("Mismatched shapes between inputs. "
        #    f"Received {vals.shape=}, {target_mesh.shape=}"
        # )

        return interpn(
            grids=(self.grid.x, self.grid.y),
            obs=(self.target_mesh.x, self.target_mesh.y),
            vals=vals,
            method=method,
        )
