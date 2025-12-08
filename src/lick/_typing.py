__all__ = ["F", "FArray1D", "FArray2D", "FArrayND"]

from typing import TypeAlias, TypeVar

import numpy as np

f32 = np.float32
f64 = np.float64
F = TypeVar("F", f32, f64)

FArray1D: TypeAlias = np.ndarray[tuple[int], np.dtype[F]]
FArray2D: TypeAlias = np.ndarray[tuple[int, int], np.dtype[F]]
FArrayND: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[F]]
