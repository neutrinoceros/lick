__all__ = ["FloatT", "FArray1D", "FArray2D", "FArrayND"]

from typing import TypeAlias, TypeVar

import numpy as np

FloatT = TypeVar("FloatT", np.float32, np.float64)
FArray1D: TypeAlias = np.ndarray[tuple[int], np.dtype[FloatT]]
FArray2D: TypeAlias = np.ndarray[tuple[int, int], np.dtype[FloatT]]
FArrayND: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[FloatT]]
