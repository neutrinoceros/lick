from collections.abc import Iterable, Iterator
from itertools import chain, combinations_with_replacement, permutations
from typing import TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pytest

from lick import interpol, lick, lick_box, lick_box_plot


@pytest.mark.parametrize("indexing", ["ij", "xy"])
def test_indexing(indexing):
    # regression test for https://github.com/la-niche/lick/issues/218
    nx = 32
    x = np.geomspace(0.1, 10, nx)
    y = np.geomspace(0.1, 5, nx)
    XX, YY = np.meshgrid(x, y, indexing=indexing)
    V1 = np.cos(XX)
    V2 = np.sin(YY)
    field = V1**2 + V2**2
    lick_box(
        XX,
        YY,
        V1,
        V2,
        field,
        size_interpolated=nx,
        method="nearest",
        xmin=1,
        xmax=9,
        ymin=1,
        ymax=4,
        niter_lic=1,
        kernel_length=3,
    )


X = np.linspace(0.0, 10.0, 32, dtype="float64")
Y = np.linspace(0.0, 20.0, 64, dtype="float64")
XX, YY = np.meshgrid(X, Y)
V1 = np.full_like(XX, 1.0)
V2 = np.full_like(XX, 2.0)
FIELD = np.full_like(XX, 3.0)

T = TypeVar("T")


class TestMixedDTypes:
    @staticmethod
    def all_mixed_combinations(values: Iterable[T], r: int) -> Iterator[T]:
        mixed_combinations = list(
            filter(
                lambda C: any(_ != C[0] for _ in C[1:]),
                combinations_with_replacement(values, r),
            )
        )
        return chain.from_iterable(permutations(c) for c in mixed_combinations)

    def test_interpol(self, subtests):
        for dt_xx, dt_yy, dt_v1, dt_v2, dt_field in self.all_mixed_combinations(
            ["float32", "float64"], 5
        ):
            with (
                subtests.test(
                    dt_xx=dt_xx,
                    dt_yy=dt_yy,
                    dt_v1=dt_v1,
                    dt_v2=dt_v2,
                    dt_field=dt_field,
                ),
                pytest.raises(TypeError, match="^Received inputs with mixed datatypes"),
            ):
                interpol(
                    XX.astype(dt_xx),
                    YY.astype(dt_yy),
                    V1.astype(dt_v1),
                    V2.astype(dt_v2),
                    FIELD.astype(dt_field),
                )

    def test_lick(self, subtests):
        for dt_v1, dt_v2 in self.all_mixed_combinations(["float32", "float64"], 2):
            with (
                subtests.test(dt_v1=dt_v1, dt_v2=dt_v2),
                pytest.raises(TypeError, match="^Received inputs with mixed datatypes"),
            ):
                lick(V1.astype(dt_v1), V2.astype(dt_v2))

    def test_lick_box(self, subtests):
        for dt_x, dt_y, dt_v1, dt_v2, dt_field in self.all_mixed_combinations(
            ["float32", "float64"], 5
        ):
            with (
                subtests.test(
                    dt_x=dt_x,
                    dt_y=dt_y,
                    dt_v1=dt_v1,
                    dt_v2=dt_v2,
                    dt_field=dt_field,
                ),
                pytest.raises(TypeError, match="^Received inputs with mixed datatypes"),
            ):
                lick_box(
                    X.astype(dt_x),
                    Y.astype(dt_y),
                    V1.astype(dt_v1),
                    V2.astype(dt_v2),
                    FIELD.astype(dt_field),
                )

    def test_lick_box_plot(self, subtests):
        fig, ax = plt.subplots()
        for dt_x, dt_y, dt_v1, dt_v2, dt_field in self.all_mixed_combinations(
            ["float32", "float64"], 5
        ):
            with (
                subtests.test(
                    dt_x=dt_x,
                    dt_y=dt_y,
                    dt_v1=dt_v1,
                    dt_v2=dt_v2,
                    dt_field=dt_field,
                ),
                pytest.raises(TypeError, match="^Received inputs with mixed datatypes"),
            ):
                lick_box_plot(
                    fig,
                    ax,
                    X.astype(dt_x),
                    Y.astype(dt_y),
                    V1.astype(dt_v1),
                    V2.astype(dt_v2),
                    FIELD.astype(dt_field),
                )
