import numpy as np
import pytest

from lick import lick_box


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
