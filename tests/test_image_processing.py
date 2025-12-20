import numpy as np
import numpy.testing as npt
import pytest

from lick._image_processing import Identity, NorthWestLightSource


def test_identity():
    prng = np.random.default_rng()
    array = prng.random((8, 7))
    processor = Identity()
    result = processor.process(array)
    npt.assert_array_equal(result, array)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_north_west_list_source(dtype):
    prng = np.random.default_rng()
    array = prng.random((8, 7)).astype(dtype)
    processor = NorthWestLightSource()
    result = processor.process(array)
    assert result.dtype == dtype
    assert not np.any(result == array)
