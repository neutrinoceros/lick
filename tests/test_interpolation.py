import numpy as np
import pytest

from lick import interpol

prng = np.random.default_rng(0)
shape = (4, 3)
size = np.prod(shape)
imgs = [prng.random(size, dtype=dt).reshape(shape) for dt in ("float32", "float64")]
min_vals = {
    "none": None,
    "int": 1,
    "float": 1.0,
    "f32": np.float32(1),
    "f64": np.float64(1),
}
max_vals = {
    "none": None,
    "int": 10,
    "float": 10.0,
    "f32": np.float32(10),
    "f64": np.float64(10),
}


@pytest.mark.parametrize(
    "xv", [np.linspace(0, 10, shape[0], dtype=dt) for dt in ("float32", "float64")]
)
@pytest.mark.parametrize(
    "yv", [np.linspace(0, 20, shape[1], dtype=dt) for dt in ("float32", "float64")]
)
@pytest.mark.parametrize("xmin", min_vals.values(), ids=min_vals.keys())
@pytest.mark.parametrize("xmax", max_vals.values(), ids=max_vals.keys())
@pytest.mark.parametrize("ymin", min_vals.values(), ids=min_vals.keys())
@pytest.mark.parametrize("ymax", max_vals.values(), ids=max_vals.keys())
@pytest.mark.parametrize("indexing", ["ij", "xy"])
@pytest.mark.parametrize("field", imgs)
@pytest.mark.parametrize("v1", imgs)
@pytest.mark.parametrize("v2", imgs)
def test_variable_precision_interpol_inputs(
    xv, yv, xmin, xmax, ymin, ymax, indexing, field, v1, v2, subtests
):
    xx, yy = np.meshgrid(xv, yv, indexing=indexing)
    xo, yo, v1o, v2o, fieldo = interpol(
        xx,
        yy,
        field,
        v1,
        v2,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        size_interpolated=3,
    )

    with subtests.test():
        assert yo.dtype == xo.dtype
    with subtests.test():
        assert v2o.dtype == v1o.dtype
    with subtests.test():
        assert fieldo.dtype == v2o.dtype
