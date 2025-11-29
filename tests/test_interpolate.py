import numpy as np
import numpy.testing as npt
import pytest

from lick._interpolate import Interval

f64 = np.float64


@pytest.mark.parametrize(
    "inputs",
    [
        pytest.param({"min": +1.0, "max": -1.0}, id="unsorted"),
        pytest.param({"min": 1.0, "max": 1.0}, id="zero-span"),
    ],
)
def test_interval_invalid_extrema(inputs):
    min = inputs["min"]
    max = inputs["max"]
    with pytest.raises(
        ValueError,
        match=rf"^max must be greater than min\. Got {min=}, {max=}$",
    ):
        Interval(min=min, max=max)


@pytest.mark.parametrize(
    "inputs",
    [
        pytest.param({"min": float("nan"), "max": 1.0}, id="minnan"),
        pytest.param({"min": 0.0, "max": float("nan")}, id="maxnan"),
        pytest.param({"min": float("nan"), "max": float("nan")}, id="allnans"),
        pytest.param({"min": float("-inf"), "max": 1.0}, id="mininf"),
        pytest.param({"min": 0.0, "max": float("inf")}, id="maxinf"),
        pytest.param({"min": float("inf"), "max": float("inf")}, id="allinfs"),
    ],
)
def test_interval_nonfinite_extrema(inputs):
    min = inputs["min"]
    max = inputs["max"]
    with pytest.raises(
        ValueError,
        match=rf"^max and min must both be finite\. Got {min=}, {max=}$",
    ):
        Interval(min=min, max=max)


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"min": -2.0, "max": 2.0}, id="min+max"),
        pytest.param({"min": -2.0}, id="min"),
        pytest.param({"max": 2.0}, id="max"),
        pytest.param({}, id="no-override"),
    ],
)
def test_interval_with_overrides(overrides):
    L1 = Interval(min=-1.0, max=1.0)
    L2 = L1.with_overrides(**overrides)
    assert L2 is not L1
    if overrides:
        assert L2 != L1
    else:
        assert L2 == L1
    assert L2.min == overrides.get("min", L1.min)
    assert L2.max == overrides.get("max", L1.max)


@pytest.mark.parametrize(
    "interval, expected",
    [
        pytest.param(Interval(min=0, max=1.0), 1.0, id="0_pos"),
        pytest.param(Interval(min=-1.0, max=0.0), 1.0, id="neg_0"),
        pytest.param(Interval(min=-1.0, max=1.0), 2.0, id="neg_pos"),
    ],
)
def test_interval_span(interval, expected):
    assert interval.span == expected


@pytest.mark.parametrize(
    "interval",
    [
        pytest.param(Interval(min=0, max=1.0), id="0_pos"),
        pytest.param(Interval(min=-1.0, max=0.0), id="neg_0"),
        pytest.param(Interval(min=-1.0, max=1.0), id="neg_pos"),
    ],
)
@pytest.mark.parametrize("size", [5, 8, 64, 128])
def test_interval_as_evenly_spaced_array(interval, size):
    a = interval.as_evenly_spaced_array(size)

    assert a.dtype is np.dtype("float64")
    assert a.shape == (size,)
    npt.assert_almost_equal(np.diff(a, 2), 0.0)
