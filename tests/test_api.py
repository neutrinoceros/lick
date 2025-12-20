import pytest

from lick import _api
from lick._image_processing import Identity, NorthWestLightSource


def test_get_kernel_invalid_call():
    with pytest.raises(
        TypeError,
        match=(
            r"^kernel and kernel_length keyword arguments are mutually exclusive, "
            r"but both were received\.$"
        ),
    ):
        _api.get_kernel(1, size=2, max_auto_size=10, dtype="float32")


def test_get_kernel_length_depr():
    with pytest.warns(
        DeprecationWarning,
        match=(
            r"^The kernel_length keyword argument is deprecated since lick v0\.10\.0 "
            r"and will be removed in a future version. Use the kernel argument instead\.$"
        ),
    ):
        kernel = _api.get_kernel(
            _api.UNSET,
            size=9,
            max_auto_size=_api.LegacyDefault.KERNEL_SIZE.value,
            dtype="float32",
        )
    assert kernel.size == 9
    assert kernel.dtype == "float32"


def test_get_kernel_default():
    with pytest.warns(
        DeprecationWarning,
        match=(
            r"^The kernel argument was not explicitly specified. "
            r"Its default value will change from 'auto-legacy' to 'auto-adjust' in a future release.\n"
            r"To silence this warning, set the argument explicitly\.$"
        ),
    ):
        kernel = _api.get_kernel(
            _api.UNSET, size=_api.UNSET, max_auto_size=10, dtype="float32"
        )

    # max_auto_size is *supposed* to be ignored by default as of lick 0.10.0
    assert kernel.size == _api.LegacyDefault.KERNEL_SIZE.value
    assert kernel.dtype == "float32"


def test_get_kernel_legacy_value():
    with pytest.warns(
        PendingDeprecationWarning,
        match=(
            r"^The kernel argument was set to 'auto\-legacy'\. "
            r"This value is scheduled for removal in a future version\. "
            r"To silence this warning, set "
            r"kernel=np\.sin\(np\.linspace\(0, np\.pi, 101, endpoint=False\)\)\.astype\(<ref_dtype>\)$"
        ),
    ):
        kernel = _api.get_kernel(
            _api.LegacyDefault.KERNEL.value,
            size=_api.UNSET,
            max_auto_size=10,
            dtype="float32",
        )

    # max_auto_size is *supposed* to be ignored by default as of lick 0.10.0
    assert kernel.size == _api.LegacyDefault.KERNEL_SIZE.value
    assert kernel.dtype == "float32"


@pytest.mark.parametrize("max_auto_size", [1, 2, 10])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_get_kernel_auto_small(max_auto_size, dtype):
    kernel = _api.get_kernel(
        "auto-adjust", size=_api.UNSET, max_auto_size=max_auto_size, dtype=dtype
    )
    assert kernel.size == max_auto_size
    assert kernel.size < _api.LegacyDefault.KERNEL_SIZE.value
    assert kernel.dtype == dtype


@pytest.mark.parametrize(
    "max_auto_size",
    [
        _api.LegacyDefault.KERNEL_SIZE.value + 1,
        _api.LegacyDefault.KERNEL_SIZE.value * 2,
    ],
)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_get_kernel_auto_large(max_auto_size, dtype):
    kernel = _api.get_kernel(
        "auto-adjust", size=_api.UNSET, max_auto_size=max_auto_size, dtype=dtype
    )
    assert kernel.size < max_auto_size
    assert kernel.size == 21
    assert kernel.dtype == dtype


def test_get_niter_lic_default():
    with pytest.warns(
        DeprecationWarning,
        match=(
            r"^The niter_lic argument was not explicitly specified. "
            r"Its default value will change from 5 to 1 in a future release.\n"
            r"To silence this warning, set the argument explicitly\.$"
        ),
    ):
        n = _api.get_niter_lic(_api.UNSET)

    assert n == _api.LegacyDefault.NITER_LIC.value


@pytest.mark.parametrize("niter", [1, 2, 10])
def test_get_niter_lic_explicit(niter):
    n = _api.get_niter_lic(niter)
    assert n == niter


def test_get_post_lic_invalid_call():
    with pytest.raises(
        TypeError,
        match=(
            r"post_lic and light_source keyword arguments are "
            r"mutually exclusive, but both were received\."
        ),
    ):
        _api.get_post_lic("north-west-light-source", light_source=True)


def test_get_post_lic_all_default():
    with pytest.warns(
        DeprecationWarning,
        match=(
            r"^The post_lic argument was not explicitly specified. "
            r"Its default value will change from 'north-west-light-source' to None in a future release\.\n"
            r"To silence this warning, set the argument explicitly\.$"
        ),
    ):
        post_lic = _api.get_post_lic(_api.UNSET, light_source=_api.UNSET)
    assert isinstance(post_lic, NorthWestLightSource)


@pytest.mark.parametrize(
    "light_source_bool, cls",
    [
        (True, NorthWestLightSource),
        (False, Identity),
    ],
)
def test_get_post_lic_light_source_bool(light_source_bool, cls):
    with pytest.warns(
        DeprecationWarning,
        match=(
            r"^The light_source keyword argument is deprecated since lick v0\.10\.0 "
            r"and will be removed in a future version\. Use the post_lic argument instead\.$"
        ),
    ):
        post_lic = _api.get_post_lic(_api.UNSET, light_source=light_source_bool)
    assert isinstance(post_lic, cls)


def test_get_post_lic_none():
    post_lic = _api.get_post_lic(None, light_source=_api.UNSET)
    assert isinstance(post_lic, Identity)
