__all___ = [
    "Identity",
    "ImageProcessor",
    "NorthWestLightSource",
]
from typing import Protocol

from lick._typing import F, FArray2D


class ImageProcessor(Protocol):
    def process(self, image: FArray2D[F]) -> FArray2D[F]: ...


class Identity:
    __slots__ = []

    def process(self, image: FArray2D[F]) -> FArray2D[F]:
        return image


class NorthWestLightSource:
    __slots__ = []

    def process(self, image: FArray2D[F]) -> FArray2D[F]:
        from matplotlib.colors import LightSource

        ls = LightSource(azdeg=0.0, altdeg=45.0)
        return ls.hillshade(image, vert_exag=5).astype(image.dtype, copy=False)  # type: ignore[no-any-return]
