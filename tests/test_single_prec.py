import numpy as np

from lick import lick


def test_single_prec():
    x = np.linspace(0.1, 10, 256, dtype="float32")
    y = np.linspace(0.1, 5, 128, dtype="float32")
    XX, YY = np.meshgrid(x, y, indexing="xy")
    V1 = np.cos(XX)
    V2 = np.sin(YY)

    lick(
        V1,
        V2,
        niter_lic=1,
        kernel="auto-adjust",
        post_lic=None,
    )
