import numpy as np

from ..binlets import binlets


def maximum_norm(x, y):
    mask = True
    for xi, yi in zip(x, y):
        mask &= np.abs(xi - yi) < 10
    return mask


def test_maximum_norm():
    data = (
        np.array([0, 2, 100, 102]),
        np.array([0, 2, 100, 200]),
    )
    expected = (
        np.array([0.5, 1.5, 100, 102]),
        np.array([0.5, 1.5, 100, 200]),
    )

    result = binlets(data, test=maximum_norm)
    assert all(np.all(r[0] == e[0]) for r, e in zip(result, expected))
