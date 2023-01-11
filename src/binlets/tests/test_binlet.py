import numpy as np

from ..binlets import binlets


def maximum_norm(x, y):
    return np.all(np.abs(x - y) < 10, axis=-1)


def test_maximum_norm():
    data = np.array([[0, 0], [2, 2], [100, 100], [102, 200]])
    expected = np.array([[0.5, 0.5], [1.5, 1.5], [100, 100], [102, 200]])

    result = binlets(data, test=maximum_norm)
    assert np.all(result == expected)
