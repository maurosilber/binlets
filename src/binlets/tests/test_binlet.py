import itertools

import hypothesis
import hypothesis.extra.numpy as st_np
import hypothesis.strategies as st
import numpy as np
import scipy.ndimage

from .._binlets import binlets, max_level


@hypothesis.given(
    data=st_np.arrays(
        np.float64,
        shape=st_np.array_shapes(min_dims=2, max_dims=4, max_side=16),
        elements=st.floats(-1e6, 1e6, allow_infinity=False, allow_nan=False),
    )
)
def test_round_trip(data):
    """As the test always returns False,
    it never thresholds any coefficient,
    and it must reconstruct the same input.
    """

    def always_false(x, y):
        return np.zeros_like(x, bool)

    for level in range(max_level(data)):
        rec = binlets(data, levels=level, test=always_false, linear=False)
        assert np.allclose(rec, data)


@hypothesis.given(
    size=st.integers(0, 4),
    ndim=st.integers(2, 4),
    seed=st.integers(0),
)
def test_full_threshold_power_of_two(size, ndim, seed):
    """As the test always returns True,
    all detail coefficients are thresholded,
    resulting in a full (ie, non-adaptive) binning of the input."""

    def always_true(x, y):
        return np.ones_like(x, bool)

    def average(data):
        return np.mean(data, axis=tuple(range(1, data.ndim)), keepdims=True)

    size = 2**size
    data = np.random.default_rng(seed).random(size=(size,) * ndim)

    level = max_level(data)
    rec = binlets(data, levels=level, test=always_true, linear=False)
    expected = average(data)
    assert np.allclose(rec, expected)


@hypothesis.given(
    data=st_np.arrays(
        np.float64,
        shape=st_np.array_shapes(
            min_dims=2,
            max_dims=4,
            min_side=2,  # nothing to do for "one pixel".
            max_side=8,
        ),
        elements=st.floats(-1e6, 1e6, allow_infinity=False, allow_nan=False),
    )
)
def test_full_threshold(data):
    """As the test always returns True,
    all detail coefficients are thresholded,
    resulting in a full (ie, non-adaptive) binning of the input."""

    def always_true(x, y):
        return np.ones_like(x, bool)

    def local_average(data, level):
        size = 2**level
        N = size ** (data.ndim - 1)
        shifts = itertools.product(
            *[range(-size // 2, size // 2) for _ in range(1, data.ndim)]
        )
        axis = tuple(range(1, data.ndim))
        filter_size = (1,) + (size,) * (data.ndim - 1)
        return (
            sum(
                scipy.ndimage.uniform_filter(
                    np.roll(data, shift, axis),
                    size=filter_size,
                    mode="wrap",
                )
                for shift in shifts
            )
            / N
        )

    level = max_level(data)
    rec = binlets(data, levels=level, test=always_true, linear=False)
    expected = local_average(data, level)
    assert np.allclose(rec, expected)


def test_maximum_norm_1d():
    data = np.array([[0, 0], [2, 2], [100, 100], [102, 200]]).T
    expected = np.array([[0.5, 0.5], [1.5, 1.5], [100, 100], [102, 200]]).T

    def maximum_norm(x, y):
        return np.all(np.abs(x - y) < 10, axis=0, keepdims=True)

    result = binlets(data, test=maximum_norm, linear=False)
    assert np.all(result == expected)
