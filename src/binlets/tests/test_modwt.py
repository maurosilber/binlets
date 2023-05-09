import hypothesis
import hypothesis.extra.numpy as st_np
import hypothesis.strategies as st
import numpy as np
import pytest

from .. import _modwt
from .._binlets import Haar, undo_1d_modwt
from .._modwt import NormalizedHaar


@pytest.mark.parametrize("wavelet", [Haar, NormalizedHaar])
@hypothesis.given(
    data=st_np.arrays(
        np.float64,
        shape=st_np.array_shapes(max_dims=1, max_side=64),
        elements=st.floats(
            min_value=-1e6,
            max_value=1e6,
            allow_infinity=False,
            allow_nan=False,
        ),
    )
)
def test_round_trip_1d(data, wavelet):
    max_level = int(np.log2(min(data.shape)))
    axes = range(len(data.shape))

    for level in range(max_level):
        approx, details = _modwt.modwt_1d(data, level, axes, wavelet=wavelet)
        data_rec = _modwt.imodwt_1d(approx, details, level, axes, wavelet=wavelet)
        assert np.allclose(data_rec, data)


@pytest.mark.parametrize("wavelet", [Haar, NormalizedHaar])
@hypothesis.given(
    data=st_np.arrays(
        np.float64,
        shape=st_np.array_shapes(max_dims=4, max_side=16),
        elements=st.floats(
            min_value=-1e6,
            max_value=1e6,
            allow_infinity=False,
            allow_nan=False,
        ),
    )
)
def test_round_trip(data, wavelet):
    max_level = int(np.log2(min(data.shape)))
    axes = range(len(data.shape))

    for level in range(max_level):
        coeffs = _modwt.modwt_nd(data, level, axes, wavelet=wavelet)
        data_rec = _modwt.imodwt_nd(coeffs, level, axes, wavelet=wavelet)
        assert np.allclose(data_rec, data)


@pytest.mark.parametrize("wavelet", [Haar, NormalizedHaar])
@hypothesis.given(
    data=st_np.arrays(
        np.float64,
        shape=st_np.array_shapes(max_dims=1, max_side=64),
        elements=st.floats(
            min_value=-1e6,
            max_value=1e6,
            allow_infinity=False,
            allow_nan=False,
        ),
    )
)
def test_reverse_1d(data, wavelet):
    level = 0
    approx, details = _modwt.modwt_1d(data, level, wavelet=wavelet)
    x, y = undo_1d_modwt(approx, details, wavelet=wavelet)
    assert np.allclose(x, data)
    assert np.allclose(y, np.roll(data, -(2**level)))
