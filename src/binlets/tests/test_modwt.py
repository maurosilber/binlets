import hypothesis
import hypothesis.extra.numpy as st_np
import hypothesis.strategies as st
import numpy as np

from .. import modwt


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
def test_round_trip_1d(data):
    max_level = int(np.log2(min(data.shape)))
    axes = range(len(data.shape))

    for level in range(max_level):
        approx, details = modwt.modwt_level(data, level, axes)
        data_rec = modwt.imodwt_level(approx, details, level, axes)
        assert np.allclose(data_rec, data)


@hypothesis.given(
    data=st_np.arrays(
        np.float64,
        shape=st_np.array_shapes(max_dims=4, max_side=32),
        elements=st.floats(
            min_value=-1e6,
            max_value=1e6,
            allow_infinity=False,
            allow_nan=False,
        ),
    )
)
def test_round_trip(data):
    max_level = int(np.log2(min(data.shape)))
    axes = range(len(data.shape))

    for level in range(max_level):
        approx, details = modwt.modwt_level_nd(data, level, axes)
        data_rec = modwt.imodwt_level_nd(approx, details, level, axes)
        assert np.allclose(data_rec, data)
