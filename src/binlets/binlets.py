from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy import stats

from .modwt import imodwt_1d, imodwt_nd, modwt_nd


def test(*inputs: tuple[np.ndarray], threshold: float) -> bool:
    """
    v.T @ M @ v
    """
    raise NotImplementedError


def threshold_from_pvalue(p_value: float, *, ndim: int) -> float:
    """Assumes a (multivariate) normal distribution."""
    return stats.chi2.isf(p_value, df=ndim)


def _linear(approx, detail, level, axis):
    return approx, detail


def _non_linear(approx, detail, level, axis):
    return imodwt_1d(approx, detail, level, axis)


def binlets(
    inputs: Sequence[np.ndarray],
    *,
    levels: int | None = None,
    test: Callable[[tuple[np.ndarray]], np.ndarray[bool]],
) -> tuple[np.ndarray]:
    """Binlets denoising.

    Parameters
    ----------
    inputs : Sequence of ndarrays
        Data to denoise.
    levels : int
        Decomposition level. Must be >= 0. If == 0, does nothing.
        Sets maximum possible binning of size 2**level.
    threshold : float, optional
        Controls the level of denoising.

    Returns
    -------
    tuple of ndarrays
        Tuple of denoised inputs.
    """
    inputs = np.broadcast_arrays(*inputs)
    axes = range(inputs[0].ndim)

    if levels is None:
        levels = int(np.log2(min(inputs[0].shape)))
    elif levels < 0:
        raise ValueError("Levels must be >= 0.")
    elif levels == 0:
        return inputs

    details_level = []
    # Decomposition
    for level in range(levels):
        coeffs = _binlet_level(inputs, level, test, axes)
        inputs = [c[0] for c in coeffs]
        details_level.append(coeffs)
    # Reconstruction
    for level, coeffs in reversed(list(enumerate(details_level))):
        for i, c in enumerate(inputs):
            coeffs[i][0] = c
        inputs = _ibinlet_level(coeffs, level, axes)

    return inputs


def _binlet_level(
    inputs: tuple[np.ndarray],
    level: int,
    test: Callable[[np.ndarray, np.ndarray], bool],
    axes: tuple[int],
):
    """Compute one level of the binlets transform."""
    # Calculate current level
    coeffs = [modwt_nd(x, level, axes) for x in inputs]
    approx = [c[0] for c in coeffs]
    details = [c[1:] for c in coeffs]

    # Threshold current level
    for axis, details_axis in enumerate(zip(*details)):
        data = [imodwt_1d(a, d, level, axis=axis) for a, d in zip(approx, details_axis)]
        shifted = [np.roll(x, -(2**level), axis=axis) for x in data]
        mask = test(data, shifted)
        for d in details_axis:
            d[mask] = 0

    return coeffs


def _ibinlet_level(
    coeffs: tuple[np.ndarray],
    level: int,
    axes: tuple[int],
):
    return [imodwt_nd(c, level, axes) for c in coeffs]
