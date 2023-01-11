from __future__ import annotations

from typing import Callable

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
    inputs: np.ndarray,
    *,
    levels: int | None = None,
    test: Callable[[np.ndarray], np.ndarray[bool]],
) -> tuple[np.ndarray]:
    """Binlets denoising.

    Parameters
    ----------
    inputs : ndarrays
        Data to denoise. Vector across last dimension.
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
    axes = range(inputs.ndim - 1)

    if levels is None:
        levels = int(np.log2(min(inputs.shape[:-1])))
    elif levels < 0:
        raise ValueError("Levels must be >= 0.")
    elif levels == 0:
        return inputs

    details_level = []
    # Decomposition
    for level in range(levels):
        coeffs = _binlet_level(inputs, level, test, axes)
        inputs = coeffs[0]
        details_level.append(coeffs)
    # Reconstruction
    for level, coeffs in reversed(list(enumerate(details_level))):
        coeffs[0] = inputs
        inputs = _ibinlet_level(coeffs, level, axes)

    return inputs


def _binlet_level(
    inputs: np.ndarray,
    level: int,
    test: Callable[[np.ndarray, np.ndarray], bool],
    axes: tuple[int],
):
    """Compute one level of the binlets transform."""
    # Calculate current level
    coeffs = modwt_nd(inputs, level, axes)
    approx = coeffs[0]
    details = coeffs[1:]

    # Threshold current level
    for axis, d in enumerate(details):
        data = imodwt_1d(approx, d, level, axis=axis)
        shifted = np.roll(data, -(2**level), axis=axis)
        mask = test(data, shifted)
        d[mask] = 0

    return coeffs


def _ibinlet_level(
    coeffs: np.ndarray,
    level: int,
    axes: tuple[int],
):
    return imodwt_nd(coeffs, level, axes)
