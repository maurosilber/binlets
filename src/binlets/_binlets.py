from __future__ import annotations

import inspect
from typing import Callable

import numpy as np

from ._modwt import (
    Haar,
    Wavelet,
    _modwt_mask_inplace,
    imodwt_nd,
    modwt_nd,
    undo_1d_modwt,
)


def max_level(data: np.ndarray) -> int:
    """Returns the biggest level L such that 2**L <= min(data.shape)."""
    shape = data.shape[1:]  # exclude channel dimension
    return min(shape).bit_length() - 1


def binlets(
    inputs: np.ndarray,
    *,
    levels: int | None = None,
    test: Callable[[np.ndarray, np.ndarray, int], np.ndarray[bool]],
    linear: bool,
    wavelet: Wavelet = Haar,
) -> tuple[np.ndarray]:
    """Binlets denoising.

    Parameters
    ----------
    inputs : ndarrays
        Data to denoise. First dimension corresponds to channels,
        and the rest to spatial dimensions.
    levels : int | None
        Decomposition level. Must be >= 0. If == 0, does nothing.
        `None` sets maximum possible level for the input data.
    test : Callable
        Test to threshold detail coefficients.
    linear : bool
        True if the test performs a linear transformation of the coefficients.

    Returns
    -------
    tuple of ndarrays
        Tuple of denoised inputs.
    """
    approx = inputs
    axes = range(1, approx.ndim)

    if levels is None:
        levels = max_level(approx)
    elif levels < 0:
        raise ValueError("Levels must be >= 0.")
    elif levels == 0:
        return approx

    test = _check_test(test)

    coeffs_level = []
    # Decomposition
    for level in range(levels):
        coeffs = modwt_nd(approx, level, axes, wavelet=wavelet)
        approx = coeffs[0]
        coeffs_level.append(coeffs)

    # Threshold details
    if linear:
        _binlets_thresholding_linear(coeffs_level, test=test, wavelet=wavelet)
    else:
        _binlets_thresholding_nonlinear(coeffs_level, test=test, wavelet=wavelet)

    # Reconstruction
    for level, coeffs in reversed(list(enumerate(coeffs_level))):
        coeffs[0] = approx
        approx = imodwt_nd(coeffs, level, axes, wavelet=wavelet)

    return approx


def _binlets_thresholding_linear(
    coefficients: list[list[np.ndarray]],
    *,
    test: Callable[[np.ndarray, np.ndarray], np.ndarray[bool]],
    wavelet: Wavelet,
):
    for level, (approx, *details) in enumerate(coefficients):
        for detail in details:
            x, y = undo_1d_modwt(approx, detail, wavelet=wavelet)
            mask = test(x, y, level=level)
            mask = np.broadcast_to(mask, detail.shape)
            detail[mask] = 0


def _binlets_thresholding_nonlinear(
    coefficients: list[list[np.ndarray]],
    *,
    test: Callable[[np.ndarray, np.ndarray], np.ndarray[bool]],
    wavelet: Wavelet,
):
    approx_0, *details_0 = coefficients[0]
    masks = [np.ones(approx_0.shape, dtype=bool) for _ in details_0]
    axes = range(approx_0.ndim)
    for level, (approx, *details) in enumerate(coefficients):
        for mask, detail in zip(masks, details):
            x, y = undo_1d_modwt(approx, detail, wavelet=wavelet)
            mask = _modwt_mask_inplace(mask, level, axes, wavelet=wavelet)
            mask &= test(x, y, level=level)
            detail[mask] = 0


def _check_test(
    test: Callable[[np.ndarray, np.ndarray], np.ndarray[bool]]
    | Callable[[np.ndarray, np.ndarray, int], np.ndarray[bool]],
) -> Callable[[np.ndarray, np.ndarray, int], np.ndarray[bool]]:
    """Check that the test callable has a level parameter.

    If not, it wraps test with a callable that does accept level.
    """
    signature = inspect.signature(test)
    if "level" in signature.parameters:
        return test

    def test_with_level(x, y, *, level: int):
        return test(x, y)

    return test_with_level
