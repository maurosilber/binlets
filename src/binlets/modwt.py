from __future__ import annotations

import numpy as np


def modwt_1d(
    data: np.ndarray,
    level: int,
    axis: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """1D Haar MODWT transform.

    Parameters
    ----------
    data : ndarray
        Input signal.
    level : int
        Decomposition level. Must be >= 0.
    axis : int, optional
        Axis along which to apply the transform.
        By default, last dimension.

    Returns
    -------
    (approx, detail) : tuple[ndarrays]
        Approximation and detail coefficients.
    """
    shifted = np.roll(data, -(2**level), axis=axis)
    approx = shifted + data
    detail = shifted - data
    return approx, detail


def imodwt_1d(
    approx: np.ndarray,
    detail: np.ndarray,
    level: int,
    axis: int = -1,
) -> np.ndarray:
    """Inverse 1D Haar MODWT transform.

    Parameters
    ----------
    approx, detail : ndarray
        Approximation and detail coefficients.
    level : int
        Decomposition level. Must be >= 0.
    axis : int, optional
        Axis along which to apply the transform.
        By default, last dimension.

    Returns
    -------
    ndarray
        Reconstructed signal.
    """
    data = (approx - detail) / 2
    shifted = np.roll((approx + detail) / 2, 2**level, axis=axis)
    return (data + shifted) / 2


def modwt_nd(
    data: np.ndarray,
    level: int,
    axes: tuple[int],
) -> list[np.ndarray]:
    """nD Haar MODWT transform.

    Parameters
    ----------
    data : ndarray
        Input signal.
    level : int
        Decomposition level. Must be >= 0.
    axes : tuple[int]
        Axes along which to apply the transform.

    Returns
    -------
    coeffs : list[ndarray]
        Approximation and detail coefficients.
    """
    coeffs = [data]
    for axis in axes:
        new_coeffs = []
        for x in coeffs:
            new_coeffs.extend(modwt_1d(x, level, axis))
        coeffs = new_coeffs
    return coeffs


def imodwt_nd(
    coeffs: list[np.ndarray],
    level: int,
    axes: tuple[int],
) -> np.ndarray:
    """nD Haar MODWT transform."""
    for axis in reversed(axes):
        pairwise = zip(coeffs[0::2], coeffs[1::2])
        coeffs = [imodwt_1d(A, D, level, axis) for A, D in pairwise]
    return coeffs[0]


def modwt_mask_1d(mask, level, axis=-1):
    mask &= np.roll(mask, -(2**level), axis=axis)
    return mask


def modwt_mask_nd(mask, level, axes):
    for axis in axes:
        mask = modwt_mask_1d(mask, level, axis)
    return mask
