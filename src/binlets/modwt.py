from functools import reduce

import numpy as np


def modwt_1d(data, level, axis=-1, approx_only=False):
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
    approx_only : bool, optional
        If True, only computes approximation coefficients.
        By default, False.

    Returns
    -------
    (approx, detail) : tuple of ndarrays
        Approximation and detail coefficients.

    If approx_only is True, returns only approximation coefficients.
    """
    shifted = np.roll(data, -(2**level), axis=axis)
    approx = shifted + data
    if approx_only:
        return approx
    else:
        detail = shifted - data
        return approx, detail


def imodwt_1d(approx, detail, level, axis=-1):
    """Inverse 1D Haar MODWT transform.

    Parameters
    ----------
    approx, detail : numpy.array
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


def modwt_nd(data, level, axes, approx_only=False):
    """nD Haar MODWT transform.

    Parameters
    ----------
    data : ndarray
        Input signal.
    level : int
        Decomposition level. Must be >= 0.
    axes : tuple of int
        Axes along which to apply the transform.
    approx_only : bool, optional
        If True, returns only approximation coefficients.

    Returns
    -------
    coeffs : list[ndarray]
        Approximation and detail coefficients.

    If approx_only is True, returns only approximation coefficients.
    """
    if approx_only:
        return reduce(
            lambda approx, axis: modwt_1d(approx, level, axis, approx_only=True),
            axes,
            data,
        )

    coeffs = [data]
    for axis in axes:
        new_coeffs = []
        for x in coeffs:
            new_coeffs.extend(modwt_1d(x, level, axis))
        coeffs = new_coeffs
    return coeffs


def imodwt_nd(coeffs, level, axes):
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