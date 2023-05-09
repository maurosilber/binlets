from __future__ import annotations

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol

import numpy as np


class Wavelet(Protocol):
    """Wavelet coefficients.

    Coefficient names based on PyWavelets.

    dec_lo: Decomposition low-pass filter values.
    dec_hi: Decomposition high-pass filter values.
    rec_lo: Reconstruction low-pass filter values.
    rec_hi: Reconstruction high-pass filter values.
    """

    dec_lo: tuple[float, float]
    dec_hi: tuple[float, float]
    rec_lo: tuple[float, float]
    rec_hi: tuple[float, float]


class Haar:
    """Unnormalized Haar wavelet.

    Decomposition can be thought of as a binning operation.
    """

    dec_lo = [1.0, 1.0]
    dec_hi = [-1.0, 1.0]
    rec_lo = [0.5, 0.5]
    rec_hi = [0.5, -0.5]


class NormalizedHaar:
    """Normalized Haar wavelet."""

    s = 1 / 2**0.5
    dec_lo = [s, s]
    dec_hi = [-s, s]
    rec_lo = [s, s]
    rec_hi = [s, -s]


def modwt_1d(
    data: np.ndarray,
    level: int,
    axis: int = -1,
    *,
    wavelet: Wavelet,
) -> tuple[np.ndarray, np.ndarray]:
    """1D MODWT transform.

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
    (approx, detail) : tuple[ndarray, ndarray]
        Approximation and detail coefficients.
    """
    shifted = np.roll(data, -(2**level), axis=axis)
    a0, a1 = wavelet.dec_lo
    approx = a0 * data + a1 * shifted
    d0, d1 = wavelet.dec_hi
    detail = d0 * data + d1 * shifted
    return approx, detail


def undo_1d_modwt(approx, detail, *, wavelet: Wavelet) -> tuple[np.ndarray, np.ndarray]:
    """Inverse 1D MODWT transform.

    Parameters
    ----------
    approx, detail : ndarray
        Approximation and detail coefficients.

    Returns
    -------
    tuple[ndarray, ndarray]
        Reconstructed left and right coefficients.
    """
    a0, a1 = wavelet.rec_lo
    d0, d1 = wavelet.rec_hi
    x = d0 * approx + d1 * detail
    y = a0 * approx + a1 * detail
    return x, y


def imodwt_1d(
    approx: np.ndarray,
    detail: np.ndarray,
    level: int,
    axis: int = -1,
    *,
    wavelet: Wavelet,
) -> np.ndarray:
    """Inverse 1D MODWT transform.

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
    data, shifted = undo_1d_modwt(approx, detail, wavelet=wavelet)
    unshifted = np.roll(shifted, 2**level, axis=axis)
    return (data + unshifted) / 2


def modwt_nd(
    data: np.ndarray,
    level: int,
    axes: tuple[int, ...],
    *,
    wavelet: Wavelet,
) -> list[np.ndarray]:
    """nD MODWT transform.

    Parameters
    ----------
    data : ndarray
        Input signal.
    level : int
        Decomposition level. Must be >= 0.
    axes : tuple[int, ...]
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
            new_coeffs.extend(modwt_1d(x, level, axis, wavelet=wavelet))
        coeffs = new_coeffs
    return coeffs


def imodwt_nd(
    coeffs: list[np.ndarray],
    level: int,
    axes: tuple[int, ...],
    *,
    wavelet: Wavelet,
) -> np.ndarray:
    """nD MODWT transform.

    Parameters
    ----------
    coeffs : list[ndarray]
        Input coefficients in the order given by modwt_nd.
    level : int
        Decomposition level. Must be >= 0.
    axes : tuple[int, ...]
        Axes along which to apply the transform.

    Returns
    -------
    coeffs : list[ndarray]
        Approximation and detail coefficients.
    """
    for axis in reversed(axes):
        pairwise = zip(coeffs[0::2], coeffs[1::2])
        coeffs = [imodwt_1d(A, D, level, axis, wavelet=wavelet) for A, D in pairwise]
    return coeffs[0]


def _modwt_mask_inplace(
    mask: np.ndarray[bool],
    level: int,
    axes: tuple[int, ...],
    *,
    wavelet: Wavelet,
):
    """Updates mask for the given level of the transform.

    Parameters
    ----------
    mask : ndarray[bool]
        Mask for coefficients.
    level : int
        Decomposition level. Must be >= 0.
    axes : tuple[int, ...]
        Axes along which to apply the transform.

    Returns
    -------
    coeffs : list[ndarray]
        Approximation and detail coefficients.
    """
    shift = -(2**level)
    for axis in axes:
        mask &= np.roll(mask, shift, axis=axis)
    return mask
