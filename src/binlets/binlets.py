import numpy as np
from numba import njit
from scipy import stats

from .modwt import imodwt_level_nd, modwt_level_mask_nd, modwt_level_nd


def binlet(valfun, covfun, is_scalar):
    """Generic binlet constructor.

    Parameters
    ----------
    valfun, covfun : callable
        Value and covariance functions.
        They should expect to be called as func(*inputs, *bin_args, *args, **kwargs).
    is_scalar : bool
        Set to True if valfun and covfun return a scalar (ie, covfun is the variance).
        Set to False if valfun returns a vector and covfun a matrix.
    """

    def binlets(
        inputs,
        levels,
        p_value=0.05,
        axes=None,
        mask=None,
        bin_args=None,
        args=None,
        kwargs=None,
    ):
        """Binlets denoising.

        Parameters
        ----------
        inputs : tuple of ndarrays
            Data to denoise.
        levels : int
            Decomposition level. Must be >= 0. If == 0, does nothing.
            Sets maximum possible binning of size 2**level.
        p_value : float, optional
            Controls the level of denoising. Default is 0.05.

        Returns
        -------
        tuple of ndarrays
            Tuple of denoised inputs.

        Other Parameters
        ----------------
        axes : tuple, optional
            Axes over which transform is applied. Default is all axes.
        mask : ndarray of bools, optional
            Marks data to denoise. Data where mask==False is not denoised.
            By default, all True.
        bin_args : tuple of ndarrays, optional
            Extra arguments used by valfun and covfun.
            They are binned the same way as inputs, but are not denoised.
        args : tuple, optional
            Extra arguments used by valfun and covfun
        kwargs : dict, optional
            Extra arguments used by valfun and covfun
        """
        if levels < 0:
            raise ValueError("Levels must be >= 0.")
        elif levels == 0:
            return inputs

        if bin_args is None:
            bin_args = tuple()
        if args is None:
            args = tuple()
        if kwargs is None:
            kwargs = {}

        shape = np.broadcast(*inputs).shape
        if mask is None:
            mask = np.ones(shape, bool)
        if axes is None:
            axes = np.nonzero(np.array(shape) > 1)[0]
        else:
            # convert axes to positive numbers
            axes = np.asarray(axes)
            axes = tuple(np.where(axes < 0, len(shape) + axes, axes))

        threshold = stats.chi2.isf(p_value, df=len(axes))
        approxs, details_list = inputs, []

        # Decomposition
        for level in range(levels):
            approxs, details, mask, bin_args = _binlet_level(
                approxs,
                level,
                threshold,
                axes,
                mask,
                valfun,
                covfun,
                is_scalar,
                bin_args,
                args,
                kwargs,
            )
            details_list.append(details)

        # Reconstruction
        for level, details in reversed(list(enumerate(details_list))):
            approxs = tuple(
                imodwt_level_nd(a, d, level, axes) for a, d in zip(approxs, details)
            )

        return approxs

    return binlets


def _binlet_level(
    inputs,
    level,
    threshold,
    axes,
    mask,
    valfun,
    covfun,
    is_scalar,
    bin_args,
    args,
    kwargs,
):
    """Compute one level of the binlets transform."""

    # Calculate threshold masks with previous level data
    mask = modwt_level_mask_nd(mask, level, axes)
    threshold_masks = _chi2_threshold_level(
        inputs,
        level,
        threshold,
        axes,
        valfun,
        covfun,
        is_scalar,
        bin_args,
        args,
        kwargs,
    )
    for m in threshold_masks.values():
        mask &= m

    # Calculate current level
    approxs, details = tuple(zip(*(modwt_level_nd(x, level, axes) for x in inputs)))
    bin_args = tuple(modwt_level_nd(x, level, axes, approx_only=True) for x in bin_args)

    # Threshold current level
    for key in threshold_masks.keys():
        for coeffs in details:
            coeffs[key][mask] = 0

    return approxs, details, mask, bin_args


def _chi2_threshold_level(
    inputs, level, threshold, axes, valfun, covfun, is_scalar, bin_args, args, kwargs
):
    """Calculates a mask for thresholding each detail coefficient."""

    val = valfun(*inputs, *bin_args, *args, **kwargs)
    cov = covfun(*inputs, *bin_args, *args, **kwargs)

    _, diff = modwt_level_nd(val, level, axes)
    cov = modwt_level_nd(cov, level, axes, approx_only=True)
    if is_scalar:
        return {
            key: _scalar_quadratic_form(cov, value) < threshold
            for key, value in diff.items()
        }
    else:
        inv_cov = np.linalg.inv(cov)
        return {
            key: _vector_quadratic_form(inv_cov, value) < threshold
            for key, value in diff.items()
        }


def _scalar_quadratic_form(variance, mean):
    """Computes a quadratic form.

    Equivalent to squared Z-score.

    Parameters
    ----------
    variance, mean : (...) ndarray.
    """
    return mean**2 / variance


@njit
def _vector_quadratic_form(matrix, vector):
    """Computes a quadratic form along the last dimension.

    Equivalent to v.T @ M @ v.

    Parameters
    ----------
    matrix : (..., N, N) ndarray
    vector : (..., N) ndarray

    Returns
    -------
    quadratic_form : (...) ndarray
    """
    # TODO: take advantage of symmetric matrices.
    out = np.zeros(vector.shape[:-1])
    for i in range(vector.shape[-1]):
        for j in range(vector.shape[-1]):
            out += matrix[..., i, j] * vector[..., i] * vector[..., j]
    return out
