import numpy as np
from scipy import stats

from .modwt import modwt_level_nd, imodwt_level_nd


def binlet(inputs: tuple,
           valfun, covfun, bin_args, args,
           p_value: float, levels: int, axes):
    """Generic binlets denoising.

    Parameters
    ----------
    inputs : tuple of arrays
        Data to denoise.
    valfun, covfun : callables f(*inputs, *bin_args, *args)
        Calculate value and covariance matrix.
    bin_args : tuple of arrays (optional)
        Other inputs of which its approximation coefficients are needed to calculate valfun and covfun.
    args : tuple (optional)
        Extra arguments used by valfun and covfun
    p_value : float
    levels : int
    axes : tuple
        Axes over which transform is applied.

    Returns
    -------
        Tuple of denoised inputs.
    """
    # TODO: convert axes to positive numbers

    ndim = len(axes)
    threshold = stats.chi2.isf(p_value, df=ndim)

    a_key = 'a' * ndim  # Approximation coefficient key
    coeffs_list = []

    approxs = inputs
    tree_mask = True
    for level in range(levels):
        (approxs, details, tree_mask), bin_args = binlet_level(approxs, threshold, tree_mask, valfun, covfun, bin_args,
                                                               args, level, axes)
        coeffs_list.append(details)

    for level, coeffs in reversed(list(enumerate(coeffs_list))):
        for approx, coeff in zip(approxs, coeffs):
            coeff[a_key] = approx
        approxs = tuple(imodwt_level_nd(coeff, level, axes) for coeff in coeffs)

    return approxs


def binlet_level(inputs, threshold, tree_mask, valfun, covfun, bin_args, args, level, axes):
    """Computes one level of the binlets transform."""
    a_key = 'a' * len(axes)  # Approximation coefficient key

    # Calculate threshold masks with previous level data
    threshold_masks = chi2_threshold_level(inputs, threshold, valfun, covfun, bin_args, args, level, axes)
    threshold_masks.pop(a_key)  # Don't threshold approximation coefficients.

    # Calculate current level
    inputs_coeffs = tuple(modwt_level_nd(x, level, axes) for x in inputs)
    bin_args = tuple(modwt_level_nd(x, level, axes, approx_only=True) for x in bin_args)

    # Threshold current level
    for key, mask in threshold_masks.items():
        tree_mask &= mask
    for key in threshold_masks.keys():
        for coeffs in inputs_coeffs:
            coeffs[key][tree_mask] = 0

    inputs_approx = tuple(coeffs.pop(a_key) for coeffs in inputs_coeffs)
    inputs_details = inputs_coeffs

    return (inputs_approx, inputs_details, tree_mask), bin_args


def chi2_threshold_level(inputs, threshold, valfun, covfun, bin_args, args, level, axes):
    """Calculates a mask for thresholding each detail coefficient."""
    val = valfun(*inputs, *bin_args, *args)
    cov = covfun(*inputs, *bin_args, *args)

    diff = modwt_level_nd(val, level, axes)
    cov = modwt_level_nd(cov, level, axes, approx_only=True)
    inv_cov = np.linalg.inv(cov)
    return {key: quadratic_form(inv_cov, value) < threshold for key, value in diff.items()}


def quadratic_form(matrix, vector):
    """Calculates a quadratic form between a matrix and a vector across the last two dimensions."""
    return (np.swapaxes(vector, -1, -2) @ matrix @ vector)[..., 0, 0]
