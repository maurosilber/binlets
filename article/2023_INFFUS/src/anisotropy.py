import numpy as np
from binlets import binlets
from common import mahalanobis
from scipy import stats


def total_intensity(parallel, perpendicular):
    return parallel + 2 * perpendicular


def anisotropy(images):
    par, per = images
    total = total_intensity(par, per)
    diff = (par - per).astype(float)
    return np.divide(diff, total, where=total > 0, out=diff)


def anisotropy_variance(images):
    par, per = images
    total = total_intensity(par, per)
    numerator = 9.0 * (par + per) * par * per
    return np.divide(numerator, total**4, where=total > 0, out=numerator)


def intensity_from_anisotropy(anisotropy, total_intensity):
    par = total_intensity / 3 * (1 + 2 * anisotropy)
    per = total_intensity / 3 * (1 - anisotropy)
    return np.stack((par, per))


def independent_binlets(
    inputs,
    *,
    sigma: float,
    levels: int = None,
    linear: bool = False,
):
    """Denoises each channel separately.

    Assumes poisson statistics for each pixel.
    """

    def test(x, y):
        diff = x - y
        var = x + y  # poisson variance
        return diff**2 < sigma**2 * var

    out = np.empty_like(inputs)
    out[:1] = binlets(inputs[:1], test=test, levels=levels, linear=linear)
    out[1:] = binlets(inputs[1:], test=test, levels=levels, linear=linear)
    return out


def joint_poisson_binlets(
    inputs, *, sigma: float, levels: int = None, linear: bool = False
):
    """Denoises each channel separately.

    Assumes poisson statistics for each pixel.
    """
    n = len(inputs)
    p_value = stats.chi.cdf(sigma, df=1)
    threshold = stats.chi2.ppf(p_value, df=n)

    def mean(x):
        return np.moveaxis(x, 0, -1)

    def test(x, y):
        cov = np.zeros((*x.shape[1:], 2, 2))
        for i in range(2):
            cov[..., i, i] = x[i] + y[i]
        return mahalanobis(mean(x), mean(y), cov) < threshold

    return binlets(
        inputs,
        test=test,
        levels=levels,
        linear=linear,
    )


def anisotropy_binlets(
    inputs,
    *,
    sigma,
    levels=None,
    linear: bool = False,
):
    """Denoises directly on the anisotropy image."""

    def test(x, y):
        diff = x[0] - y[0]
        var = x[1] + y[1]
        return diff**2 < sigma**2 * var

    data = np.empty_like(inputs, float)
    data[0] = anisotropy(inputs)
    data[1] = anisotropy_variance(inputs)
    denoised = binlets(data, test=test, levels=levels, linear=linear)
    return intensity_from_anisotropy(
        denoised[0],
        total_intensity=1,  # It doesn't matter.
    )


def joint_binlets(
    inputs,
    *,
    sigma: float,
    levels: int = None,
    linear: bool = False,
):
    """Denoises both channels jointly by comparing anisotropies."""

    def test(x, y):
        diff = anisotropy(x) - anisotropy(y)
        var = anisotropy_variance(x) + anisotropy_variance(y)
        return diff**2 < sigma**2 * var

    return binlets(inputs, test=test, levels=levels, linear=linear)


if __name__ == "__main__":
    a = np.array([0.1, 0.9, 0.1, 0.9])
    N = 100 * np.array([1, 1, 10, 10])

    images_true = intensity_from_anisotropy(a, N)
    images_noisy = np.random.default_rng(42).poisson(images_true)
    denoised = joint_binlets(images_noisy, sigma=2)

    (
        np.array([images_noisy[0], denoised[0]]),
        anisotropy(denoised),
    )
