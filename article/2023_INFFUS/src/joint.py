import numpy as np
from binlets import binlets
from binlets._modwt import NormalizedHaar
from common import mahalanobis
from scipy import stats


def independent_binlets(
    inputs, *, sigma: float, levels: int = None, linear: bool = True
):
    """Denoises each channel separately.

    Assumes poisson statistics for each pixel.
    """
    out = np.empty_like(inputs)
    for i, xi in enumerate(inputs):
        out[i] = joint_binlets(
            xi[np.newaxis],
            sigma=sigma,
            levels=levels,
            linear=linear,
        )[0]
    return out


def joint_binlets(inputs, *, sigma: float, levels: int = None, linear: bool = True):
    """Denoises each channel separately.

    Assumes poisson statistics for each pixel.
    """
    n = len(inputs)
    p_value = stats.chi.cdf(sigma, df=1)
    threshold = stats.chi2.ppf(p_value, df=n)

    def mean(x):
        return np.moveaxis(x, 0, -1)

    def test(x, y):
        cov = np.eye(len(x))
        return mahalanobis(mean(x), mean(y), cov) < threshold

    return binlets(
        inputs,
        test=test,
        wavelet=NormalizedHaar,
        levels=levels,
        linear=linear,
    )


def generate_vectors(snr_big, snr_small, *, n=4):
    delta = np.arctan2(snr_big, snr_small)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angles += delta
    points = snr_big * np.exp(1j * angles)
    return points.view(float).reshape(n, 2)


def build_signal(vectors: list[tuple[float]], size: int, segment_size: float):
    rng = np.random.default_rng(42)
    signal = []
    while len(signal) < size:
        v = rng.choice(vectors)
        for _ in range(int(rng.exponential(segment_size))):
            signal.append(v)
    return np.array(signal[:size]).T


if __name__ == "__main__":
    from common import chi2_sweep

    vectors = generate_vectors(5, 0.5, n=4)
    signal_true = build_signal(vectors, size=256, segment_size=6)
    signal_noisy = np.random.normal(signal_true)

    sigma = 5
    signal_independent = independent_binlets(signal_noisy, sigma=sigma)
    signal_ioint = joint_binlets(signal_noisy, sigma=sigma)

    chi2_sweep(
        sigmas=np.linspace(0, 8, 15),
        x_true=signal_true,
        x_var_true=1,
        measure_func=np.random.default_rng(42).normal,
        binlets_funcs={
            "independent": independent_binlets,
            "joint": joint_binlets,
        },
        repeats=5,
    )
