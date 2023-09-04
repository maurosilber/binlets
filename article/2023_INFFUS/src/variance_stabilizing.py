import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import seaborn
import seaborn.objects as so
from binlets import binlets
from binlets._modwt import NormalizedHaar
from matplotlib.figure import Figure
from scipy import stats

try:
    from .common import figure_path, normalize, sweep
except Exception:
    from common import normalize, sweep


def anscombe_transform(x):
    """The Anscombe transform

    Anscombe, F. J. (1948)
    "The transformation of Poisson, binomial and negative-binomial data"
    Biometrika
    doi:10.1093/biomet/35.3-4.246
    """
    return 2 * np.sqrt(x + 3 / 8)


def anscombe_tranform_inverse(x):
    """Unbiased inverse of the Anscombe transform.

    Mäkitalo, M.; Foi, A. (2011)
    "A closed-form approximation of the exact unbiased inverse of
    the Anscombe variance-stabilizing transformation"
    IEEE Transactions on Image Processing
    doi:10.1109/TIP.2011.2121085
    """
    return (
        x**2 / 4
        - 1 / 8
        + np.sqrt(3 / 8) / x
        - (11 / 8) / x**2
        + (5 / 8) * np.sqrt(3 / 2) / x**3
    )


def binlets_anscombe(
    data,
    *,
    sigma: float,
    levels: int = None,
    linear: bool = True,
):
    def test(x, y):
        return np.abs(x - y) < 2**0.5 * sigma

    y = anscombe_transform(data)
    y = binlets(
        y[np.newaxis],
        test=test,
        wavelet=NormalizedHaar,
        levels=levels,
        linear=linear,
    )[0]
    y = anscombe_tranform_inverse(y)
    return y


def binlets_zscore(
    data,
    *,
    sigma: float,
    levels: int = None,
    linear: bool = True,
):
    def test(x, y):
        diff = x - y
        var = x + y
        return diff**2 < sigma**2 * var

    return binlets(data[np.newaxis], test=test, levels=levels, linear=linear)[0]


def binlets_ctest(data, *, sigma: float, levels: int = None):
    threshold = stats.chi.sf(sigma, df=1)

    def zscore_test(x, y):
        diff = x - y
        var = x + y
        return diff**2 < sigma**2 * var

    def ctest_test(x, y):
        k = x + y
        dist = stats.binom(n=k, p=0.5)
        pvalue = 2 * np.minimum(dist.cdf(x), dist.sf(x) + dist.pmf(x))
        return pvalue > threshold

    def test(x, y):
        """Combined test. Uses Z-score approximation for >= 20."""
        out = np.empty_like(x, bool)
        mask = (x < 20) | (y < 20)
        out[mask] = ctest_test(x[mask], y[mask])
        out[~mask] = zscore_test(x[~mask], y[~mask])
        return out

    return binlets(data[np.newaxis], test=test, levels=levels)[0]


def generate_true(*, N, size):
    x_true = pywt.data.demo_signal("Blocks", n=size)
    x_true = N * (0.5 + normalize(x_true))
    return x_true


def panel_A(fig: Figure, N, sigma, size):
    x_true = generate_true(N=N, size=size)
    x_noisy = np.random.default_rng(42).poisson(x_true)

    data = {
        "Measured": x_noisy,
        "Anscombe": binlets_anscombe(x_noisy, sigma=sigma),
        "Z-score": binlets_zscore(x_noisy, sigma=sigma),
    }

    fig.suptitle(f"Mean: {N} | $\sigma$: {sigma}")
    axes = fig.subplots(3, sharex=True, sharey=True, gridspec_kw=dict(hspace=0.15))

    for ax, (label, signal) in zip(axes, data.items()):
        ax.plot(signal)
        ax.text(0.03, 1, label, va="top", transform=ax.transAxes)
        seaborn.despine(ax=ax, top=True, right=True)
    axes[0].plot(x_true)
    axes[0].set(xticks=(0, len(x_true)))


def chi2(*, N: list[int], size: int, sigmas: list[float]):
    df = []
    for Ni in N:
        x_true = generate_true(N=Ni, size=size)
        df.append(
            sweep(
                sigmas=sigmas,
                x_true=x_true,
                x_var_true=x_true,
                measure_func=np.random.default_rng(42).poisson,
                binlets_funcs={
                    "Anscombe": binlets_anscombe,
                    "Z-score": binlets_zscore,
                },
                repeats=100,
            ).assign(N=Ni, size=size)
        )
    return pd.concat(df)


def panel_B(fig: Figure, df: pd.DataFrame):
    (
        so.Plot(df, x="sigma", y="RMSE", color="variant")
        .label(x="Threshold $\sigma$", y="RMSE", col="Mean:")
        .facet(col="N")
        .add(so.Line(), so.Est())
        .add(so.Band(alpha=1), so.Est(errorbar="se"))
        .add(so.Band(), so.Est(errorbar="sd"))
        .scale(y="log")
        .theme(plt.rcParams)
        .on(fig)
        .plot()
    )
    seaborn.despine(fig, top=True, right=True)


if __name__ == "__main__":
    fig = Figure(figsize=(6, 3))
    figs = fig.subfigures(1, 3, width_ratios=[1, 0.2, 3])
    panel_A(figs[0], N=10, sigma=3, size=2**12)
    panel_B(
        figs[2],
        df=chi2(
            N=[1, 5, 10],
            size=2**12,
            sigmas=np.linspace(0, 6, 20),
        ),
    )
    fig.savefig(figure_path / "variance_stabilizing", bbox_inches="tight")
