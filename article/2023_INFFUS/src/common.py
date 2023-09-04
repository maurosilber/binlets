import itertools
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np
import pandas as pd
from numba import guvectorize

figure_path = Path("article/figures")


def weighted_rmse(y, y_true, y_cov):
    try:
        return np.mean(mahalanobis(y, y_true, y_cov)) ** 0.5
    except ValueError:
        return np.mean((y - y_true) ** 2 / y_cov) ** 0.5


def reduced_bias(y, y_true, y_cov):
    if np.ndim(y_cov) > np.ndim(y_true):
        y_cov = np.diag(y_cov)
    return np.mean((y - y_true) / y_cov)


def normalize(x):
    x_max = np.max(x)
    x_min = np.min(x)
    return (x - x_min) / (x_max - x_min)


def identity(x):
    return x


def _sweep(
    *,
    sigmas: Iterable[float],
    x_true,
    x_var_true,
    measure_func: Callable,
    value_func: Callable,
    binlets_funcs: dict[str, Callable],
) -> Iterator[dict]:
    x_measured = measure_func(x_true)
    for s in sigmas:
        for name, func in binlets_funcs.items():
            denoised = func(x_measured, sigma=s)
            yield dict(
                sigma=s,
                variant=name,
                RMSE=weighted_rmse(
                    value_func(denoised),
                    x_true,
                    x_var_true,
                ),
                bias=reduced_bias(
                    value_func(denoised),
                    x_true,
                    x_var_true,
                ),
            )


def sweep(
    *,
    sigmas: Iterable[float],
    x_true,
    x_var_true,
    measure_func: Callable,
    value_func: Callable = identity,
    binlets_funcs: dict[str, Callable],
    repeats: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        itertools.chain.from_iterable(
            _sweep(
                sigmas=sigmas,
                x_true=x_true,
                x_var_true=x_var_true,
                measure_func=measure_func,
                value_func=value_func,
                binlets_funcs=binlets_funcs,
            )
            for _ in range(repeats)
        )
    )


@guvectorize(
    "void(float64[:], float64[:], float64[:,:], float64[:])",
    "(n),(n),(n,n)->()",
    nopython=True,
)
def mahalanobis(x, y, cov, d):
    diff = x - y
    if np.all(diff == 0):
        d[:] = 0
    elif np.linalg.det(cov) == 0:
        d[:] = np.inf
    else:
        vi = np.linalg.inv(cov)
        d[:] = diff @ (vi @ diff)
