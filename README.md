# Binlets: denoising via adaptive binning

![Package](https://img.shields.io/pypi/v/binlets?label=binlets)
![PyVersion](https://img.shields.io/pypi/pyversions/binlets?label=python)
![License](https://img.shields.io/pypi/l/binlets?label=license)
![CodeStyle](https://img.shields.io/badge/code%20style-black-000000.svg)
[![CI](https://github.com/maurosilber/binlets/actions/workflows/test.yml/badge.svg)](https://github.com/maurosilber/binlets/actions/workflows/ci.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/maurosilber/binlets/master.svg)](https://results.pre-commit.ci/latest/github/maurosilber/binlets/master)
[![Paper](https://img.shields.io/badge/DOI-10.1016/j.inffus.2023.101999-darkgreen)](https://doi.org/10.1016/j.inffus.2023.101999)

## Usage

`binlets` requires an array of data,
and a test function to compare two data points.

### Single-channel with Poisson statistics

For instance,
for single-channel signal with Poisson statistics:

```python
def chi2_test(x: NDArray, y: NDArray) -> NDArray[bool]:
    """Compares two values with Poisson noise using a χ² test."""
    diff = x - y
    var_diff = x + y  # Poisson variance
    return diff**2 <= var_diff


denoised = binlets(
    data,
    test=chi2_test,
    levels=None,  # max averaging area is 2**levels. By default, floor(log2(min(data.shape)))
    linear=True,  # the transformation is linear (x - y)
)
```

We recomend wrapping this in a function,
and providing an extra parameter to adjust the significance level:

```python
def poisson_binlets(data: NDArray, *, n_sigma: float, levels: int | None = None):
    def chi2_test(x: NDArray, y: NDArray) -> NDArray[bool]:
        """Compares two values with Poisson noise using a χ² test."""
        diff = x - y
        var_diff = x + y  # Poisson variance
        return diff**2 <= n_sigma**2 * var_diff

    denoised = binlets(
        data,
        test=chi2_test,
        linear=True,
    )
    return denoised
```

### Ratio of multichannel-data

For multichannel data,
binlets expects channels to be in the first dimension of the data array.
That is, `data.shape` should be `(N_CHANNELS, *spatial_dimensions)`.

```python
def ratio(channels):
    """Statistic of interest."""
    return channels[1] / channels[0]


def test(x, y):
    # The test of your choice. For instance, a χ² test.
    diff = ratio(x) - ratio(y)
    var_diff = ratio_var(x) + ratio_var(y)
    return diff**2 <= var_diff


denoised = binlets(data, test=test, ...)  # the same as before
denoised_ratio = ratio(denoised)
```

## Installation

Binlets can be installed from PyPI:

```
pip install binlets
```

or conda-forge:

```
conda installl -c conda-forge binlets
```

## Development

To set up a development environment in a new conda environment, run the following commands:

```
git clone https://github.com/maurosilber/binlets
cd binlets
conda env create -f environment.yml
conda activate binlets
pre-commit install
```
