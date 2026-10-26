import jax.numpy as jnp
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

class Perturbation(ABC):

    @abstractmethod
    def __call__(self, x, key=None):
        pass

    @property
    @abstractmethod
    def name(self):
        pass

    def get_config(self):
        return {'type': self.name}

    def compute_stats(self, xmean, xstd):
        return xmean, xstd

    def apply_series(self, x, bid, dates):
        """Perturb one basin's full input series `x` (time, n_features) on `dates`."""
        y, _ = self(jnp.asarray(x)[None])
        return np.asarray(y[0])

class NoPerturbation(Perturbation):

    def __call__(self, x, key=None):
        return x, key

    @property
    def name(self):
        return "none"

class ZeroPrecipitation(Perturbation):

    def __init__(self, dims=(0,)):
        self.dims = dims

    def __call__(self, x, key=None):
        for dim in self.dims:
            x = x.at[:, :, dim].set(0.0)
        return x, key

    @property
    def name(self):
        return "zero"

    def get_config(self):
        return {
            "type": self.name,
            "dims": self.dims
        }

    def compute_stats(self, xmean, xstd):
        xmean = xmean.copy()
        xstd = xstd.copy()
        return xmean, xstd

class RandomPerturbation(Perturbation):
    """Multiplicative random error, P' = P * max(0, 1 + stddev * eps), eps ~ N(0, 1).

    The noise is a fixed field indexed by (seed, basin, date): each basin gets
    one noisy precipitation series over the whole record, and training windows,
    evaluation windows and VIC forcing files all read from that same series.
    The standard normal draws do not depend on `stddev`, so the error levels
    share one underlying realization and differ only in magnitude.
    """

    ORIGIN = pd.Timestamp("1980-01-01")
    NDAYS = 14976  # through 2020-12-31, covers the CAMELS record

    def __init__(self, stddev=0.1, dims=(0,), seed=0):
        self.dims = dims
        self.stddev = stddev
        self.seed = seed

    def __call__(self, x, key=None):
        raise NotImplementedError(
            "RandomPerturbation is indexed by basin and date; use apply_series "
            "or multipliers instead of perturbing windows independently."
        )

    def multipliers(self, bid, dates):
        """Noise multipliers for basin `bid` on `dates`, shape (len(dates), len(dims))."""
        offset = (pd.DatetimeIndex(dates) - self.ORIGIN).days.values
        if offset.min() < 0 or offset.max() >= self.NDAYS:
            raise ValueError(f"Dates outside the noise field: {dates[0]} to {dates[-1]}")
        rng = np.random.default_rng([self.seed, int(bid)])
        eps = rng.standard_normal((self.NDAYS, len(self.dims)))
        return np.maximum(0.0, 1.0 + self.stddev * eps[offset]).astype(np.float32)

    def apply_series(self, x, bid, dates):
        x = np.array(x, dtype=np.float32)
        m = self.multipliers(bid, dates)
        for i, dim in enumerate(self.dims):
            x[:, dim] *= m[:, i]
        return x

    @property
    def name(self):
        return "random"

    def get_config(self):
        return {
            "type": self.name,
            "dims": self.dims,
            "stddev": self.stddev,
            "seed": self.seed,
            "noise": "clipped multiplicative Gaussian, fixed per (seed, basin, date)",
        }

    def compute_stats(self, xmean, xstd):
        xmean = xmean.copy()
        xstd = xstd.copy()
        return xmean, xstd

class BiasPerturbation(Perturbation):

    def __init__(self, mbias=1.1, dims=(0,)):
        self.dims = dims
        self.bias = mbias

    def __call__(self, x, key=None):
        for dim in self.dims:
            x = x.at[:, :, dim].multiply(self.bias)
        return x, key

    @property
    def name(self) -> str:
        return "bias"

    def get_config(self) -> dict:
        return {
            "type": self.name,
            "dims": self.dims,
            "bias": self.bias,
        }

    def compute_stats(self, xmean, xstd):
        xmean = xmean.copy()
        xstd = xstd.copy()
        return xmean, xstd
