import jax.random as jrn
import jax.numpy as jnp
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
        for dim in self.dims:
            xmean[dim] = 0.0
            xstd[dim] = 1.0
        return xmean, xstd

class RandomPerturbation(Perturbation):

    def __init__(self, stddev=0.1, dims=(0,)):
        self.dims = dims
        self.stddev = stddev

    def __call__(self, x, key=None):
        if key is None:
            key = jrn.PRNGKey(0)
        for dim in self.dims:
            noise_key, key = jrn.split(key)
            # x.shape is (batch, seq_len, n_features); we perturb the feature dim
            noise_shape = (x.shape[0], x.shape[1])
            noise = jrn.normal(noise_key, noise_shape) * self.stddev + 1.0
            x = x.at[:, :, dim].multiply(noise)
        return x, key

    @property
    def name(self):
        return "random"

    def get_config(self):
        return {
            "type": self.name,
            "dims": self.dims,
            "stddev": self.stddev,
        }

    def compute_stats(self, xmean, xstd):
        xmean = xmean.copy()
        xstd = xstd.copy()
        for dim in self.dims:
            xstd[dim] = xstd[dim] * jnp.sqrt(1 + self.stddev**2)
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
        for dim in self.dims:
            xmean[dim] = xmean[dim] * self.bias
            xstd[dim] = xstd[dim] * self.bias
        return xmean, xstd
