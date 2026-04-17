import h5py
import pandas as pd
import numpy as np
import equinox as eqx
import jax
import jax.random as jrn
import jax.numpy as jnp
from tqdm import tqdm

from lstm import camels, config

def evaluate(model, forcing, tstart=None, tend=None, bids=None, datadir="data", perturbation=None, xmean=None, xstd=None):
    """
    Evaluates the model on specified basins.
    It handles data normalization using mean and standard deviation stored in the HDF5 file.

    Parameters:
    -----------
    model : equinox.Module
        The trained model to evaluate.
    forcing : str
        Forcing dataset name.
    tstart : str or datetime
        Start date for evaluation.
    tend : str or datetime
        End date for evaluation.
    bids : list
       List of basin IDs to evaluate on.
    datadir : str, optional
        Directory path containing data files (default: "data")
    perturbation : Perturbation, optional
        Perturbation to apply to raw inputs before normalisation (default: None).
    xmean : array-like, optional
        Mean of training data for normalization. If None, reads from HDF5 file.
    xstd : array-like, optional
        Standard deviation of training data for normalization. If None, reads from HDF5 file.

    Returns:
    --------
    tuple
        Tuple of (mod, obs) where:
        - mod: pandas.DataFrame containing model predictions for each basin
        - obs: pandas.DataFrame containing observed values for each basin
    """
    h5path = f"{datadir}/{forcing}.h5"
    with h5py.File(h5path) as f:
        if xmean is None:
            xmean = f.attrs["xmean"]
        if xstd is None:
            xstd = f.attrs["xstd"]
        seq_len = f.attrs["seq_len"]
        if bids is None:
            bids = [b.decode() for b in f["bids"][:]]
        if tstart is None or tend is None:
            tstart = f.attrs["tstart"]
            tend = f.attrs["tend"]
    attribs = camels.read_static_attributes(config.STATIC_VARS, datadir)
    xt, xst, yt, bids = camels.get_data(bids, forcing, tstart, tend, seq_len, attribs, datadir)
    dt = pd.date_range(tstart, tend)
    mod = {}
    obs = {}
    for bi, bid in enumerate(tqdm(bids, desc="Evaluating")):
        xt_ = np.stack([xt[bi][t:t+seq_len, :] for t in range(xt[bi].shape[0]-seq_len+1)])
        xst_ = np.stack([xst[bi] for _ in range(xt[bi].shape[0]-seq_len+1)])
        xt_ = jnp.asarray(xt_)
        if perturbation is not None:
            xt_, _ = perturbation(xt_)
        xt_ = (xt_ - xmean) / xstd
        inf_model = eqx.nn.inference_mode(model)
        pred = jax.vmap(inf_model)(xt_, xst_, key=jrn.split(jrn.PRNGKey(0), xt_.shape[0]))
        mod[bid] = pd.Series(pred[:, 0], dt)
        obs[bid] = pd.Series(yt[bi][:, 0], dt)
    return pd.DataFrame(mod), pd.DataFrame(obs)

def nse(obs, sim):
    """Calculate Nash-Sutcliffe Efficiency

    Parameters:
    -----------
    obs : array-like
        Observed values
    sim : array-like
        Simulated values

    Returns:
    --------
    float
        Nash-Sutcliffe Efficiency score
    """
    return 1 - jnp.sum((obs - sim)**2) / jnp.sum((obs - jnp.mean(obs))**2)

def kge(obs, sim):
    """Calculate Kling-Gupta Efficiency

    Parameters:
    -----------
    obs : array-like
        Observed values
    sim : array-like
        Simulated values

    Returns:
    --------
    float
        Kling-Gupta Efficiency score
    """
    r = jnp.corrcoef(obs, sim)[0, 1]
    alpha = jnp.std(sim) / jnp.std(obs)
    beta = jnp.mean(sim) / jnp.mean(obs)
    return 1 - jnp.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
