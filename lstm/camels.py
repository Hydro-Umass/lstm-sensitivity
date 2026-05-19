import pandas as pd
import numpy as np
import h5py
from typing import Optional

import jax.random as jrn
import jax.numpy as jnp


def read_met(metfile):
    """
    Reads CAMELS forcing dataset.

    Parameters:
    -----------
    metfile : str
        Path to the meteorological forcing file

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing meteorological data with Date as index
    """
    header = [
        "Year",
        "Month",
        "Day",
        "Hr",
        "Dayl",
        "Prcp",
        "Srad",
        "Swe",
        "Tmax",
        "Tmin",
        "Vp",
    ]
    m = pd.read_csv(metfile, sep="\\s+", skiprows=4, names=header)
    m["Date"] = pd.to_datetime(dict(year=m.Year, month=m.Month, day=m.Day))
    m = m.set_index("Date")
    return m


def read_q(qfile):
    """
    Reads CAMELS discharge observations.

    Parameters:
    -----------
    qfile : str
        Path to the discharge observations file

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing discharge data with Date as index
    """
    q = pd.read_csv(
        qfile,
        sep="\\s+",
        header=None,
        na_values="-999.00",
        names=["Gauge", "Year", "Month", "Day", "Flow", "Flag"],
    )
    q["Date"] = pd.to_datetime(dict(year=q.Year, month=q.Month, day=q.Day))
    q = q.set_index("Date")
    return q


def read_static_attributes(acols, datadir="data"):
    """
    Reads static attributes for CAMELS basins.

    Parameters:
    -----------
    acols : list
        List of column names to select from the static attributes
    datadir : str, optional
        Directory path containing the static attribute files (default: "data")

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing selected static attributes for all basins
    """
    veg = pd.read_csv(
        f"{datadir}/camels_vege.txt",
        sep=";",
        index_col="gauge_id",
        dtype={"gauge_id": str},
    )
    clim = pd.read_csv(
        f"{datadir}/camels_clim.txt",
        sep=";",
        index_col="gauge_id",
        dtype={"gauge_id": str},
    )
    geol = pd.read_csv(
        f"{datadir}/camels_geol.txt",
        sep=";",
        index_col="gauge_id",
        dtype={"gauge_id": str},
    )
    hyd = pd.read_csv(
        f"{datadir}/camels_hydro.txt",
        sep=";",
        index_col="gauge_id",
        dtype={"gauge_id": str},
    )
    soil = pd.read_csv(
        f"{datadir}/camels_soil.txt",
        sep=";",
        index_col="gauge_id",
        dtype={"gauge_id": str},
    )
    topo = pd.read_csv(
        f"{datadir}/camels_topo.txt",
        sep=";",
        index_col="gauge_id",
        dtype={"gauge_id": str},
    )
    attribs = pd.concat([veg, clim, geol, hyd, soil, topo], axis=1, join="inner")
    return attribs.loc[:, acols]


def get_data(bids, forcing, tstart, tend, seq_len, attribs, datadir="data"):
    """
    Extracts and processes data for specified basins over a time period.

    Parameters:
    -----------
    bids : list
        List of basin IDs to process
    forcing : str
        Forcing dataset name
    tstart : str or datetime
        Start date for data extraction
    tend : str or datetime
        End date for data extraction
    seq_len : int
        Length of sequences to generate
    attribs : pandas.DataFrame
        Static attributes DataFrame
    datadir : str, optional
        Directory path containing data files (default: "data")

    Returns:
    --------
    tuple
        Tuple of (xt, xst, yt, valid_bids) where:
        - xt: list of dynamic input sequences
        - xst: list of static input features
        - yt: list of target outputs
        - valid_bids: list of valid basin IDs
    """
    tstart = pd.to_datetime(tstart)
    tend = pd.to_datetime(tend)
    ndays = (tend - tstart).days + 1
    xt = []
    xst = []
    yt = []
    valid_bids = []
    for bid in bids:
        metfile = f"{datadir}/{forcing}/{bid}_lump_forcing_leap.txt"
        m = read_met(metfile)
        qfile = f"{datadir}/usgs/{bid}_streamflow_qc.txt"
        q = read_q(qfile)
        data = pd.merge(
            m.loc[:, ["Prcp", "Srad", "Tmax", "Tmin", "Vp"]],
            q.loc[:, "Flow"],
            left_index=True,
            right_index=True,
        ).dropna()
        sdata = (attribs.loc[bid, :] - attribs.mean()) / attribs.std()
        tr = data.loc[
            tstart - pd.DateOffset(days=seq_len-1) : tend,
            ["Prcp", "Tmax", "Tmin", "Srad", "Vp"],
        ]
        # REVIEW: if intermittent basins cause issues we can comment this back in
        if len(tr) == ndays + seq_len - 1:  # and (q['Flow'] > 0).all():
            xt_ = tr.values.astype(np.float32)
            xst_ = sdata.values.astype(np.float32)
            yt_ = data.loc[tstart:tend, "Flow"].values.reshape(-1, 1)
            yt_ = (
                yt_ * 0.0283168 * 86400 / (attribs.loc[bid, "area_gages2"] * 1e6)
            )  # convert from cfs to mm/day
            xt.append(xt_)
            xst.append(xst_)
            yt.append(yt_)
            valid_bids.append(bid)
    return xt, xst, yt, valid_bids


def write_data(xt, xst, yt, sb, seq_len, h5path, chunksize=512):
    """
    Writes processed data to an HDF5 file with normalization.

    Parameters:
    -----------
    xt : list
        List of dynamic input sequences
    xst : list
        List of static input features
    yt : list
        List of target outputs
    sb : list
        List of target (flow) standard deviation values
    seq_len : int
        Length of sequences
    h5path : str
        Path to output HDF5 file
    chunksize : int, optional
        Size of data chunks for HDF5 compression (default: 512)
    """
    n_samples = sum(len(y) for y in yt)
    n_dynamic = xt[0].shape[-1]
    n_static = xst[0].shape[-1]
    with h5py.File(h5path, "w") as f:
        chunk_rows = min(chunksize, n_samples)
        ds_x = f.create_dataset(
            "x",
            shape=(n_samples, seq_len, n_dynamic),
            dtype="float32",
            chunks=(chunk_rows, seq_len, n_dynamic),
            compression="gzip",
            compression_opts=1,
        )
        ds_xs = f.create_dataset(
            "xs",
            shape=(n_samples, n_static),
            dtype="float32",
            chunks=(chunk_rows, n_static),
            compression="gzip",
            compression_opts=1,
        )
        ds_y = f.create_dataset(
            "y",
            shape=(n_samples, 1),
            dtype="float32",
            chunks=(chunk_rows, 1),
            compression="gzip",
            compression_opts=1,
        )
        ds_s = f.create_dataset(
            "s",
            shape=(n_samples, 1),
            dtype="float32",
            chunks=(chunk_rows, 1),
            compression="gzip",
            compression_opts=1,
        )
        xsum = np.zeros(n_dynamic)
        xsum2 = np.zeros(n_dynamic)
        xcount = 0
        cursor = 0
        for b, (x_b, xs_b, y_b) in enumerate(zip(xt, xst, yt)):
            n = len(y_b)
            xsum += np.sum(x_b, axis=0)
            xsum2 += np.sum(x_b**2, axis=0)
            xcount += x_b.shape[0]
            x_seq = np.stack([x_b[t : t + seq_len] for t in range(n)], axis=0)
            ds_x[cursor : cursor + n] = x_seq
            ds_xs[cursor : cursor + n] = np.tile(xs_b, (n, 1))
            ds_y[cursor : cursor + n] = y_b.reshape(n, 1)
            ds_s[cursor : cursor + n] = float(sb[b])
            cursor += n
        xmean = xsum / xcount
        xstd = np.sqrt(xsum2 / xcount - xmean**2)
        xstd = np.maximum(xstd, 1e-8)  # avoid division with zero
        f.attrs["xmean"] = xmean.astype(np.float32)
        f.attrs["xstd"] = xstd.astype(np.float32)
        f.attrs["seq_len"] = seq_len

def generate_h5file(forcing, tstart, tend, seq_len, acols, datadir):
    bids = pd.read_csv(f"{datadir}/bids.csv", header=None, dtype=str).iloc[:, 0].values
    attribs = read_static_attributes(acols, datadir)
    xt, xst, yt, bids = get_data(bids, forcing, tstart, tend, seq_len, attribs)
    sb = np.array([y.std() for y in yt])
    h5path = f"{datadir}/{forcing}.h5"
    write_data(xt, xst, yt, sb, seq_len, h5path)
    with h5py.File(h5path, "a") as f:
        ds_bid = f.create_dataset("bids", shape=(len(bids), ), dtype=h5py.string_dtype())
        ds_bid[:] = np.array(bids)
        f.attrs["tstart"] = tstart
        f.attrs["tend"] = tend

def dataloader(h5path: str, batch_size:int, key: jrn.PRNGKey, shuffle: bool=True, preload: bool=False, perturbation: Optional['Perturbation']=None, pert_key: Optional[jrn.PRNGKey]=None):
    """
    Creates a data loader for HDF5 dataset.

    Parameters:
    -----------
    h5path : str
        Path to the HDF5 file
    batch_size : int
        Size of each batch to yield
    key : jax.random.PRNGKey
        Random key for shuffling
    shuffle : bool, optional
        Whether to shuffle data (default: True)
    preload : bool, optional
        Whether to load the entire dataset into RAM at startup (default: False).
        Significantly faster if enough RAM is available (recommended if >64GB RAM).
    perturbation : Perturbation, optional
        Perturbation object to apply to inputs (default: None)
    pert_key : jax.random.PRNGKey, optional
        Random key for reproducible perturbations

    Yields:
    -------
    tuple
        Batch of (x, xs, y, s) tensors
    """
    if preload:
        print("Loading dataset into memory...")
        with h5py.File(h5path, "r") as f:
            x_all  = f["x"][:]
            xs_all = f["xs"][:]
            y_all  = f["y"][:]
            s_all  = f["s"][:]
            xmean  = f.attrs["xmean"]
            xstd   = f.attrs["xstd"]
        print("Dataset loaded.")
        # update scaling parameters if perturbation is used
        if perturbation is not None:
            xmean, xstd = perturbation.compute_stats(xmean, xstd)
        n_samples = x_all.shape[0]
        while True:
            indices = np.arange(n_samples)
            if shuffle:
                key, subkey = jrn.split(key)
                np_seed = int(jrn.randint(subkey, (), 0, 2**31 - 1))
                rng = np.random.default_rng(np_seed)
                rng.shuffle(indices)
            for start in range(0, n_samples - batch_size + 1, batch_size):
                batch_idx = indices[start : start + batch_size]
                x_batch = jnp.asarray(x_all[batch_idx])
                if perturbation is not None:
                    if pert_key is not None:
                        x_batch, pert_key = perturbation(x_batch, pert_key)
                    else:
                        x_batch, _ = perturbation(x_batch)
                x_batch = (x_batch - xmean) / xstd
                yield (
                    x_batch,
                    jnp.asarray(xs_all[batch_idx]),
                    jnp.asarray(y_all[batch_idx]),
                    jnp.asarray(s_all[batch_idx]),
                )
    else:
        with h5py.File(h5path, "r") as f:
            n_samples = f["x"].shape[0]
            xmean = f.attrs["xmean"]
            xstd  = f.attrs["xstd"]
            while True:
                indices = np.arange(n_samples)
                if shuffle:
                    key, subkey = jrn.split(key)
                    np_seed = int(jrn.randint(subkey, (), 0, 2**31 - 1))
                    rng = np.random.default_rng(np_seed)
                    rng.shuffle(indices)
                for start in range(0, n_samples - batch_size + 1, batch_size):
                    batch_idx = indices[start : start + batch_size]
                    sorted_pos = np.argsort(batch_idx)
                    sorted_idx = batch_idx[sorted_pos]
                    x_np  = f["x"][sorted_idx]
                    xs_np = f["xs"][sorted_idx]
                    y_np  = f["y"][sorted_idx]
                    s_np  = f["s"][sorted_idx]
                    inv = np.argsort(sorted_pos)
                    x_batch = jnp.asarray(x_np[inv])
                    if perturbation is not None:
                        if pert_key is not None:
                            x_batch, pert_key = perturbation(x_batch, pert_key)
                        else:
                            x_batch, _ = perturbation(x_batch)
                    x_batch = (x_batch - xmean) / xstd
                    yield (
                        x_batch,
                        jnp.asarray(xs_np[inv]),
                        jnp.asarray(y_np[inv]),
                        jnp.asarray(s_np[inv]),
                    )
