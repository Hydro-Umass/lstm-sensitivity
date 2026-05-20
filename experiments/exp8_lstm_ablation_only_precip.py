import h5py
import numpy as np
import pandas as pd
import equinox as eqx
import jax
import jax.random as jrn
import jax.numpy as jnp
from pathlib import Path
from tqdm import tqdm

from lstm.train import train_ealstm
from lstm import camels, config


def evaluate_precip_only(model, forcing, tstart=None, tend=None, bids=None, datadir="data"):
    """
    Evaluates model trained on precipitation-only input.
    Reads metadata from the precip-only HDF5.
    """
    h5path = f"{datadir}/{forcing}_precip_only.h5"
    with h5py.File(h5path, "r") as f:
        xmean = f.attrs["xmean"]
        xstd = f.attrs["xstd"]
        seq_len = int(f.attrs["seq_len"])
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
        xt_precip = xt[bi][:, 0:1] # Slice to only precipitation (column 0)
        n = xt_precip.shape[0] - seq_len + 1
        xt_ = np.stack([xt_precip[t:t+seq_len, :] for t in range(n)])
        xst_ = np.stack([xst[bi] for _ in range(n)])

        xt_ = jnp.asarray(xt_)
        xt_ = (xt_ - xmean) / xstd

        inf_model = eqx.nn.inference_mode(model)
        pred = jax.vmap(inf_model)(xt_, xst_, key=jrn.split(jrn.PRNGKey(0), xt_.shape[0]))
        mod[bid] = pd.Series(pred[:, 0], dt)
        obs[bid] = pd.Series(yt[bi][:, 0], dt)

    return pd.DataFrame(mod), pd.DataFrame(obs)


def main():
    args = config.parse_args_lstm()
    forcing = args.forcing

    h5path = Path(args.data_dir) / f"{forcing}_precip_only.h5"
    if not h5path.exists():
        raise FileNotFoundError(
            f"Precip-only HDF5 file not found: {h5path}. "
            f"Run create_precip_only_h5.py first to generate it."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"ealstm_{forcing}_precip_only.eqx"
    config_path = output_dir / f"ealstm_{forcing}_precip_only.toml"

    model = train_ealstm(
        h5path=str(h5path),
        hidden_size=args.hidden_size,
        dropout_rate=args.dropout_rate,
        batch_size=args.batch_size,
        seed=args.seed,
        epochs=args.epochs,
        clip_gradient_norm=args.clip_gradient_norm,
        preload=args.preload,
    )

    eqx.tree_serialise_leaves(str(model_path), model)
    print(f"\nModel saved to  : {model_path}")

    val_tstart = "2000-10-01"
    val_tend = "2008-09-30"
    run_config = config.build(args, h5path, model_path, config_path)
    run_config["output"]["valid_tstart"] = val_tstart
    run_config["output"]["valid_tend"] = val_tend
    run_config["experiment"] = {"type": "ablation", "dynamic_inputs": ["Prcp"]}
    config.save(run_config, config_path)
    print(f"Config saved to : {config_path}")

    print(f"\nEvaluating with '{forcing}' precip-only (training period)...")
    mod, obs = evaluate_precip_only(model, forcing, datadir=args.data_dir)
    mod.to_csv(f"{output_dir}/ealstm_{forcing}_precip_only_train_predictions.csv")
    obs.to_csv(f"{output_dir}/ealstm_{forcing}_precip_only_train_observations.csv")

    print(f"Evaluating with '{forcing}' precip-only (validation period)...")
    mod, obs = evaluate_precip_only(model, forcing, val_tstart, val_tend, datadir=args.data_dir)
    mod.to_csv(f"{output_dir}/ealstm_{forcing}_precip_only_valid_predictions.csv")
    obs.to_csv(f"{output_dir}/ealstm_{forcing}_precip_only_valid_observations.csv")

    print(f"\nExperiment 8 complete")


if __name__ == "__main__":
    main()