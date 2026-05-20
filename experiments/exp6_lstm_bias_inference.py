import h5py
import equinox as eqx
import jax.random as jrn
from pathlib import Path

from lstm.train import train_ealstm
from lstm.models import EALSTM
from lstm import config
from lstm.evals import evaluate
from lstm.perturb import BiasPerturbation


DIMS = (0,)


def get_biases(args):
    if args.bias is None:
        return [0.5, 1.5]
    return args.bias

def load_model(model_path, h5path, args):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            f"Train the baseline model with {args.forcing}."
        )
    else:
        with h5py.File(str(h5path), "r") as f:
            nvars = f["x"].shape[2]
            nsvars = f["xs"].shape[1]
            train_xmean = f.attrs["xmean"]
            train_xstd = f.attrs["xstd"]
        model_key = jrn.split(jrn.PRNGKey(args.seed), 4)[1]
        skeleton = EALSTM(nvars, nsvars, args.hidden_size, 1, args.dropout_rate, key=model_key)
        model = eqx.tree_deserialise_leaves(str(model_path), skeleton)
        print(f"Model loaded from : {model_path}")
    return model, train_xmean, train_xstd


def main():
    args = config.parse_args_lstm()
    biases = get_biases(args)
    forcing = args.forcing

    h5path = Path(args.data_dir) / f"{forcing}.h5"
    if not h5path.exists():
        raise FileNotFoundError(
            f"HDF5 file not found: {h5path}. "
            f"Expected a file named '{forcing}.h5' in '{args.data_dir}'."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"ealstm_{forcing}.eqx"
    config_path = output_dir / f"ealstm_{forcing}.toml"

    model, train_xmean, train_xstd = load_model(model_path, h5path, args)

    val_tstart = "2000-10-01"
    val_tend = "2008-09-30"

    if not config_path.exists():
        run_config = config.build(args, h5path, model_path, config_path)
        run_config["output"]["valid_tstart"] = val_tstart
        run_config["output"]["valid_tend"] = val_tend
        config.save(run_config, config_path)
        print(f"Config saved to   : {config_path}")
    else:
        print(f"Config already exists at {config_path}, skipping save.")

    for bias in biases:
        perturbation = BiasPerturbation(mbias=bias, dims=DIMS)
        suffix = f"infer_bias{bias}"

        print(f"\nEvaluating with '{forcing}' forcings (training period)...")
        mod, obs = evaluate(model, forcing, datadir=args.data_dir, xmean=train_xmean, xstd=train_xstd, perturbation=perturbation)
        mod.to_csv(f"{output_dir}/ealstm_{forcing}_{suffix}_train_predictions.csv")
        obs.to_csv(f"{output_dir}/ealstm_{forcing}_{suffix}_train_observations.csv")

        print(f"Evaluating with '{forcing}' forcings (validation period)...")
        mod, obs = evaluate(model, forcing, val_tstart, val_tend, datadir=args.data_dir, xmean=train_xmean, xstd=train_xstd, perturbation=perturbation)
        mod.to_csv(f"{output_dir}/ealstm_{forcing}_{suffix}_valid_predictions.csv")
        obs.to_csv(f"{output_dir}/ealstm_{forcing}_{suffix}_valid_observations.csv")


if __name__ == "__main__":
    main()
