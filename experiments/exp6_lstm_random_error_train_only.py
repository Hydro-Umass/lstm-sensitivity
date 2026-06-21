import h5py
import equinox as eqx
import jax.random as jrn
from pathlib import Path

from lstm.models import EALSTM
from lstm import config
from lstm.evals import evaluate
from lstm.perturb import RandomPerturbation

DIMS = (0,)


def get_stddevs(args):
    if args.stddev is None:
        return [0.1, 0.25, 0.5]
    return args.stddev


def load_model(model_path, h5path, args):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            f"Run exp4_lstm_random_error.py first to train the perturbed model."
        )
    with h5py.File(str(h5path), "r") as f:
        nvars = f["x"].shape[2]
        nsvars = f["xs"].shape[1]
    model_key = jrn.split(jrn.PRNGKey(args.seed), 4)[1]
    skeleton = EALSTM(nvars, nsvars, args.hidden_size, 1, args.dropout_rate, key=model_key)
    model = eqx.tree_deserialise_leaves(str(model_path), skeleton)
    print(f"Model loaded from : {model_path}")
    return model


def main():
    args = config.parse_args_lstm()
    stddevs = get_stddevs(args)
    forcing = args.forcing

    h5path = Path(args.data_dir) / f"{forcing}.h5"
    if not h5path.exists():
        raise FileNotFoundError(
            f"HDF5 file not found: {h5path}. "
            f"Expected a file named '{forcing}.h5' in '{args.data_dir}'."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    val_tstart = "2000-10-01"
    val_tend = "2008-09-30"

    for stddev in stddevs:
        perturbation = RandomPerturbation(stddev=stddev, dims=DIMS)
        train_suffix = f"random_error_std{stddev}"
        eval_suffix = f"random_error_std{stddev}_noperturb"

        model_path = output_dir / f"ealstm_{forcing}_{train_suffix}.eqx"
        model = load_model(model_path, h5path, args)

        # evaluate on baseline (no perturbation); xmean/xstd read from H5 file,
        # which stores the unperturbed baseline statistics.
        print(f"\nEvaluating '{forcing}' baseline forcings with model "
              f"trained on {train_suffix} (training period)...")
        mod, obs = evaluate(model, forcing, datadir=args.data_dir)
        mod.to_csv(f"{output_dir}/ealstm_{forcing}_{eval_suffix}_train_predictions.csv")
        obs.to_csv(f"{output_dir}/ealstm_{forcing}_{eval_suffix}_train_observations.csv")

        print(f"Evaluating '{forcing}' baseline forcings with model "
              f"trained on {train_suffix} (validation period)...")
        mod, obs = evaluate(model, forcing, val_tstart, val_tend, datadir=args.data_dir)
        mod.to_csv(f"{output_dir}/ealstm_{forcing}_{eval_suffix}_valid_predictions.csv")
        obs.to_csv(f"{output_dir}/ealstm_{forcing}_{eval_suffix}_valid_observations.csv")


if __name__ == "__main__":
    main()
