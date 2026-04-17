import h5py
import equinox as eqx
from pathlib import Path

from lstm.train import train_ealstm
from lstm import config
from lstm.evals import evaluate
from lstm.perturb import BiasPerturbation


DIMS = (0,)


def get_biases(args):
    if args.bias is None:
        return [0.5, 1.5]
    return args.bias


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

    with h5py.File(str(h5path), "r") as f:
        base_xmean = f.attrs["xmean"][:]
        base_xstd = f.attrs["xstd"][:]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    val_tstart = "2000-10-01"
    val_tend = "2008-09-30"

    for bias in biases:
        perturbation = BiasPerturbation(mbias=bias, dims=DIMS)
        suffix = f"bias{bias}"
        print(f"\n{'='*60}")
        print(f"Training with BiasPerturbation bias={bias}")
        print(f"{'='*60}")

        xmean, xstd = perturbation.compute_stats(base_xmean, base_xstd)

        model_path = output_dir / f"ealstm_{forcing}_{suffix}.eqx"
        config_path = output_dir / f"ealstm_{forcing}_{suffix}.toml"

        model = train_ealstm(
            h5path=str(h5path),
            hidden_size=args.hidden_size,
            dropout_rate=args.dropout_rate,
            batch_size=args.batch_size,
            seed=args.seed,
            epochs=args.epochs,
            clip_gradient_norm=args.clip_gradient_norm,
            preload=args.preload,
            perturbation=perturbation,
        )

        eqx.tree_serialise_leaves(str(model_path), model)
        print(f"\nModel saved to  : {model_path}")

        run_config = config.build(args, h5path, model_path, config_path)
        run_config["output"]["valid_tstart"] = val_tstart
        run_config["output"]["valid_tend"] = val_tend
        run_config["perturbation"] = perturbation.get_config()
        config.save(run_config, config_path)
        print(f"Config saved to : {config_path}")

        print(f"\nEvaluating with '{forcing}' forcings + {suffix} (training period)...")
        mod, obs = evaluate(
            model, forcing, datadir=args.data_dir,
            perturbation=perturbation, xmean=xmean, xstd=xstd
        )
        mod.to_csv(f"{output_dir}/ealstm_{forcing}_{suffix}_train_predictions.csv")
        obs.to_csv(f"{output_dir}/ealstm_{forcing}_{suffix}_train_observations.csv")

        print(f"Evaluating with '{forcing}' forcings + {suffix} (validation period)...")
        mod, obs = evaluate(
            model, forcing, val_tstart, val_tend, datadir=args.data_dir,
            perturbation=perturbation, xmean=xmean, xstd=xstd
        )
        mod.to_csv(f"{output_dir}/ealstm_{forcing}_{suffix}_valid_predictions.csv")
        obs.to_csv(f"{output_dir}/ealstm_{forcing}_{suffix}_valid_observations.csv")


if __name__ == "__main__":
    main()
