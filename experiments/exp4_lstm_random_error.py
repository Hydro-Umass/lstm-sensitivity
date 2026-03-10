import argparse
import equinox as eqx
from pathlib import Path

from lstm.train import train_ealstm
from lstm import config
from lstm.evals import evaluate
from lstm.perturb import RandomPerturbation


DIMS = (0,)


def get_stddevs(args):
    if args.stddev is None:
        return [0.1, 0.25, 0.5]
    return args.stddev


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
        suffix = f"random_error_std{stddev}"
        print(f"\n{'='*60}")
        print(f"Training with RandomPerturbation stddev={stddev}")
        print(f"{'='*60}")

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
        mod, obs = evaluate(model, forcing, datadir=args.data_dir, perturbation=perturbation)
        mod.to_csv(f"{output_dir}/ealstm_{forcing}_{suffix}_train_predictions.csv")
        obs.to_csv(f"{output_dir}/ealstm_{forcing}_{suffix}_train_observations.csv")

        print(f"Evaluating with '{forcing}' forcings + {suffix} (validation period)...")
        mod, obs = evaluate(model, forcing, val_tstart, val_tend, datadir=args.data_dir, perturbation=perturbation)
        mod.to_csv(f"{output_dir}/ealstm_{forcing}_{suffix}_valid_predictions.csv")
        obs.to_csv(f"{output_dir}/ealstm_{forcing}_{suffix}_valid_observations.csv")


if __name__ == "__main__":
    main()
