import equinox as eqx
from pathlib import Path

from lstm.train import train_ealstm
from lstm import config
from lstm.evals import evaluate

def main():
    args = config.parse_args_lstm()

    h5path = Path(args.data_dir) / f"{args.forcing}.h5"
    if not h5path.exists():
        raise FileNotFoundError(
            f"HDF5 file not found: {h5path}. "
            f"Expected a file named '{args.forcing}.h5' in '{args.data_dir}'."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"ealstm_{args.forcing}.eqx"
    config_path = output_dir / f"ealstm_{args.forcing}.toml"

    run_config = config.build(args, h5path, model_path, config_path)

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
    run_config["output"]["valid_tstart"] = val_tstart
    run_config["output"]["valid_tend"] = val_tend
    config.save(run_config, config_path)
    print(f"Config saved to : {config_path}")

    mod, obs = evaluate(model, args.forcing, datadir=args.data_dir)
    mod.to_csv(f"{output_dir}/ealstm_{args.forcing}_train_predictions.csv")
    obs.to_csv(f"{output_dir}/ealstm_{args.forcing}_train_observations.csv")
    mod, obs = evaluate(model, args.forcing, val_tstart, val_tend, datadir=args.data_dir)
    mod.to_csv(f"{output_dir}/ealstm_{args.forcing}_valid_predictions.csv")
    obs.to_csv(f"{output_dir}/ealstm_{args.forcing}_valid_observations.csv")

if __name__ == "__main__":
    main()
