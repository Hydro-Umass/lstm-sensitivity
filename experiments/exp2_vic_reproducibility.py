import h5py
from pathlib import Path

from lstm import config, vic

def main():
    args = config.parse_args_vic()

    train_forcing = args.forcing

    output_dir = Path(args.output_dir)

    with h5py.File(f"{args.data_dir}/{train_forcing}.h5") as f:
        tstart = f.attrs["tstart"]
        tend = f.attrs["tend"]
        bids = [b.decode() for b in f["bids"][:]]

    val_tstart = "2000-10-01"
    val_tend = "2008-09-30"

    eval_forcings = [f for f in config.FORCINGS if f != train_forcing]
    print(f"\nModel calibrated on {train_forcing} will be evaluated on: {eval_forcings}")

    for eval_forcing in eval_forcings:
        eval_soilfile = f"{output_dir}/vic_{train_forcing}_baseline_soil.txt"

        print(f"\nEvaluating with '{eval_forcing}' forcings (training period)...")
        mod, obs = vic.evaluate(bids, outfile, eval_forcing, tstart, tend, datadir=args.data_dir, vic_exec=args.vic_exec)
        mod.to_csv(f"{output_dir}/vic_train{train_forcing}_eval{eval_forcing}_train_predictions.csv")
        obs.to_csv(f"{output_dir}/vic_train{train_forcing}_eval{eval_forcing}_train_observations.csv")

        print(f"Evaluating with '{eval_forcing}' forcings (validation period)...")
        mod, obs = vic.evaluate(bids, outfile, eval_forcing, val_tstart, val_tend, datadir=args.data_dir, vic_exec=args.vic_exec)
        mod.to_csv(f"{output_dir}/vic_train{train_forcing}_eval{eval_forcing}_valid_predictions.csv")
        obs.to_csv(f"{output_dir}/vic_train{train_forcing}_eval{eval_forcing}_valid_observations.csv")


if __name__ == "__main__":
    main()
