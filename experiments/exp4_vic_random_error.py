from pathlib import Path
from tqdm import tqdm
import h5py
import numpy as np

from lstm import config, vic, perturb

# TODO: we should make perturb use numpy instead of jax but need to make sure it's compatible with the LSTM dataloader
import warnings

def fxn():
    warnings.warn("deprecated", RuntimeWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

def get_stddevs(args):
    if args.stddev is None:
        return [0.1, 0.25, 0.5]
    return args.stddev

def main():
    args = config.parse_args_vic()
    stddevs = get_stddevs(args)
    forcing = args.forcing

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(f"{args.data_dir}/{forcing}.h5") as f:
        startdate = f.attrs["tstart"]
        enddate = f.attrs["tend"]
        bids = [b.decode() for b in f["bids"][:]]

    soilfile = f"{args.data_dir}/vic/vic.nldas.mexico.soil.txt"

    for stddev in stddevs:
        suffix = f"random_error_std{stddev}"
        perturbation = perturb.RandomPerturbation(stddev=stddev, dims=(0,))
        print(f"\n{'='*60}")
        print(f"Calibrating VIC with {forcing} forcings + {suffix}...")
        print(f"{'='*60}")

        outfile = f"{output_dir}/vic_{forcing}_{suffix}_baseline_soil.txt"
        with open(outfile, "w") as fout:
            for bid in tqdm(bids):
                line = vic.calibrate(
                    bid, soilfile, forcing, startdate, enddate,
                    datadir=args.data_dir,
                    vic_exec=args.vic_exec,
                    perturbation=perturbation,
                    seed=args.seed
                )
                fout.write(line)

        print(f"\nEvaluating {forcing} + {suffix} (training period)...")
        mod, obs = vic.evaluate(
            bids, outfile, forcing, startdate, enddate,
            datadir=args.data_dir,
            vic_exec=args.vic_exec,
            perturbation=perturbation,
            seed=args.seed
        )
        mod.to_csv(f"{output_dir}/vic_{forcing}_{suffix}_train_predictions.csv")
        obs.to_csv(f"{output_dir}/vic_{forcing}_{suffix}_train_observations.csv")

        val_tstart = "2000-10-01"
        val_tend = "2008-09-30"
        print(f"Evaluating {forcing} + {suffix} (validation period)...")
        mod, obs = vic.evaluate(
            bids, outfile, forcing, val_tstart, val_tend,
            datadir=args.data_dir,
            vic_exec=args.vic_exec,
            perturbation=perturbation,
            seed=args.seed
        )
        mod.to_csv(f"{output_dir}/vic_{forcing}_{suffix}_valid_predictions.csv")
        obs.to_csv(f"{output_dir}/vic_{forcing}_{suffix}_valid_observations.csv")

if __name__ == "__main__":
    main()
