from pathlib import Path
from tqdm import tqdm
import h5py

from lstm import config, vic
from lstm.perturb import BiasPerturbation

import warnings

def fxn():
    warnings.warn("deprecated", RuntimeWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

DIMS = (0,)


def get_biases(args):
    if args.bias is None:
        return [0.5, 1.5]
    return args.bias


def main():
    args = config.parse_args_vic()
    biases = get_biases(args)
    forcing = args.forcing

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(f"{args.data_dir}/{forcing}.h5") as f:
        startdate = f.attrs["tstart"]
        enddate = f.attrs["tend"]
        bids = [b.decode() for b in f["bids"][:]]

    val_tstart = "2000-10-01"
    val_tend = "2008-09-30"

    for bias in biases:
        train_suffix = f"bias{bias}"
        eval_suffix = f"bias{bias}_noperturb"

        # Soil file produced by exp5 (calibrated with biased precipitation).
        soilfile = f"{output_dir}/vic_{forcing}_{train_suffix}_baseline_soil.txt"
        if not Path(soilfile).exists():
            raise FileNotFoundError(
                f"Soil file not found: {soilfile}. "
                f"Run exp5_vic_bias.py first to calibrate VIC with the biased forcing."
            )

        # Evaluate with baseline (unperturbed) precipitation.
        print(f"\nEvaluating VIC '{forcing}' baseline forcings with model "
              f"calibrated on {train_suffix} (training period)...")
        mod, obs = vic.evaluate(
            bids, soilfile, forcing, startdate, enddate,
            datadir=args.data_dir,
            vic_exec=args.vic_exec,
        )
        mod.to_csv(f"{output_dir}/vic_{forcing}_{eval_suffix}_train_predictions.csv")
        obs.to_csv(f"{output_dir}/vic_{forcing}_{eval_suffix}_train_observations.csv")

        print(f"Evaluating VIC '{forcing}' baseline forcings with model "
              f"calibrated on {train_suffix} (validation period)...")
        mod, obs = vic.evaluate(
            bids, soilfile, forcing, val_tstart, val_tend,
            datadir=args.data_dir,
            vic_exec=args.vic_exec,
        )
        mod.to_csv(f"{output_dir}/vic_{forcing}_{eval_suffix}_valid_predictions.csv")
        obs.to_csv(f"{output_dir}/vic_{forcing}_{eval_suffix}_valid_observations.csv")


if __name__ == "__main__":
    main()
