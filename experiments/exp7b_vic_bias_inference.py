from pathlib import Path
from tqdm import tqdm
import h5py

from lstm import config, vic
from lstm.perturb import BiasPerturbation

# TODO: we should make perturb use numpy instead of jax but need to make sure it's compatible with the LSTM dataloader
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

    with h5py.File(f"{args.data_dir}/{args.forcing}.h5") as f:
        startdate = f.attrs["tstart"]
        enddate = f.attrs["tend"]
        bids = [b.decode() for b in f["bids"][:]]

    soilfile = f"{args.data_dir}/vic/vic.nldas.mexico.soil.txt"

    for bias in biases:
        perturbation = BiasPerturbation(mbias=1 + bias, dims=DIMS)
        suffix = f"infer_bias{bias}"
        outfile = f"{output_dir}/vic_{args.forcing}_baseline_soil.txt"

        print(f"\nEvaluating {args.forcing} + {suffix} (training period)...")
        mod, obs = vic.evaluate(
            bids, outfile, args.forcing, startdate, enddate,
            datadir=args.data_dir,
            vic_exec=args.vic_exec,
            perturbation=perturbation,
        )
        mod.to_csv(f"{output_dir}/vic_{args.forcing}_{suffix}_train_predictions.csv")
        obs.to_csv(f"{output_dir}/vic_{args.forcing}_{suffix}_train_observations.csv")

        val_tstart = "2000-10-01"
        val_tend = "2008-09-30"
        print(f"Evaluating {args.forcing} + {suffix} (validation period)...")
        mod, obs = vic.evaluate(
            bids, outfile, args.forcing, val_tstart, val_tend,
            datadir=args.data_dir,
            vic_exec=args.vic_exec,
            perturbation=perturbation,
        )
        mod.to_csv(f"{output_dir}/vic_{args.forcing}_{suffix}_valid_predictions.csv")
        obs.to_csv(f"{output_dir}/vic_{args.forcing}_{suffix}_valid_observations.csv")

if __name__ == "__main__":
    main()
