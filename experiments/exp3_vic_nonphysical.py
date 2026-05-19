from pathlib import Path
from tqdm import tqdm
import h5py

from lstm import config, vic
from lstm.perturb import ZeroPrecipitation

PERTURBATION = ZeroPrecipitation(dims=(0,))
SUFFIX = "zero_precip"

def main():
    args = config.parse_args_vic()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(f"{args.data_dir}/{args.forcing}.h5") as f:
        startdate = f.attrs["tstart"]
        enddate = f.attrs["tend"]
        bids = [b.decode() for b in f["bids"][:]]

    soilfile = f"{args.data_dir}/vic/vic.nldas.mexico.soil.txt"
    outfile = f"{output_dir}/vic_{args.forcing}_{SUFFIX}_baseline_soil.txt"

    # Calibrate with zero precipitation
    print(f"Calibrating VIC with {args.forcing} forcings + {SUFFIX}...")
    with open(outfile, "w") as fout:
        for bid in tqdm(bids):
            line = vic.calibrate(
                bid, soilfile, args.forcing, startdate, enddate,
                datadir=args.data_dir,
                vic_exec=args.vic_exec,
                perturbation=PERTURBATION,
            )
            fout.write(line)

    # Evaluate training period
    print(f"\nEvaluating {args.forcing} + {SUFFIX} (training period)...")
    mod, obs = vic.evaluate(
        bids, outfile, args.forcing, startdate, enddate,
        datadir=args.data_dir,
        vic_exec=args.vic_exec,
        perturbation=PERTURBATION,
    )
    mod.to_csv(f"{output_dir}/vic_{args.forcing}_{SUFFIX}_train_predictions.csv")
    obs.to_csv(f"{output_dir}/vic_{args.forcing}_{SUFFIX}_train_observations.csv")

    # Evaluate validation period
    val_tstart = "2000-10-01"
    val_tend = "2008-09-30"
    print(f"Evaluating {args.forcing} + {SUFFIX} (validation period)...")
    mod, obs = vic.evaluate(
        bids, outfile, args.forcing, val_tstart, val_tend,
        datadir=args.data_dir,
        vic_exec=args.vic_exec,
        zero_precip=True,
    )
    mod.to_csv(f"{output_dir}/vic_{args.forcing}_{SUFFIX}_valid_predictions.csv")
    obs.to_csv(f"{output_dir}/vic_{args.forcing}_{SUFFIX}_valid_observations.csv")

if __name__ == "__main__":
    main()
