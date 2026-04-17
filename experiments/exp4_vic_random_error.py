from pathlib import Path
from tqdm import tqdm
import h5py

from lstm import config, vic


DIMS_LABEL = "precip"


def get_stddevs(args):
    if args.stddev is None:
        return [0.1, 0.25, 0.5]
    return args.stddev


def main():
    args = config.parse_args_vic()
    stddevs = get_stddevs(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(f"{args.data_dir}/{args.forcing}.h5") as f:
        startdate = f.attrs["tstart"]
        enddate = f.attrs["tend"]
        bids = [b.decode() for b in f["bids"][:]]

    soilfile = f"{args.data_dir}/vic/vic.nldas.mexico.soil.txt"

    val_tstart = "2000-10-01"
    val_tend = "2008-09-30"

    for stddev in stddevs:
        suffix = f"random_error_std{stddev}"
        print(f"Calibrating VIC with {args.forcing} + {suffix}")

        outfile = f"{output_dir}/vic_{args.forcing}_{suffix}_baseline_soil.txt"

        #Calibrate with noisy precipitation
        with open(outfile, "w") as fout:
            for bid in tqdm(bids):
                line = vic.calibrate(
                    bid, soilfile, args.forcing, startdate, enddate,
                    datadir=args.data_dir,
                    vic_exec=args.vic_exec,
                    noise_stddev=stddev,
                )
                fout.write(line)

        #Evaluate training period
        print(f"\nEvaluating {args.forcing} + {suffix} (training period)...")
        mod, obs = vic.evaluate(
            bids, outfile, args.forcing, startdate, enddate,
            datadir=args.data_dir,
            vic_exec=args.vic_exec,
            noise_stddev=stddev,
        )
        mod.to_csv(f"{output_dir}/vic_{args.forcing}_{suffix}_train_predictions.csv")
        obs.to_csv(f"{output_dir}/vic_{args.forcing}_{suffix}_train_observations.csv")

        #Evaluate validation period
        print(f"Evaluating {args.forcing} + {suffix} (validation period)...")
        mod, obs = vic.evaluate(
            bids, outfile, args.forcing, val_tstart, val_tend,
            datadir=args.data_dir,
            vic_exec=args.vic_exec,
            noise_stddev=stddev,
        )
        mod.to_csv(f"{output_dir}/vic_{args.forcing}_{suffix}_valid_predictions.csv")
        obs.to_csv(f"{output_dir}/vic_{args.forcing}_{suffix}_valid_observations.csv")

    print("\nAll stddevs done!")


if __name__ == "__main__":
    main()