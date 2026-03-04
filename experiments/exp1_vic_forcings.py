from pathlib import Path
from tqdm import tqdm
import h5py

from lstm import config, vic

def main():
    args = config.parse_args_vic()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(f"{args.data_dir}/{args.forcing}.h5") as f:
        startdate = f.attrs["tstart"]
        enddate = f.attrs["tend"]
        bids = [b.decode() for b in f["bids"][:]]
    soilfile = f"{args.data_dir}/vic/vic.nldas.mexico.soil.txt"
    outfile = f"{output_dir}/vic_{args.forcing}_baseline_soil.txt"
    with open(outfile, "w") as fout:
        for bid in tqdm(bids):
            line = vic.calibrate(bid, soilfile, args.forcing, startdate, enddate, datadir=args.data_dir)
            fout.write(line)
    mod, obs = vic.evaluate(bids, outfile, args.forcing, startdate, enddate, args.data_dir)
    mod.to_csv(f"{output_dir}/vic_{args.forcing}_train_predictions.csv")
    obs.to_csv(f"{output_dir}/vic_{args.forcing}_train_observations.csv")
    val_tstart = "2000-10-01"
    val_tend = "2008-09-30"
    mod, obs = vic.evaluate(bids, outfile, args.forcing, val_tstart, val_tend, args.data_dir)
    mod.to_csv(f"{output_dir}/vic_{args.forcing}_valid_predictions.csv")
    obs.to_csv(f"{output_dir}/vic_{args.forcing}_valid_observations.csv")


if __name__ == "__main__":
    main()
