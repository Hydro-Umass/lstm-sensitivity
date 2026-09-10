# compute_ensemble_mean.py
import pandas as pd
from pathlib import Path
import argparse

SEEDS = [111, 222, 333, 444, 555]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("forcing", type=str)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    forcing = args.forcing

    for period in ["train", "valid"]:
        preds = []
        for seed in SEEDS:
            f = output_dir / f"ealstm_{forcing}_{seed}_{period}_predictions.csv"
            if not f.exists():
                print(f"Missing: {f}")
                continue
            preds.append(pd.read_csv(f, index_col=0))
            print(f"Loaded: {f}")

        if len(preds) == 0:
            print(f"No predictions found for {period} period!")
            continue

        ensemble = sum(preds) / len(preds)
        out_path = output_dir / f"ealstm_{forcing}_ensemble_{period}_predictions.csv"
        ensemble.to_csv(out_path)
        print(f"Saved ensemble mean ({len(preds)} models): {out_path}")


if __name__ == "__main__":
    main()