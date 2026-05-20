# create_precip_only_h5.py
"""
Reads existing full HDF5 (5 dynamic features) and creates a new one
with only precipitation (feature index 0).
"""
import h5py
import numpy as np
import argparse
from pathlib import Path


def create_precip_only(input_h5, output_h5):
    print(f"Input:  {input_h5}")
    print(f"Output: {output_h5}")

    with h5py.File(input_h5, "r") as fin, h5py.File(output_h5, "w") as fout:
        n_samples, seq_len, n_features = fin["x"].shape
        n_static = fin["xs"].shape[1]
        print(f" Original x shape: ({n_samples}, {seq_len}, {n_features})")
        print(f" New x shape:      ({n_samples}, {seq_len}, 1)")

        # Creating only 1 dynamic feature datasets in new file
        chunk_rows = min(512, n_samples)
        ds_x = fout.create_dataset(
            "x", shape=(n_samples, seq_len, 1), dtype="float32",
            chunks=(chunk_rows, seq_len, 1), compression="gzip", compression_opts=1,
        )
        ds_xs = fout.create_dataset(
            "xs", shape=(n_samples, n_static), dtype="float32",
            chunks=(chunk_rows, n_static), compression="gzip", compression_opts=1,
        )
        ds_y = fout.create_dataset(
            "y", shape=(n_samples, 1), dtype="float32",
            chunks=(chunk_rows, 1), compression="gzip", compression_opts=1,
        )
        ds_s = fout.create_dataset(
            "s", shape=(n_samples, 1), dtype="float32",
            chunks=(chunk_rows, 1), compression="gzip", compression_opts=1,
        )

        #copying data in batches
        batch = 50000 #50,000 rows at a time for memory safety
        for start in range(0, n_samples, batch):
            end = min(start + batch, n_samples)
            ds_x[start:end] = fin["x"][start:end, :, 0:1]   #only precipitation
            ds_xs[start:end] = fin["xs"][start:end]          #unchanged
            ds_y[start:end] = fin["y"][start:end]            #unchanged
            ds_s[start:end] = fin["s"][start:end]            #unchanged
            print(f"  Copied {end}/{n_samples} ({100*end/n_samples:.1f}%)")

        #copying basin IDs
        ds_bid = fout.create_dataset("bids", shape=fin["bids"].shape, dtype=h5py.string_dtype())
        ds_bid[:] = fin["bids"][:]

        #copying attributes (only precip stats)
        fout.attrs["xmean"] = fin.attrs["xmean"][0:1]   #only precip mean
        fout.attrs["xstd"] = fin.attrs["xstd"][0:1]     #only precip std
        fout.attrs["seq_len"] = fin.attrs["seq_len"]
        fout.attrs["tstart"] = fin.attrs["tstart"]
        fout.attrs["tend"] = fin.attrs["tend"]


def main():
    parser = argparse.ArgumentParser(description="Creating precip-only HDF5 from full HDF5.")
    parser.add_argument("forcing", type=str, choices=["daymet", "nldas", "maurer"])
    parser.add_argument("--data-dir", type=str, default="data")
    args = parser.parse_args()

    input_h5 = Path(args.data_dir) / f"{args.forcing}.h5"
    output_h5 = Path(args.data_dir) / f"{args.forcing}_precip_only.h5"

    if not input_h5.exists():
        raise FileNotFoundError(f"Not found: {input_h5}")

    create_precip_only(str(input_h5), str(output_h5))


if __name__ == "__main__":
    main()