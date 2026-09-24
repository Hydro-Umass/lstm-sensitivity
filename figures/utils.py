import pandas as pd
import numpy as np

def nse(o, m):
    return 1 - ((o -m)**2).sum(axis=0) / ((o - o.mean())**2).sum(axis=0)

def kge(o, m):
    r = o.corrwith(m)
    a = m.std() / o.std()
    b = m.mean() / o.mean()
    return 1 - np.sqrt((r - 1)**2 + (a - 1)**2 + (b - 1)**2)

def read_met(metfile):
    header = ["Year", "Month", "Day", "Hr", "Dayl", "Prcp", "Srad", "Swe", "Tmax", "Tmin", "Vp",]
    m = pd.read_csv(metfile, sep="\\s+", skiprows=4, names=header)
    m["Date"] = pd.to_datetime(dict(year=m.Year, month=m.Month, day=m.Day))
    m["Gauge"] = metfile.split("/")[-1].split("_")[0]
    m = m.set_index("Date")
    return m
    
def read_ensemble_files(output_dir, forcing, seeds=None, variant="",
                        model="ealstm", period="valid", skip_bad=True):
    """
    Read one prediction file per seed plus the shared observations file.

    Returns (obs, pred, used_seeds).
    """
    output_dir = Path(output_dir)
    prefix = f"{model}_{forcing}" + (f"_{variant}" if variant else "")
    suffix = f"_{period}_predictions.csv"

    if seeds is None:
        found = []
        for p in output_dir.glob(f"{prefix}_*{suffix}"):
            tail = p.name[len(prefix) + 1:-len(suffix)]
            if tail.isdigit():                    # rejects unseeded / other variants
                found.append(int(tail))
        seeds = sorted(found)
        if not seeds:
            raise FileNotFoundError(f"No seeded files matching {prefix}_*{suffix}")

    obs_path = output_dir / f"{prefix}_{period}_observations.csv"
    if not obs_path.exists():
        obs_path = output_dir / f"{model}_{forcing}_{period}_observations.csv"
    obs = pd.read_csv(obs_path, index_col=0, parse_dates=True).sort_index()

    pred_frames, used_seeds = [], []
    for seed in seeds:
        path = output_dir / f"{prefix}_{seed}{suffix}"
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
        except FileNotFoundError:
            print(f"  {prefix} seed {seed}: file not found, skipping")
            continue
            
        #check if the file has any nan value or corruption
        if skip_bad: 
            n_nan = int(df.isna().to_numpy().sum())
            if n_nan:
                print(f"  {prefix} seed {seed}: {n_nan} NaN predictions - EXCLUDED")
                continue

        df = df.loc[:, [c for c in obs.columns if c in df.columns]]
        pred_frames.append(df)
        used_seeds.append(seed)

    if not pred_frames:
        raise ValueError(f"no usable prediction files for {prefix}")

    return obs, pred_frames, used_seeds
    
def ensemble_mean(pred_frames, seeds=None):
    """Average predictions across seeds. Unchanged from your version."""
    if not pred_frames:
        raise ValueError("no prediction frames given")
    if seeds is None:
        seeds = list(range(len(pred_frames)))

    ref = pred_frames[0]
    for seed, df in zip(seeds[1:], pred_frames[1:]):
        if not df.index.equals(ref.index):
            raise ValueError(f"seed {seed}: time index differs from seed {seeds[0]}")
        if list(df.columns) != list(ref.columns):
            raise ValueError(f"seed {seed}: basin columns differ from seed {seeds[0]}")

    stack = np.stack([df.to_numpy(dtype=float) for df in pred_frames])
    return pd.DataFrame(stack.mean(axis=0), index=ref.index, columns=ref.columns)