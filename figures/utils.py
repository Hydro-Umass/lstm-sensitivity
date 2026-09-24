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
    
def ensemble_mean(pred_frames, seeds=None):
    """
    Average predictions across seeds.
    """
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