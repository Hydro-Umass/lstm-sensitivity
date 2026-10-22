import pandas as pd
import numpy as np
from pathlib import Path

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
    
def _find_csv(path):
    """Return `path` or its gzipped twin, whichever exists."""
    path = Path(path)
    for p in (path, path.with_name(path.name + ".gz")):
        if p.exists():
            return p
    raise FileNotFoundError(f"{path}[.gz] not found")


def find_seeds(output_dir, prefix, period="valid"):
    """Seeds with a prediction file named {prefix}_{seed}_{period}_predictions.csv[.gz]."""
    suffix = f"_{period}_predictions.csv"
    found = set()
    for p in Path(output_dir).glob(f"{prefix}_*{suffix}*"):
        name = p.name.removesuffix(".gz")
        if not name.endswith(suffix):
            continue
        tail = name[len(prefix) + 1:-len(suffix)]
        if tail.isdigit():
            found.add(int(tail))
    return sorted(found)


def read_ensemble_files(output_dir, forcing, seeds=None, variant="",
                        model="ealstm", period="valid", skip_bad=True):
    """
    Read one prediction file per seed plus the shared observations file.

    `forcing` may also be a cross-forcing stem such as "traindaymet_evalnldas".
    Pass seeds=[None] to read a single unseeded run (VIC, or legacy LSTM
    outputs written before seed-tagged filenames); its seed is reported as None.

    Returns (obs, pred_frames, used_seeds).
    """
    output_dir = Path(output_dir)
    prefix = f"{model}_{forcing}" + (f"_{variant}" if variant else "")
    suffix = f"_{period}_predictions.csv"

    if seeds is None:
        seeds = find_seeds(output_dir, prefix, period)
        if not seeds:
            raise FileNotFoundError(f"No seeded files matching {prefix}_*{suffix}")

    obs = pd.read_csv(_find_csv(output_dir / f"{prefix}_{period}_observations.csv"),
                      index_col=0, parse_dates=True).sort_index()

    pred_frames, used_seeds = [], []
    for seed in seeds:
        stem = prefix if seed is None else f"{prefix}_{seed}"
        try:
            df = pd.read_csv(_find_csv(output_dir / f"{stem}{suffix}"),
                             index_col=0, parse_dates=True).sort_index()
        except FileNotFoundError:
            print(f"  {prefix} seed {seed}: file not found, skipping")
            continue

        #check if the file has any nan value or corruption
        if skip_bad:
            n_nan = int(df.isna().to_numpy().sum())
            if n_nan:
                print(f"  {prefix} seed {seed}: {n_nan} NaN predictions - excluded")
                continue

        df = df.loc[:, [c for c in obs.columns if c in df.columns]]
        pred_frames.append(df)
        used_seeds.append(seed)

    if not pred_frames:
        raise ValueError(f"no usable prediction files for {prefix}")

    return obs, pred_frames, used_seeds

def ensemble_mean(pred_frames, seeds=None):
    """
    Average predictions across seeds.

    Use only for baseline skill (NSE/KGE of the ensemble hydrograph). Averaging
    members suppresses variance by construction, so alpha, beta, r, FHV, FLV and
    rMAE must come from seed_metrics/aggregate_seeds instead.
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


# Per-seed metrics
#
# Metrics are computed on each seed's hydrograph and then aggregated across
# seeds; they are never computed on the ensemble-mean hydrograph (see
# ensemble_mean). Within a seed, every metric is evaluated per basin on the
# days where both observation and simulation are available.

METRICS = ["NSE", "KGE", "r", "alpha", "beta", "FHV", "FLV", "rMAE"]


def basin_metrics(obs, sim, fhv_pct=0.98, flv_pct=0.30):
    """
    Per-basin metrics for one simulation.

    FHV and FLV are % volume errors on the days where observed flow is at or
    above the `fhv_pct` quantile, or at or below the `flv_pct` quantile. rMAE is
    the MAE divided by the mean observed flow, in %. Basins whose observed
    variance, mean or flow volume is ~0 get NaN for the affected metrics.

    Returns a DataFrame indexed by basin with one column per metric in METRICS.
    """
    basins = obs.columns.intersection(sim.columns)
    o = obs[basins]
    s = sim[basins].reindex(o.index)
    valid = o.notna() & s.notna()
    o, s = o.where(valid), s.where(valid)

    eps = 1e-6
    ss_obs = ((o - o.mean()) ** 2).sum()
    nse_ = 1 - ((o - s) ** 2).sum() / ss_obs.where(ss_obs > eps)
    r = o.corrwith(s)
    alpha = s.std() / o.std()
    beta = s.mean() / o.mean().where(o.mean().abs() > eps)
    kge_ = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    def volume_error(mask):
        vo = o.where(mask).sum()
        vs = s.where(mask).sum()
        return (vs - vo) / vo.where(vo.abs() > eps) * 100

    fhv = volume_error(o >= o.quantile(fhv_pct))
    flv = volume_error(o <= o.quantile(flv_pct))
    rmae = (s - o).abs().mean() / o.mean().where(o.mean().abs() > eps) * 100

    out = pd.DataFrame({"NSE": nse_, "KGE": kge_, "r": r, "alpha": alpha,
                        "beta": beta, "FHV": fhv, "FLV": flv, "rMAE": rmae})
    out.index.name = "basin"
    return out


def seed_metrics(obs, pred_frames, seeds=None, **kwargs):
    """
    Per-basin metrics for every seed.

    Returns a long DataFrame with a (seed, basin) MultiIndex and one column
    per metric in METRICS.
    """
    if seeds is None:
        seeds = list(range(len(pred_frames)))
    # an unseeded run (seed None) is labelled "run"; groupby would drop a None key
    seeds = ["run" if seed is None else seed for seed in seeds]
    frames = {seed: basin_metrics(obs, df, **kwargs) for seed, df in zip(seeds, pred_frames)}
    return pd.concat(frames, names=["seed", "basin"])


def aggregate_seeds(per_seed, stat="median"):
    """
    Seed-aggregated summary of seed_metrics output.

    For each seed the per-basin metric is reduced across basins with `stat`
    (default: median); the per-seed values are then summarized across seeds.

    Returns a DataFrame indexed by metric with columns:
      value   across-seed median of the per-seed basin `stat` (the reported number)
      min/max smallest and largest per-seed value
      spread  max - min
      n_seeds number of seeds contributing
    For a single run (e.g. VIC) spread is 0 and n_seeds is 1.
    """
    by_seed = per_seed.groupby(level="seed").agg(stat)
    return pd.DataFrame({
        "value": by_seed.median(),
        "min": by_seed.min(),
        "max": by_seed.max(),
        "spread": by_seed.max() - by_seed.min(),
        "n_seeds": len(by_seed),
    })


def basin_seed_median(per_seed):
    """
    Across-seed median of each per-basin metric (basin x metric).

    For CDFs and maps, where one value per basin is needed. This is a median of
    per-seed metrics, not a metric of the ensemble-mean hydrograph.
    """
    return per_seed.groupby(level="basin").median()


def ensemble_summary(output_dir, forcing, variant="", model="ealstm",
                     period="valid", seeds=None, stat="median", **kwargs):
    """
    Read a run (all available seeds) and return (aggregate, per_seed).

    VIC runs and legacy unseeded LSTM runs are read as a single member.
    """
    prefix = f"{model}_{forcing}" + (f"_{variant}" if variant else "")
    if seeds is None and not find_seeds(output_dir, prefix, period):
        seeds = [None]
    obs, preds, used = read_ensemble_files(output_dir, forcing, seeds=seeds,
                                           variant=variant, model=model, period=period)
    per_seed = seed_metrics(obs, preds, used, **kwargs)
    return aggregate_seeds(per_seed, stat=stat), per_seed
