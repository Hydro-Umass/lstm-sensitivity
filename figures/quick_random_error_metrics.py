"""Quick summary of KGE components and FHV across random error experiments."""

import pandas as pd
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("outputs")
FORCING = "daymet"
STD_LEVELS = [0.1, 0.25, 0.5]


def load(prefix, obs_cols):
    p = OUTPUT_DIR / f"{prefix}_valid_predictions.csv"
    if not p.exists():
        p = OUTPUT_DIR / f"{prefix}_valid_predictions.csv.gz"
    return pd.read_csv(p, index_col=0, parse_dates=True).loc[:, obs_cols]


def kge_components(obs, sim):
    r = obs.corrwith(sim)
    alpha = sim.std() / obs.std()
    beta  = sim.mean() / obs.mean()
    kge   = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    return r, alpha, beta, kge


def fhv(obs, sim, pct=0.98):
    """High-flow volume error: % error in total flow above obs pct quantile."""
    results = {}
    for basin in obs.columns:
        o = obs[basin].dropna()
        s = sim[basin].reindex(o.index)
        thresh = o.quantile(pct)
        mask = o >= thresh
        vol_obs = o[mask].sum()
        vol_sim = s[mask].sum()
        if vol_obs > 0:
            results[basin] = (vol_sim - vol_obs) / vol_obs * 100
    return pd.Series(results)


def summarise(r, alpha, beta, kge, fhv_vals):
    def med(s): return f"{s.median():+.3f}"
    return {"KGE": med(kge), "r": med(r), "α": med(alpha), "β": med(beta),
            "FHV (%)": f"{fhv_vals.median():+.1f}"}


rows = []

for model, prefix_base in [("LSTM", f"ealstm_{FORCING}"),
                             ("VIC",  f"vic_{FORCING}")]:
    obs = pd.read_csv(OUTPUT_DIR / f"{prefix_base}_valid_observations.csv.gz",
                      index_col=0, parse_dates=True)

    sim = load(prefix_base, obs.columns)
    r, a, b, k = kge_components(obs, sim)
    row = {"Model": model, "Condition": "baseline", "σ": "—", **summarise(r, a, b, k, fhv(obs, sim))}
    rows.append(row)

    for std in STD_LEVELS:
        lvl = f"{std:g}"
        for cond, stem in [("train+eval",   f"random_error_std{lvl}"),
                            ("train-only",   f"random_error_std{lvl}_noperturb"),
                            ("infer-only",   f"infer_random_error_std{lvl}")]:
            try:
                sim = load(f"{prefix_base}_{stem}", obs.columns)
                r, a, b, k = kge_components(obs, sim)
                row = {"Model": model, "Condition": cond, "σ": f"{int(std*100)}%",
                       **summarise(r, a, b, k, fhv(obs, sim))}
                rows.append(row)
            except FileNotFoundError:
                rows.append({"Model": model, "Condition": cond, "σ": f"{int(std*100)}%",
                             "KGE": "N/A", "r": "N/A", "α": "N/A", "β": "N/A", "FHV (%)": "N/A"})

df = pd.DataFrame(rows)
for model in ["LSTM", "VIC"]:
    print(f"\n{'='*60}")
    print(f"  {model}")
    print('='*60)
    sub = df[df.Model == model].drop(columns="Model")
    print(sub.to_string(index=False))
