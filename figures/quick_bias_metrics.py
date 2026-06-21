"""Quick summary of β, FHV, FLV, and relative MAE across bias train-only experiments.

Tests the proposed mechanisms for the bias-direction asymmetry documented in Figure 10:
  - Sensitivity inversion: β > 1 for negative-bias-trained, β < 1 for positive-bias-trained.
  - Peak overprediction:   FHV > 0 for negative-bias-trained (amplified gain hits peaks hardest).
  - Spurious baseflow:     FLV > 0 for negative-bias-trained (hypersensitive response on low-P days).
  - NSE geometry vs. real: rMAE asymmetry compared to NSE asymmetry — if rMAE is more symmetric,
                            NSE geometry (mechanism #2) is contributing to the visual asymmetry.

VIC is included as a control: a calibrated process model should show roughly symmetric ±50%
degradation if the LSTM's asymmetry is a property of its learning process rather than the design.
"""

import pandas as pd
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("outputs")
FORCING = "daymet"
# Ordered by signed magnitude so the asymmetry pattern reads top-to-bottom.
BIAS_LEVELS = [-0.5, -0.25, -0.1, 0.1, 0.25, 0.5]


def load(prefix, obs_cols):
    p = OUTPUT_DIR / f"{prefix}_valid_predictions.csv"
    if not p.exists():
        p = OUTPUT_DIR / f"{prefix}_valid_predictions.csv.gz"
    return pd.read_csv(p, index_col=0, parse_dates=True).loc[:, obs_cols]


def kge_components(obs, sim):
    r = obs.corrwith(sim)
    alpha = sim.std() / obs.std()
    beta = sim.mean() / obs.mean()
    return r, alpha, beta


def flow_volume_error(obs, sim, pct, side):
    """% error in flow volume on basin-days where obs is above (high) or below (low) pct quantile."""
    results = {}
    for basin in obs.columns:
        o = obs[basin].dropna()
        s = sim[basin].reindex(o.index)
        thresh = o.quantile(pct)
        mask = (o >= thresh) if side == "high" else (o <= thresh)
        vol_obs = o[mask].sum()
        vol_sim = s[mask].sum()
        if abs(vol_obs) > 1e-6:
            results[basin] = (vol_sim - vol_obs) / vol_obs * 100
    return pd.Series(results)


def rel_mae(obs, sim):
    """Per-basin MAE normalized by mean observed flow (relative MAE, %)."""
    results = {}
    for basin in obs.columns:
        o = obs[basin].dropna()
        s = sim[basin].reindex(o.index).dropna()
        common = o.index.intersection(s.index)
        if len(common) == 0:
            continue
        denom = o.loc[common].mean()
        if abs(denom) > 1e-6:
            results[basin] = (s.loc[common] - o.loc[common]).abs().mean() / denom * 100
    return pd.Series(results)


def summarise(obs, sim):
    r, a, b = kge_components(obs, sim)
    fhv = flow_volume_error(obs, sim, 0.98, "high")
    flv = flow_volume_error(obs, sim, 0.30, "low")
    rmae = rel_mae(obs, sim)

    def med(s, fmt=":+.3f"):
        return format(s.median(), fmt[1:])

    return {
        "r":        med(r),
        "α":        med(a),
        "β":        med(b),
        "FHV (%)":  med(fhv, ":+.1f"),
        "FLV (%)":  med(flv, ":+.1f"),
        "rMAE (%)": med(rmae, ":.1f"),
    }


rows = []

for model, prefix_base in [("LSTM", f"ealstm_{FORCING}"),
                            ("VIC",  f"vic_{FORCING}")]:
    obs = pd.read_csv(OUTPUT_DIR / f"{prefix_base}_valid_observations.csv.gz",
                      index_col=0, parse_dates=True)

    sim = load(prefix_base, obs.columns)
    rows.append({"Model": model, "Bias": "baseline", **summarise(obs, sim)})

    for signed in BIAS_LEVELS:
        stem = f"bias{signed:g}_noperturb"
        label = f"{int(signed*100):+d}%"
        try:
            sim = load(f"{prefix_base}_{stem}", obs.columns)
            rows.append({"Model": model, "Bias": label, **summarise(obs, sim)})
        except FileNotFoundError:
            rows.append({"Model": model, "Bias": label,
                         "r": "N/A", "α": "N/A", "β": "N/A",
                         "FHV (%)": "N/A", "FLV (%)": "N/A", "rMAE (%)": "N/A"})

df = pd.DataFrame(rows)
for model in ["LSTM", "VIC"]:
    print(f"\n{'='*72}")
    print(f"  {model} — bias train-only, evaluated on baseline P (Daymet)")
    print("=" * 72)
    sub = df[df.Model == model].drop(columns="Model")
    print(sub.to_string(index=False))
