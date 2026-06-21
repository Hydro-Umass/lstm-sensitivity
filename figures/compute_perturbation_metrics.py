"""Compute per-basin median statistics for perturbation experiments.

Outputs two LaTeX tables:
  Table S2 — random error (NSE, KGE, r, α, β, FHV) for LSTM and VIC
  Table S3 — systematic bias (NSE, KGE, r, α, β, FHV, FLV, rMAE) for LSTM and VIC

Run from ~/Projects/lstm:
  .venv/bin/python lstm-sensitivity/figures/compute_perturbation_metrics.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("outputs")
FORCING = "daymet"


def load_pred(prefix, obs_cols):
    for ext in ["", ".gz"]:
        p = OUTPUT_DIR / f"{prefix}_valid_predictions.csv{ext}"
        if p.exists():
            df = pd.read_csv(p, index_col=0, parse_dates=True)
            common = obs_cols.intersection(df.columns)
            return df.loc[:, common]
    raise FileNotFoundError(f"No predictions found for {prefix}")


def per_basin_nse(obs, sim):
    results = {}
    for basin in obs.columns:
        o = obs[basin].dropna()
        s = sim[basin].reindex(o.index).dropna()
        idx = o.index.intersection(s.index)
        if not len(idx):
            continue
        o, s = o.loc[idx], s.loc[idx]
        denom = ((o - o.mean()) ** 2).sum()
        if denom > 1e-6:
            results[basin] = 1 - ((o - s) ** 2).sum() / denom
    return pd.Series(results)


def kge_components(obs, sim):
    r = obs.corrwith(sim)
    alpha = sim.std() / obs.std()
    beta = sim.mean() / obs.mean()
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return r, alpha, beta, kge


def flow_volume_error(obs, sim, pct, side):
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
    results = {}
    for basin in obs.columns:
        o = obs[basin].dropna()
        s = sim[basin].reindex(o.index).dropna()
        idx = o.index.intersection(s.index)
        if not len(idx):
            continue
        denom = o.loc[idx].mean()
        if abs(denom) > 1e-6:
            results[basin] = (s.loc[idx] - o.loc[idx]).abs().mean() / denom * 100
    return pd.Series(results)


def align(obs, sim):
    common = obs.columns.intersection(sim.columns)
    return obs[common], sim[common]


def stats_random(obs, sim):
    obs, sim = align(obs, sim)
    nse_vals = per_basin_nse(obs, sim)
    r, a, b, k = kge_components(obs, sim)
    fhv = flow_volume_error(obs, sim, 0.98, "high")
    return dict(
        NSE=nse_vals.median(),
        KGE=k.median(),
        r=r.median(),
        alpha=a.median(),
        beta=b.median(),
        FHV=fhv.median(),
    )


def stats_bias(obs, sim):
    obs, sim = align(obs, sim)
    d = stats_random(obs, sim)
    d["FLV"] = flow_volume_error(obs, sim, 0.30, "low").median()
    d["rMAE"] = rel_mae(obs, sim).median()
    return d


# ---------------------------------------------------------------------------
# Collect random error rows
# ---------------------------------------------------------------------------
STD_LEVELS = [0.1, 0.25, 0.5]

rand_rows = []
for model, prefix_base in [("LSTM", f"ealstm_{FORCING}"), ("VIC", f"vic_{FORCING}")]:
    obs = pd.read_csv(
        OUTPUT_DIR / f"{prefix_base}_valid_observations.csv.gz",
        index_col=0, parse_dates=True,
    )
    rand_rows.append(
        dict(Model=model, Condition="Baseline", sigma="---",
             **stats_random(obs, load_pred(prefix_base, obs.columns)))
    )
    for std in STD_LEVELS:
        lvl = f"{std:g}"
        for cond_label, stem in [
            ("Train \\& eval", f"random_error_std{lvl}"),
            ("Train only",     f"random_error_std{lvl}_noperturb"),
            ("Infer only",     f"infer_random_error_std{lvl}"),
        ]:
            try:
                sim = load_pred(f"{prefix_base}_{stem}", obs.columns)
                rand_rows.append(
                    dict(Model=model, Condition=cond_label,
                         sigma=f"{int(std*100)}\\%", **stats_random(obs, sim))
                )
            except FileNotFoundError:
                rand_rows.append(
                    dict(Model=model, Condition=cond_label,
                         sigma=f"{int(std*100)}\\%",
                         NSE=None, KGE=None, r=None, alpha=None, beta=None, FHV=None)
                )

rand_df = pd.DataFrame(rand_rows)

# ---------------------------------------------------------------------------
# Collect bias rows
# ---------------------------------------------------------------------------
BIAS_LEVELS = [-0.5, -0.25, -0.1, 0.1, 0.25, 0.5]

bias_rows = []
for model, prefix_base in [("LSTM", f"ealstm_{FORCING}"), ("VIC", f"vic_{FORCING}")]:
    obs = pd.read_csv(
        OUTPUT_DIR / f"{prefix_base}_valid_observations.csv.gz",
        index_col=0, parse_dates=True,
    )
    bias_rows.append(
        dict(Model=model, Condition="Baseline", bias="---",
             **stats_bias(obs, load_pred(prefix_base, obs.columns)))
    )
    for b in BIAS_LEVELS:
        sign = f"{int(b*100):+d}\\%"
        for cond_label, stem in [
            ("Train \\& eval", f"bias{b:g}"),
            ("Train only",     f"bias{b:g}_noperturb"),
            ("Infer only",     f"infer_bias{b:g}"),
        ]:
            try:
                sim = load_pred(f"{prefix_base}_{stem}", obs.columns)
                bias_rows.append(
                    dict(Model=model, Condition=cond_label,
                         bias=sign, **stats_bias(obs, sim))
                )
            except FileNotFoundError:
                bias_rows.append(
                    dict(Model=model, Condition=cond_label,
                         bias=sign,
                         NSE=None, KGE=None, r=None, alpha=None,
                         beta=None, FHV=None, FLV=None, rMAE=None)
                )

bias_df = pd.DataFrame(bias_rows)


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------

def fmt(v, decimals=2, signed=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "---"
    fmt_str = f"{{:+.{decimals}f}}" if signed else f"{{:.{decimals}f}}"
    return fmt_str.format(v)


def latex_row_rand(row, is_lstm_baseline=False):
    vals = " & ".join([
        fmt(row["NSE"]),
        fmt(row["KGE"]),
        fmt(row["r"], 3),
        fmt(row["alpha"], 3),
        fmt(row["beta"], 3),
        fmt(row["FHV"], 1, signed=True),
    ])
    return vals


def print_table_s2(df):
    lstm = df[df.Model == "LSTM"].reset_index(drop=True)
    vic  = df[df.Model == "VIC"].reset_index(drop=True)
    assert len(lstm) == len(vic)

    lines = []
    lines.append(r"\begin{table}")
    lines.append(r"  \settablenum{S2}")
    lines.append(r"  \caption{Per-basin median statistics for random error perturbation experiments")
    lines.append(r"    (Daymet forcing, validation period 2000--2008). Condition indicates whether")
    lines.append(r"    the perturbation was applied during training, evaluation, or both.")
    lines.append(r"    Metrics: NSE = Nash--Sutcliffe efficiency; KGE = Kling--Gupta efficiency;")
    lines.append(r"    $r$ = Pearson correlation; $\alpha$ = variability ratio ($\sigma_\mathrm{sim}/\sigma_\mathrm{obs}$);")
    lines.append(r"    $\beta$ = bias ratio ($\mu_\mathrm{sim}/\mu_\mathrm{obs}$);")
    lines.append(r"    FHV = high-flow volume error (\% error in total flow above the 98th-percentile threshold).}")
    lines.append(r"  \label{tab:s2}")
    lines.append(r"  \footnotesize\centering")
    lines.append(r"  \begin{tabular}{llrrrrrrrrrrrr}")
    lines.append(r"    \hline")
    lines.append(r"    & & \multicolumn{6}{c}{LSTM} & \multicolumn{6}{c}{VIC} \\")
    lines.append(r"    \cline{3-8}\cline{9-14}")
    lines.append(r"    Condition & $\sigma$ & NSE & KGE & $r$ & $\alpha$ & $\beta$ & FHV(\%) & NSE & KGE & $r$ & $\alpha$ & $\beta$ & FHV(\%) \\")
    lines.append(r"    \hline")

    prev_cond = None
    for i in range(len(lstm)):
        lr, vr = lstm.iloc[i], vic.iloc[i]
        cond = lr["Condition"]
        sig = lr["sigma"]
        if cond == "Baseline":
            lines.append(
                f"    {cond} & {sig} & {latex_row_rand(lr)} & {latex_row_rand(vr)} \\\\"
            )
            lines.append(r"    \hline")
        else:
            if cond != prev_cond and prev_cond != "Baseline":
                lines.append(r"    \hline")
            lines.append(
                f"    {cond} & {sig} & {latex_row_rand(lr)} & {latex_row_rand(vr)} \\\\"
            )
        prev_cond = cond

    lines.append(r"    \hline")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def latex_row_bias(row):
    vals = " & ".join([
        fmt(row["NSE"]),
        fmt(row["KGE"]),
        fmt(row["r"], 3),
        fmt(row["alpha"], 3),
        fmt(row["beta"], 3),
        fmt(row["FHV"], 1, signed=True),
        fmt(row["FLV"], 1, signed=True),
        fmt(row["rMAE"], 1),
    ])
    return vals


def print_table_s3(df):
    lstm = df[df.Model == "LSTM"].reset_index(drop=True)
    vic  = df[df.Model == "VIC"].reset_index(drop=True)
    assert len(lstm) == len(vic)

    lines = []
    lines.append(r"\begin{table}")
    lines.append(r"  \settablenum{S3}")
    lines.append(r"  \caption{Per-basin median statistics for systematic bias perturbation experiments")
    lines.append(r"    (Daymet forcing, validation period 2000--2008). Bias levels are stated as signed")
    lines.append(r"    percentage changes relative to the baseline precipitation (e.g., $-50\%$ corresponds")
    lines.append(r"    to $P \times 0.5$; $+50\%$ to $P \times 1.5$). Condition indicates whether the")
    lines.append(r"    perturbation was applied during training, evaluation, or both.")
    lines.append(r"    Metrics: NSE, KGE, $r$, $\alpha$, $\beta$, FHV as in Table~\ref{tab:s2};")
    lines.append(r"    FLV = low-flow volume error (\% error in total flow below the 30th-percentile threshold);")
    lines.append(r"    rMAE = relative mean absolute error (MAE normalized by mean observed flow, \%).}")
    lines.append(r"  \label{tab:s3}")
    lines.append(r"  \footnotesize\centering")
    lines.append(r"  \begin{tabular}{llrrrrrrrrrrrrrrrr}")
    lines.append(r"    \hline")
    lines.append(r"    & & \multicolumn{8}{c}{LSTM} & \multicolumn{8}{c}{VIC} \\")
    lines.append(r"    \cline{3-10}\cline{11-18}")
    lines.append(r"    Condition & Bias & NSE & KGE & $r$ & $\alpha$ & $\beta$ & FHV(\%) & FLV(\%) & rMAE(\%) &"
                 r" NSE & KGE & $r$ & $\alpha$ & $\beta$ & FHV(\%) & FLV(\%) & rMAE(\%) \\")
    lines.append(r"    \hline")

    prev_cond = None
    for i in range(len(lstm)):
        lr, vr = lstm.iloc[i], vic.iloc[i]
        cond = lr["Condition"]
        bval = lr["bias"]
        if cond == "Baseline":
            lines.append(
                f"    {cond} & {bval} & {latex_row_bias(lr)} & {latex_row_bias(vr)} \\\\"
            )
            lines.append(r"    \hline")
        else:
            if cond != prev_cond and prev_cond != "Baseline":
                lines.append(r"    \hline")
            lines.append(
                f"    {cond} & {bval} & {latex_row_bias(lr)} & {latex_row_bias(vr)} \\\\"
            )
        prev_cond = cond

    lines.append(r"    \hline")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(print_table_s2(rand_df))
    print()
    print(print_table_s3(bias_df))
