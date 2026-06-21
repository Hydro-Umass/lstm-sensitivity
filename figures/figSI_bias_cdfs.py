# Figures S8-S12: Systematic-bias CDFs for the SI (companions to Fig 9)
#
# One 3x2 figure per (forcing, metric):
#   Rows: (a,b) train+eval, (c,d) train-only, (e,f) inference-only
#   Columns: left = LSTM, right = VIC
# Metric in {NSE, KGE}. Mirrors fig9_bias_cdfs.py (Daymet, NSE, main text).
#
# Bias file naming: signed fractions relative to baseline
#   bias-0.5 = P x 0.5 (-50%),  bias0.5 = P x 1.5 (+50%)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from pathlib import Path
from utils import nse, kge

sns.set_context("paper", font_scale=2.5)

OUTPUT_DIR = Path("outputs")
BIAS_LEVELS = [0.1, 0.25, 0.5]
METRICS = {"nse": nse, "kge": kge}

# (forcing, metric, output filename stem)
FIGURES = [
    ("maurer", "nse", "figS8_bias_nse_maurer"),
    ("nldas",  "nse", "figS9_bias_nse_nldas"),
    ("daymet", "kge", "figS10_bias_kge_daymet"),
    ("maurer", "kge", "figS11_bias_kge_maurer"),
    ("nldas",  "kge", "figS12_bias_kge_nldas"),
]

NEG_COLORS = ["#a1d99b", "#41ab5d", "#006d2c"]
POS_COLORS = ["#fdae6b", "#f16913", "#a63603"]
FAM_STYLES = ["--", "--", "-"]


def load_predictions(path, obs_cols):
    p = Path(path)
    if not p.exists():
        gz = Path(str(path) + ".gz")
        if gz.exists():
            p = gz
        else:
            raise FileNotFoundError(f"Cannot find {path} or {path}.gz")
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    common = [c for c in obs_cols if c in df.columns]
    return df.loc[:, common]


def plot_ecdf(ax, vals_series, label, color, linestyle, linewidth=1.5):
    vals = np.sort(vals_series.dropna().clip(lower=-1.0).values)
    cdf = np.arange(1, len(vals) + 1) / len(vals)
    ax.plot(vals, cdf, color=color, linestyle=linestyle,
            linewidth=linewidth, label=label)


def _pred_path(prefix, row_type, signed_bias):
    lvl = f"{signed_bias:g}"
    if row_type == "train_eval":
        stem = f"bias{lvl}"
    elif row_type == "train_only":
        stem = f"bias{lvl}_noperturb"
    else:  # infer_only
        stem = f"infer_bias{lvl}"
    return OUTPUT_DIR / f"{prefix}_{stem}_valid_predictions.csv"


def build_panel(ax, prefix, obs, row_type, panel_label, title, metric_fn):
    base_pred = load_predictions(OUTPUT_DIR / f"{prefix}_valid_predictions.csv", obs.columns)
    plot_ecdf(ax, metric_fn(obs[base_pred.columns], base_pred), "Baseline", "black", "-", linewidth=2.0)

    for lvl, col, ls in zip(BIAS_LEVELS, NEG_COLORS, FAM_STYLES):
        path = _pred_path(prefix, row_type, -lvl)
        try:
            pred = load_predictions(path, obs.columns)
            plot_ecdf(ax, metric_fn(obs[pred.columns], pred), f"Bias −{int(lvl*100)}%", col, ls)
        except FileNotFoundError:
            print(f"Warning: {path} not found, skipping.")

    for lvl, col, ls in zip(BIAS_LEVELS, POS_COLORS, FAM_STYLES):
        path = _pred_path(prefix, row_type, lvl)
        try:
            pred = load_predictions(path, obs.columns)
            plot_ecdf(ax, metric_fn(obs[pred.columns], pred), f"Bias +{int(lvl*100)}%", col, ls)
        except FileNotFoundError:
            print(f"Warning: {path} not found, skipping.")

    ax.axvline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"{panel_label} {title}")


def make_figure(forcing, metric_key, out_stem):
    metric_fn = METRICS[metric_key]
    metric_name = metric_key.upper()
    lstm_obs = pd.read_csv(OUTPUT_DIR / f"ealstm_{forcing}_valid_observations.csv.gz",
                           index_col=0, parse_dates=True)
    vic_obs  = pd.read_csv(OUTPUT_DIR / f"vic_{forcing}_valid_observations.csv.gz",
                           index_col=0, parse_dates=True)

    fig, axes = plt.subplots(3, 2, figsize=(18, 20), sharex=True, sharey=True)

    panels = [
        (axes[0, 0], f"ealstm_{forcing}", lstm_obs, "train_eval",  "(a)", "LSTM - train & eval perturbed"),
        (axes[0, 1], f"vic_{forcing}",    vic_obs,  "train_eval",  "(b)", "VIC - train & eval perturbed"),
        (axes[1, 0], f"ealstm_{forcing}", lstm_obs, "train_only",  "(c)", "LSTM - train perturbed, eval baseline"),
        (axes[1, 1], f"vic_{forcing}",    vic_obs,  "train_only",  "(d)", "VIC - train perturbed, eval baseline"),
        (axes[2, 0], f"ealstm_{forcing}", lstm_obs, "infer_only",  "(e)", "LSTM - train baseline, eval perturbed"),
        (axes[2, 1], f"vic_{forcing}",    vic_obs,  "infer_only",  "(f)", "VIC - train baseline, eval perturbed"),
    ]

    for ax, prefix, obs, row_type, label, title in panels:
        build_panel(ax, prefix, obs, row_type, label, title, metric_fn)

    for ax in axes[:, 0]:
        ax.set_ylabel("Fraction of basins")
    for ax in axes[2, :]:
        ax.set_xlabel(metric_name)

    legend_handles = (
        [mlines.Line2D([], [], color="black", linestyle="-", linewidth=2, label="Baseline")] +
        [mlines.Line2D([], [], color=c, linestyle=ls, label=f"Bias −{int(lvl*100)}%")
         for lvl, c, ls in zip(BIAS_LEVELS, NEG_COLORS, FAM_STYLES)] +
        [mlines.Line2D([], [], color=c, linestyle=ls, label=f"Bias +{int(lvl*100)}%")
         for lvl, c, ls in zip(BIAS_LEVELS, POS_COLORS, FAM_STYLES)]
    )
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, 0.01))

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out = Path(f"documents/paper/figures/{out_stem}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    for forcing, metric_key, out_stem in FIGURES:
        make_figure(forcing, metric_key, out_stem)
