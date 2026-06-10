# Figure 9: Systematic bias experiments — three-way train/eval contrast
#
# 3×2 grid:
#   Rows: (a,b) train+eval    — trained AND evaluated with biased P
#         (c,d) train-only    — trained with biased P, evaluated on baseline P
#         (e,f) inference-only — trained on baseline P, evaluated with biased P
#   Columns: left = LSTM, right = VIC. Daymet forcing throughout.
#
# Bias file naming convention: signed fractions relative to baseline
#   bias-0.5 = P×0.5 (−50%),  bias0.5 = P×1.5 (+50%)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from pathlib import Path
from utils import nse

sns.set_context("paper", font_scale=2.5)

FORCING = "daymet"
OUTPUT_DIR = Path("outputs")
BIAS_LEVELS = [0.1, 0.25, 0.5]


def load_predictions(path, obs_cols):
    p = Path(path)
    if not p.exists():
        gz = Path(str(path) + ".gz")
        if gz.exists():
            p = gz
        else:
            raise FileNotFoundError(f"Cannot find {path} or {path}.gz")
    return pd.read_csv(p, index_col=0, parse_dates=True).loc[:, obs_cols]


def plot_ecdf(ax, nse_vals, label, color, linestyle, linewidth=1.5):
    vals = np.sort(nse_vals.dropna().clip(lower=-1.0).values)
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


def build_panel(ax, prefix, obs, row_type, panel_label, title):
    base_pred = load_predictions(OUTPUT_DIR / f"{prefix}_valid_predictions.csv", obs.columns)
    plot_ecdf(ax, nse(obs, base_pred), "Baseline", "black", "-", linewidth=2.0)

    # Negative bias — green family, light → dark
    neg_colors = ["#a1d99b", "#41ab5d", "#006d2c"]
    neg_styles = ["--", "--", "-"]
    for lvl, col, ls in zip(BIAS_LEVELS, neg_colors, neg_styles):
        path = _pred_path(prefix, row_type, -lvl)
        try:
            pred = load_predictions(path, obs.columns)
            plot_ecdf(ax, nse(obs, pred), f"Bias −{int(lvl*100)}%", col, ls)
        except FileNotFoundError:
            print(f"Warning: {path} not found, skipping.")

    # Positive bias — orange-red family, light → dark
    pos_colors = ["#fdae6b", "#f16913", "#a63603"]
    pos_styles = ["--", "--", "-"]
    for lvl, col, ls in zip(BIAS_LEVELS, pos_colors, pos_styles):
        path = _pred_path(prefix, row_type, lvl)
        try:
            pred = load_predictions(path, obs.columns)
            plot_ecdf(ax, nse(obs, pred), f"Bias +{int(lvl*100)}%", col, ls)
        except FileNotFoundError:
            print(f"Warning: {path} not found, skipping.")

    ax.axvline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"{panel_label} {title}")


def bias_cdfs():
    lstm_obs = pd.read_csv(OUTPUT_DIR / f"ealstm_{FORCING}_valid_observations.csv.gz",
                           index_col=0, parse_dates=True)
    vic_obs  = pd.read_csv(OUTPUT_DIR / f"vic_{FORCING}_valid_observations.csv.gz",
                           index_col=0, parse_dates=True)

    fig, axes = plt.subplots(2, 3, figsize=(27, 12), sharex=True, sharey=True)

    panels = [
        (axes[0, 0], f"ealstm_{FORCING}", lstm_obs, "train_eval",  "(a)", "LSTM - train & eval perturbed"),
        (axes[0, 1], f"ealstm_{FORCING}", lstm_obs, "train_only",  "(b)", "LSTM - train perturbed, eval baseline"),
        (axes[0, 2], f"ealstm_{FORCING}", lstm_obs, "infer_only",  "(c)", "LSTM - train baseline, eval perturbed"),
        (axes[1, 0], f"vic_{FORCING}",    vic_obs,  "train_eval",  "(d)", "VIC - train & eval perturbed"),
        (axes[1, 1], f"vic_{FORCING}",    vic_obs,  "train_only",  "(e)", "VIC - train perturbed, eval baseline"),
        (axes[1, 2], f"vic_{FORCING}",    vic_obs,  "infer_only",  "(f)", "VIC - train baseline, eval perturbed"),
    ]

    for ax, prefix, obs, row_type, label, title in panels:
        build_panel(ax, prefix, obs, row_type, label, title)

    for ax in axes[:, 0]:
        ax.set_ylabel("Fraction of basins")
    for ax in axes[1, :]:
        ax.set_xlabel("NSE")

    neg_colors = ["#a1d99b", "#41ab5d", "#006d2c"]
    pos_colors = ["#fdae6b", "#f16913", "#a63603"]
    legend_handles = (
        [mlines.Line2D([], [], color="black", linestyle="-", linewidth=2, label="Baseline")] +
        [mlines.Line2D([], [], color=c, linestyle=ls, label=f"Bias −{int(lvl*100)}%")
         for lvl, c, ls in zip(BIAS_LEVELS, neg_colors, ["--", "--", "-"])] +
        [mlines.Line2D([], [], color=c, linestyle=ls, label=f"Bias +{int(lvl*100)}%")
         for lvl, c, ls in zip(BIAS_LEVELS, pos_colors, ["--", "--", "-"])]
    )
    fig.legend(handles=legend_handles, loc="lower center", ncol=7,
               frameon=False, bbox_to_anchor=(0.5, 0.01))

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out = Path("documents/paper/figures/fig9_bias_cdfs.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved {out}")


if __name__ == "__main__":
    bias_cdfs()
