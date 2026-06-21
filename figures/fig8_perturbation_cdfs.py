# Figure 8: Random error experiments — three-way train/eval contrast
#
# 3×2 grid:
#   Rows: (a,b) train+eval    — trained AND evaluated with σ-perturbed P
#         (c,d) train-only    — trained with σ-perturbed P, evaluated on baseline P
#         (e,f) inference-only — trained on baseline P, evaluated with σ-perturbed P
#   Columns: left = LSTM, right = VIC. Daymet forcing throughout.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from utils import nse

sns.set_context("paper", font_scale=2.5)

FORCING = "daymet"
OUTPUT_DIR = Path("outputs")
STD_LEVELS = [0.1, 0.25, 0.5]


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


def plot_ecdf(ax, nse_vals, label, color, linestyle, linewidth=1.5):
    vals = np.sort(nse_vals.dropna().clip(lower=-1.0).values)
    cdf = np.arange(1, len(vals) + 1) / len(vals)
    ax.plot(vals, cdf, color=color, linestyle=linestyle,
            linewidth=linewidth, label=label)


def _pred_path(prefix, row_type, std):
    lvl = f"{std:g}"
    if row_type == "train_eval":
        stem = f"random_error_std{lvl}"
    elif row_type == "train_only":
        stem = f"random_error_std{lvl}_noperturb"
    else:  # infer_only
        stem = f"infer_random_error_std{lvl}"
    return OUTPUT_DIR / f"{prefix}_{stem}_valid_predictions.csv"


def build_panel(ax, prefix, obs, row_type, panel_label, title):
    base_pred = load_predictions(OUTPUT_DIR / f"{prefix}_valid_predictions.csv", obs.columns)
    plot_ecdf(ax, nse(obs[base_pred.columns], base_pred), "Baseline", "black", "-", linewidth=2.0)

    rand_colors = ["#9ecae1", "#4292c6", "#084594"]
    rand_styles = ["--", "--", "-"]
    for std, col, ls in zip(STD_LEVELS, rand_colors, rand_styles):
        path = _pred_path(prefix, row_type, std)
        try:
            pred = load_predictions(path, obs.columns)
            plot_ecdf(ax, nse(obs[pred.columns], pred), f"σ = {int(std*100)}%", col, ls)
        except FileNotFoundError:
            print(f"Warning: {path} not found, skipping.")

    ax.axvline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"{panel_label} {title}")


def random_error_cdfs():
    lstm_obs = pd.read_csv(OUTPUT_DIR / f"ealstm_{FORCING}_valid_observations.csv.gz",
                           index_col=0, parse_dates=True)
    vic_obs  = pd.read_csv(OUTPUT_DIR / f"vic_{FORCING}_valid_observations.csv.gz",
                           index_col=0, parse_dates=True)

    fig, axes = plt.subplots(3, 2, figsize=(18, 20), sharex=True, sharey=True)

    panels = [
        (axes[0, 0], f"ealstm_{FORCING}", lstm_obs, "train_eval",  "(a)", "LSTM - train & eval perturbed"),
        (axes[0, 1], f"vic_{FORCING}",    vic_obs,  "train_eval",  "(b)", "VIC - train & eval perturbed"),
        (axes[1, 0], f"ealstm_{FORCING}", lstm_obs, "train_only",  "(c)", "LSTM - train perturbed, eval baseline"),
        (axes[1, 1], f"vic_{FORCING}",    vic_obs,  "train_only",  "(d)", "VIC - train perturbed, eval baseline"),
        (axes[2, 0], f"ealstm_{FORCING}", lstm_obs, "infer_only",  "(e)", "LSTM - train baseline, eval perturbed"),
        (axes[2, 1], f"vic_{FORCING}",    vic_obs,  "infer_only",  "(f)", "VIC - train baseline, eval perturbed"),
    ]

    for ax, prefix, obs, row_type, label, title in panels:
        build_panel(ax, prefix, obs, row_type, label, title)

    for ax in axes[:, 0]:
        ax.set_ylabel("Fraction of basins")
    for ax in axes[2, :]:
        ax.set_xlabel("NSE")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, 0.01))

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out = Path("documents/paper/figures/fig8_random_error_cdfs.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved {out}")


if __name__ == "__main__":
    random_error_cdfs()
