# Figure S2: Ablation skill CDFs across all three forcings (SI companion to Fig 6)
# ECDF of per-basin skill for baseline LSTM, zero-P LSTM, precip-only LSTM,
# and VIC zero-P. Top row = NSE, bottom row = KGE; columns = Daymet, NLDAS, Maurer.
#
# Requires outputs from exp1 (baseline), exp3 (zero-P), and the precip-only
# ablation experiment (ealstm_{FORCING}_precip_only_valid_predictions.csv.gz),
# for each of the three forcings.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import nse, kge

sns.set_context("paper", font_scale=2.5)

FORCINGS = ["daymet", "nldas", "maurer"]


def forcing_label(forcing):
    return forcing.upper() if forcing == "nldas" else forcing.capitalize()


def ablation_skill(forcing, metric):
    obs = pd.read_csv(
        f"outputs/ealstm_{forcing}_valid_observations.csv.gz",
        index_col=0, parse_dates=True,
    )
    lstm_base = pd.read_csv(
        f"outputs/ealstm_{forcing}_valid_predictions.csv.gz",
        index_col=0, parse_dates=True,
    ).loc[:, obs.columns]
    lstm_zerop = pd.read_csv(
        f"outputs/ealstm_{forcing}_zero_precip_valid_predictions.csv.gz",
        index_col=0, parse_dates=True,
    ).loc[:, obs.columns]
    lstm_precip = pd.read_csv(
        f"outputs/ealstm_{forcing}_precip_only_valid_predictions.csv.gz",
        index_col=0, parse_dates=True,
    ).loc[:, obs.columns]

    obs_zerop = pd.read_csv(
        f"outputs/ealstm_{forcing}_zero_precip_valid_observations.csv.gz",
        index_col=0, parse_dates=True,
    )
    vic_zerop = pd.read_csv(
        f"outputs/vic_{forcing}_zero_precip_valid_predictions.csv.gz",
        index_col=0, parse_dates=True,
    ).loc[:, obs_zerop.columns]

    return [
        (metric(obs, lstm_base),         "LSTM baseline", "-",  "#1f77b4"),
        (metric(obs, lstm_zerop),        "LSTM zero P",   "--", "#1f77b4"),
        (metric(obs, lstm_precip),       "LSTM P only",   ":",  "#1f77b4"),
        (metric(obs_zerop, vic_zerop),   "VIC zero P",    "--", "#d62728"),
    ]


def plot_ecdf(ax, skill_vals, label, ls, color):
    vals = np.sort(skill_vals.dropna().clip(lower=-1.0).values)
    cdf = np.arange(1, len(vals) + 1) / len(vals)
    ax.plot(vals, cdf, linestyle=ls, color=color, linewidth=1.5, label=label)


def ablation_cdfs():
    metrics = [("NSE", nse), ("KGE", kge)]
    fig, axes = plt.subplots(2, 3, figsize=(24, 14), sharex=True, sharey=True)

    for j, forcing in enumerate(FORCINGS):
        for i, (metric_name, metric_fun) in enumerate(metrics):
            ax = axes[i, j]
            for skill_vals, label, ls, color in ablation_skill(forcing, metric_fun):
                plot_ecdf(ax, skill_vals, label, ls, color)
            ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
            ax.set_xlim(-1, 1)
            ax.set_ylim(0, 1)
            if i == 0:
                ax.set_title(forcing_label(forcing))
            ax.set_xlabel(metric_name)
            if j == 0:
                ax.set_ylabel("Fraction of basins")

    axes[0, 0].legend(frameon=False, fontsize="small")
    fig.tight_layout()
    fig.savefig(
        "documents/paper/figures/figS2_ablation_cdfs.png",
        dpi=300, bbox_inches="tight", pad_inches=0.1,
    )


if __name__ == "__main__":
    ablation_cdfs()
