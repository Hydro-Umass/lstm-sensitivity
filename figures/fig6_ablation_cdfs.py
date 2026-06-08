# Figure 6: Ablation skill CDFs
# ECDF of per-basin NSE for baseline LSTM, zero-P LSTM, precip-only LSTM,
# and VIC zero-P.
#
# Requires outputs from exp1 (baseline), exp3 (zero-P), and the precip-only
# ablation experiment (ealstm_{FORCING}_precip_only_valid_predictions.csv.gz).

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import nse

sns.set_context("paper", font_scale=2.5)

FORCING = "daymet"


def ablation_cdfs():
    obs = pd.read_csv(
        f"outputs/ealstm_{FORCING}_valid_observations.csv.gz",
        index_col=0, parse_dates=True,
    )
    lstm_base = pd.read_csv(
        f"outputs/ealstm_{FORCING}_valid_predictions.csv.gz",
        index_col=0, parse_dates=True,
    ).loc[:, obs.columns]
    lstm_zerop = pd.read_csv(
        f"outputs/ealstm_{FORCING}_zero_precip_valid_predictions.csv.gz",
        index_col=0, parse_dates=True,
    ).loc[:, obs.columns]
    lstm_precip = pd.read_csv(
        f"outputs/ealstm_{FORCING}_precip_only_valid_predictions.csv.gz",
        index_col=0, parse_dates=True,
    ).loc[:, obs.columns]

    obs_zerop = pd.read_csv(
        f"outputs/ealstm_{FORCING}_zero_precip_valid_observations.csv.gz",
        index_col=0, parse_dates=True,
    )
    vic_zerop = pd.read_csv(
        f"outputs/vic_{FORCING}_zero_precip_valid_predictions.csv.gz",
        index_col=0, parse_dates=True,
    ).loc[:, obs_zerop.columns]

    nse_base = nse(obs, lstm_base)
    nse_zerop = nse(obs_zerop, lstm_zerop)
    nse_precip = nse(obs, lstm_precip)
    nse_vic_zerop = nse(obs_zerop, vic_zerop)

    configs = [
        (nse_base,      "LSTM baseline",   "-",  "#1f77b4"),
        (nse_zerop,     "LSTM zero P",     "--", "#1f77b4"),
        (nse_precip,    "LSTM P only", ":", "#1f77b4"),
        (nse_vic_zerop, "VIC zero P",      "--", "#d62728"),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    for nse_vals, label, ls, color in configs:
        vals = np.sort(nse_vals.dropna().clip(lower=-1.0).values)
        cdf = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, cdf, linestyle=ls, color=color, linewidth=1.5, label=label)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("NSE")
    ax.set_ylabel("Fraction of basins")
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("documents/paper/figures/fig6_ablation_cdfs.png", dpi=300, bbox_inches="tight", pad_inches=0.1)


if __name__ == "__main__":
    ablation_cdfs()
