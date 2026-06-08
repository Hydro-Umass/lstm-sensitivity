# Figure 5: Example hydrograph under zero-precipitation forcing
# Single basin, three traces: observed / LSTM (zero-P) / VIC (zero-P).
# Pass a basin ID as the first command-line argument; defaults to DEFAULT_BASIN.
# Preferred selection: a snowmelt-influenced basin where the LSTM seasonal cycle
# is clearly visible and VIC collapse is unambiguous.  Update DEFAULT_BASIN after
# inspecting the Figure 6b spatial map.

import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("paper", font_scale=2.5)

FORCING = "daymet"
DEFAULT_BASIN = "11266500"  # Kings River, CA — snowmelt-influenced; update if needed


def zerop_hydrograph(basin_id=DEFAULT_BASIN):
    basins = pd.read_csv(
        "data/camels_topo.txt", dtype={"gauge_id": str}, sep=";", index_col="gauge_id"
    )
    factor = basins.loc[basin_id, "area_geospa_fabric"] * 1e6 / 86400  # mm/day → m³/s

    obs = pd.read_csv(
        f"outputs/ealstm_{FORCING}_zero_precip_valid_observations.csv.gz",
        index_col=0, parse_dates=True,
    )
    lstm = pd.read_csv(
        f"outputs/ealstm_{FORCING}_zero_precip_valid_predictions.csv.gz",
        index_col=0, parse_dates=True,
    )
    vic = pd.read_csv(
        f"outputs/vic_{FORCING}_zero_precip_valid_predictions.csv.gz",
        index_col=0, parse_dates=True,
    )

    o = obs[basin_id] * factor
    l = lstm[basin_id] * factor
    v = vic[basin_id] * factor

    fig, ax = plt.subplots(1, 1, figsize=(16, 5))
    ax.plot(o.index, o.values, color="black", linewidth=1.5, label="Observed")
    ax.plot(l.index, l.values, color="#1f77b4", linewidth=1.2, linestyle="--", label="LSTM")
    ax.plot(v.index, v.values, color="#d62728", linewidth=1.2, linestyle=":", label="VIC")
    ax.set_ylabel("Discharge (m³/s)")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig("documents/paper/figures/fig5_hydrograph_zerop.png", dpi=300, bbox_inches="tight", pad_inches=0.1)


if __name__ == "__main__":
    basin_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASIN
    zerop_hydrograph(basin_id)
