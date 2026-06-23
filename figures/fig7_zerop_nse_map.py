# Figure 7: Spatial pattern of zero-P LSTM NSE across CAMELS
# Tests whether seasonal-cycle recovery concentrates in snowmelt-dominated basins.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
from utils import nse

sns.set_context("paper", font_scale=2.5)

FORCING = "daymet"


def zerop_nse_map():
    basins = pd.read_csv(
        "data/camels_topo.txt", dtype={"gauge_id": str}, sep=";", index_col="gauge_id"
    )

    obs_zerop = pd.read_csv(
        f"outputs/ealstm_{FORCING}_zero_precip_valid_observations.csv.gz",
        index_col=0, parse_dates=True,
    )
    lstm_zerop = pd.read_csv(
        f"outputs/ealstm_{FORCING}_zero_precip_valid_predictions.csv.gz",
        index_col=0, parse_dates=True,
    ).loc[:, obs_zerop.columns]

    nse_zerop = nse(obs_zerop, lstm_zerop)

    common = sorted(set(nse_zerop.index) & set(basins.index))
    plot_df = pd.DataFrame({
        "lat": basins.loc[common, "gauge_lat"],
        "lon": basins.loc[common, "gauge_lon"],
        "nse": nse_zerop.reindex(common).clip(lower=-1.0),
    }).dropna()
    plot_df = plot_df.sort_values("nse")  # draw low-skill points first

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(
        1, 1, 1,
        projection=ccrs.AlbersEqualArea(central_latitude=39.5, central_longitude=-98.35),
    )

    states = cfeature.NaturalEarthFeature(
        category="cultural", name="admin_1_states_provinces_lakes",
        scale="50m", facecolor="none", edgecolor="0.5", linewidth=0.4,
    )
    ax.add_feature(states)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4, linestyle="--")
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f0f0f0", edgecolor="none", zorder=0)
    ax.set_extent([-125, -66, 24, 50], crs=ccrs.PlateCarree())

    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    sc = ax.scatter(
        plot_df["lon"].values,
        plot_df["lat"].values,
        c=plot_df["nse"].values,
        cmap="RdYlGn",
        norm=norm,
        s=22,
        edgecolors="none",
        alpha=0.85,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02, aspect=30, extend='min')
    cb.set_label("NSE")

    fig.savefig("documents/paper/figures/fig7_zerop_nse_map.png", dpi=300, bbox_inches="tight", pad_inches=0.1)


if __name__ == "__main__":
    zerop_nse_map()
