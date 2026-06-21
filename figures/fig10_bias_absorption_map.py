# Figure 10: Spatial pattern of bias absorption
#
# Two-panel CONUS map: ΔNSE = NSE(bias_train-only) − NSE(baseline) per basin
# for LSTM trained on −50% biased P (left) and +50% biased P (right), both
# evaluated on unperturbed baseline. Diverging colormap: green = absorption
# succeeded (ΔNSE ≈ 0); red = absorption failed. LSTM / Daymet only.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
from pathlib import Path
from utils import nse

sns.set_context("paper", font_scale=2.5)

FORCING = "daymet"
OUTPUT_DIR = Path("outputs")
DATA_DIR = Path("data")
PROJ = ccrs.AlbersEqualArea(central_latitude=39.5, central_longitude=-98.35)


def load_predictions(path, obs_cols):
    p = Path(path)
    if not p.exists():
        gz = Path(str(path) + ".gz")
        if gz.exists():
            p = gz
        else:
            raise FileNotFoundError(f"Cannot find {path} or {path}.gz")
    return pd.read_csv(p, index_col=0, parse_dates=True).loc[:, obs_cols]


def add_map_features(ax):
    states = cfeature.NaturalEarthFeature(
        category="cultural", name="admin_1_states_provinces_lakes",
        scale="50m", facecolor="none", edgecolor="0.5", linewidth=0.4,
    )
    ax.add_feature(states)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4, linestyle="--")
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f0f0f0", edgecolor="none", zorder=0)
    ax.set_extent([-125, -66, 24, 50], crs=ccrs.PlateCarree())


def bias_absorption_map():
    basins = pd.read_csv(
        DATA_DIR / "camels_topo.txt", dtype={"gauge_id": str}, sep=";", index_col="gauge_id"
    )

    prefix = f"ealstm_{FORCING}"
    obs = pd.read_csv(
        OUTPUT_DIR / f"{prefix}_valid_observations.csv.gz",
        index_col=0, parse_dates=True,
    )

    nse_base = nse(obs, load_predictions(OUTPUT_DIR / f"{prefix}_valid_predictions.csv", obs.columns)).clip(lower=-1.0)

    # compute ΔNSE for both bias levels; clip per-basin NSE to [-1, 1] before differencing
    # to prevent extreme outliers from collapsing the color scale.
    bias_conditions = [(-0.5, "(a)", "Bias −50%"), (0.5, "(b)", "Bias +50%")]
    dnse_series = {}
    for signed_bias, _, _ in bias_conditions:
        lvl = f"{signed_bias:g}"
        try:
            pred = load_predictions(
                OUTPUT_DIR / f"{prefix}_bias{lvl}_noperturb_valid_predictions.csv", obs.columns
            )
            nse_bias = nse(obs, pred).clip(lower=-1.0)
            dnse_series[signed_bias] = nse_bias - nse_base
        except FileNotFoundError:
            print(f"Warning: bias{lvl} noperturb predictions not found.")

    if not dnse_series:
        raise RuntimeError("No bias absorption data found.")

    pooled = pd.concat(dnse_series.values()).dropna()
    vmin = min(float(np.percentile(pooled, 5)), -0.05)
    vmax = max(float(np.percentile(pooled, 95)), 0.05)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), subplot_kw={"projection": PROJ})

    sc = None
    for ax, (signed_bias, panel_label, title) in zip(axes, bias_conditions):
        add_map_features(ax)
        ax.set_title(f"{panel_label} {title}")

        if signed_bias not in dnse_series:
            continue

        dnse = dnse_series[signed_bias].clip(lower=vmin, upper=vmax)
        common = sorted(set(dnse.index) & set(basins.index))
        plot_df = pd.DataFrame({
            "lat": basins.loc[common, "gauge_lat"],
            "lon": basins.loc[common, "gauge_lon"],
            "dnse": dnse.reindex(common),
        }).dropna().sort_values("dnse", ascending=False)  # draw failures (most negative) last = on top

        sc = ax.scatter(
            plot_df["lon"].values,
            plot_df["lat"].values,
            c=plot_df["dnse"].values,
            cmap="RdYlGn",
            norm=norm,
            s=22,
            edgecolors="none",
            alpha=0.85,
            transform=ccrs.PlateCarree(),
            zorder=5,
        )

    if sc is not None:
        cb = fig.colorbar(sc, ax=axes.tolist(), shrink=0.65, pad=0.02, aspect=30,
                          orientation="horizontal", location="bottom")
        cb.set_label("ΔNSE (bias train-only − baseline)", labelpad=4)
        cb.ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=6, prune="both"))

    out = Path("documents/paper/figures/fig10_bias_absorption_map.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved {out}")


if __name__ == "__main__":
    bias_absorption_map()
