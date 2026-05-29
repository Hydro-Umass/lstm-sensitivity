# Figure 4: Spatial pattern of cross-forcing reproducibility discrepancies

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns

sns.set_context("paper", font_scale=2.5)

basins = pd.read_csv(
    "data/camels_topo.txt", dtype={"gauge_id": str}, sep=";", index_col="gauge_id"
)
obs = pd.read_csv(
    "outputs/ealstm_traindaymet_evalnldas_valid_observations.csv.gz",
    index_col=0, parse_dates=True,
)
lstm_daymet = pd.read_csv(
    "outputs/ealstm_traindaymet_evalnldas_valid_predictions.csv.gz",
    index_col=0, parse_dates=True,
)
lstm_maurer = pd.read_csv(
    "outputs/ealstm_trainmaurer_evalnldas_valid_predictions.csv.gz",
    index_col=0, parse_dates=True,
)
vic_daymet = pd.read_csv(
    "outputs/vic_traindaymet_evalnldas_valid_predictions.csv.gz",
    index_col=0, parse_dates=True,
)
vic_maurer = pd.read_csv(
    "outputs/vic_trainmaurer_evalnldas_valid_predictions.csv.gz",
    index_col=0, parse_dates=True,
)

area = basins.T.loc["area_geospa_fabric", :]
factor = area * 1e6 / 86400  # mm/day × km² → m³/s

obs_m3s   = obs         * factor
lstm_d_m3s = lstm_daymet * factor
lstm_m_m3s = lstm_maurer * factor
vic_d_m3s  = vic_daymet  * factor
vic_m_m3s  = vic_maurer  * factor

common = sorted(
    set(obs_m3s.columns)
    & set(lstm_d_m3s.columns)
    & set(lstm_m_m3s.columns)
    & set(vic_d_m3s.columns)
    & set(vic_m_m3s.columns)
    & set(basins.index)
)
obs_m3s    = obs_m3s[common]
lstm_d_m3s = lstm_d_m3s[common]
lstm_m_m3s = lstm_m_m3s[common]
vic_d_m3s  = vic_d_m3s[common]
vic_m_m3s  = vic_m_m3s[common]

mean_obs = obs_m3s.mean()

lstm_diff = abs(lstm_d_m3s - lstm_m_m3s).mean()
vic_diff  = abs(vic_d_m3s  - vic_m_m3s).mean()

lstm_rel_mean = lstm_diff / mean_obs
vic_rel_mean  = vic_diff  / mean_obs

contrast = lstm_rel_mean.abs() - vic_rel_mean.abs()

plot_df = pd.DataFrame({
    "lat":      basins.loc[common, "gauge_lat"],
    "lon":      basins.loc[common, "gauge_lon"],
    "contrast": contrast.reindex(common),
    "lstm_rel_mean": lstm_rel_mean.reindex(common),
    "vic_rel_mean":  vic_rel_mean.reindex(common),
}).dropna()

lo = np.percentile(plot_df["contrast"], 5)
hi = np.percentile(plot_df["contrast"], 95)
vmax = max(abs(lo), abs(hi))

plot_df = plot_df.sort_values("contrast", key=lambda s: s.abs())

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.AlbersEqualArea(
    central_latitude=39.5, central_longitude=-98.35))

states = cfeature.NaturalEarthFeature(
    category='cultural', name='admin_1_states_provinces_lakes',
    scale='50m', facecolor='none', edgecolor='0.5', linewidth=0.4)
ax.add_feature(states)
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.6)
ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.4, linestyle='--')
ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='#f0f0f0', edgecolor='none', zorder=0)
ax.set_extent([-125, -66, 24, 50], crs=ccrs.PlateCarree())

norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
cmap = plt.cm.RdBu  # Red = positive (LSTM discrepancy larger), Blue = negative (VIC discrepancy larger)

sc = ax.scatter(
    plot_df["lon"].values,
    plot_df["lat"].values,
    c=plot_df["contrast"].values,
    cmap=cmap,
    norm=norm,
    s=22,
    edgecolors="none",
    alpha=0.85,
    transform=ccrs.PlateCarree(),
    zorder=5,
)

cb = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02, aspect=30)
cb.set_label(r"$\mathrm{\overline{\Delta Q_{LSTM}}} - \mathrm{\overline{\Delta Q_{VIC}}}$")
# cb.set_label(r"$|\mathrm{LSTM}$ $\mathrm{rel}$ $\mathrm{mean}|$ $-$ $|\mathrm{VIC}$ $\mathrm{rel}$ $\mathrm{mean}|$")
cb.ax.text(2.5, -0.02, r"VIC larger",
           transform=cb.ax.transAxes, ha="center", va="top", fontsize=10)
cb.ax.text(2.5, 1.02, r"LSTM larger",
           transform=cb.ax.transAxes, ha="center", va="bottom", fontsize=10)

fig.savefig("figures/fig4_reproducibility_map.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
