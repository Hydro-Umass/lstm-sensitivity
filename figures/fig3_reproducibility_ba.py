# Figure 3: Bland-Altman style plot for hydrologic reproducibility

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns

sns.set_context("paper", font_scale=2.5)

def reproducible_hydrologic_signal():
    basins = pd.read_csv("data/camels_topo.txt", dtype={"gauge_id": str}, sep=";", index_col="gauge_id")
    obs = pd.read_csv("outputs/ealstm_traindaymet_evalnldas_valid_observations.csv.gz", index_col=0, parse_dates=True)
    lstmd = pd.read_csv("outputs/ealstm_traindaymet_evalnldas_valid_predictions.csv.gz", index_col=0, parse_dates=True)
    lstmm = pd.read_csv("outputs/ealstm_trainmaurer_evalnldas_valid_predictions.csv.gz", index_col=0, parse_dates=True)
    vicd = pd.read_csv("outputs/vic_traindaymet_evalnldas_valid_predictions.csv.gz", index_col=0, parse_dates=True)
    vicm = pd.read_csv("outputs/vic_trainmaurer_evalnldas_valid_predictions.csv.gz", index_col=0, parse_dates=True)
    # filter negative or zero values
    obs.values[obs.values <= 0] = 1e-6
    lstmd.values[lstmd.values <= 0] = 1e-6
    lstmm.values[lstmm.values <= 0] = 1e-6
    vicd.values[vicd.values <= 0] = 1e-6
    vicm.values[vicm.values <= 0] = 1e-6
    # resample to monthly
    # obs = obs.resample("MS").mean()
    # lstmd = lstmd.resample("MS").mean()
    # lstmm = lstmm.resample("MS").mean()
    # vicd = vicd.resample("MS").mean()
    # vicm = vicm.resample("MS").mean()
    # convert to m3/s
    obs = pd.melt((obs * basins.T.loc['area_geospa_fabric', :] * 1e6 / 86400).reset_index().rename(columns={"index": "date"}), id_vars="date", var_name="gauge")
    lstmd = pd.melt((lstmd * basins.T.loc['area_geospa_fabric', :] * 1e6 / 86400).reset_index().rename(columns={"index": "date"}), id_vars="date", var_name="gauge")
    lstmm = pd.melt((lstmm * basins.T.loc['area_geospa_fabric', :] * 1e6 / 86400).reset_index().rename(columns={"index": "date"}), id_vars="date", var_name="gauge")
    vicd = pd.melt((vicd * basins.T.loc['area_geospa_fabric', :] * 1e6 / 86400).reset_index().rename(columns={"index": "date"}), id_vars="date", var_name="gauge")
    vicm = pd.melt((vicm * basins.T.loc['area_geospa_fabric', :] * 1e6 / 86400).reset_index().rename(columns={"index": "date"}), id_vars="date", var_name="gauge")
    # join data frames
    lstm = pd.merge(lstmd, lstmm, on=["date", "gauge"]).rename(columns={"value_x": "daymet", "value_y": "maurer"})
    vic = pd.merge(vicd, vicm, on=["date", "gauge"]).rename(columns={"value_x": "daymet", "value_y": "maurer"})
    lstm["diff"] = (lstm.daymet - lstm.maurer)
    # lstm['diff'] =  ((lstm.set_index('gauge').daymet - lstm.set_index('gauge').maurer) / obs.groupby('gauge').apply(lambda d: d.value.mean(), include_groups=False)).values
    vic["diff"] = (vic.daymet - vic.maurer)
    # vic['diff'] = ((vic.set_index('gauge').daymet - vic.set_index('gauge').maurer) / obs.groupby('gauge').apply(lambda d: d.value.mean(), include_groups=False)).values
     # obs['value'] = (obs.set_index('gauge').value / obs.groupby('gauge').apply(lambda d: d.value.mean(), include_groups=False)).values
    def calc_monthly(df):
        df['year'] = df.date.dt.year
        df['month'] = df.date.dt.month
        df = df.groupby(['year', 'month', 'gauge']).apply(lambda d: d.mean() if isinstance(d.iloc[0], np.float64)  else d.iloc[0], include_groups=False)
        return df.reset_index().drop(columns=['year', 'month'])
    # vic = calc_monthly(vic)
    # lstm = calc_monthly(lstm)
    def remove_outliers(df):
        df = df.dropna()
        df.loc[df['diff'] > np.percentile(df['diff'], 99), 'diff'] = np.nan
        df = df.dropna()
        df.loc[df['diff'] < np.percentile(df['diff'], 1), 'diff'] = np.nan
        df = df.dropna()
        return df
    # Empirical 95% interval and mean diff on the unclipped data — the
    # load-bearing numbers for §3.2. Computed before remove_outliers so the
    # bounds reflect the true daily distribution, not the 1/99 visualization clip.
    pre_lstm = pd.merge(obs, lstm, on=["date", "gauge"]).dropna()
    pre_vic = pd.merge(obs, vic, on=["date", "gauge"]).dropna()
    def emp_stats(df):
        d = df["diff"]
        return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)
    stats_lstm = emp_stats(pre_lstm)
    stats_vic = emp_stats(pre_vic)

    df_lstm = remove_outliers(pre_lstm.copy())
    df_vic = remove_outliers(pre_vic.copy())

    # plot
    fig, ax = plt.subplots(1, 2, figsize=(20, 8), sharex=True, sharey=True,
                           constrained_layout=True)

    # Shared bin edges across both panels so the colormap is comparable.
    bins = 150
    x_min = max(min(df_lstm.value.min(), df_vic.value.min()), 1e-4)
    x_max = max(df_lstm.value.max(), df_vic.value.max())
    y_min = min(df_lstm["diff"].min(), df_vic["diff"].min())
    y_max = max(df_lstm["diff"].max(), df_vic["diff"].max())
    x_edges = np.geomspace(x_min, x_max, bins + 1)
    y_edges = np.linspace(y_min, y_max, bins + 1)
    h_lstm, _, _ = np.histogram2d(df_lstm.value, df_lstm["diff"],
                                  bins=[x_edges, y_edges])
    h_vic, _, _ = np.histogram2d(df_vic.value, df_vic["diff"],
                                 bins=[x_edges, y_edges])
    norm = LogNorm(vmin=1, vmax=max(h_lstm.max(), h_vic.max()))

    panels = [
        (ax[0], df_lstm, stats_lstm, "LSTM (Daymet − Maurer)"),
        (ax[1], df_vic, stats_vic, "VIC (Daymet − Maurer)"),
    ]
    for a, df, (m, lo, hi), title in panels:
        _, _, _, mesh = a.hist2d(df.value, df["diff"], bins=[x_edges, y_edges],
                                 norm=norm, cmap="Blues")
        a.set_xscale("log")
        a.axhline(m, color="black", linestyle="-", linewidth=1.2)
        a.axhline(lo, color="black", linestyle="--", linewidth=1.2)
        a.axhline(hi, color="black", linestyle="--", linewidth=1.2)
        a.set_title(title)
        a.set_xlabel("Observed Discharge (m³/s)")
    ax[0].set_ylabel("Predicted Discharge Difference (m³/s)")

    fig.colorbar(mesh, ax=ax, shrink=0.85, pad=0.02, label="Count")
    fig.savefig("hydro_reproducibility_ba.png", dpi=300,
                bbox_inches="tight", pad_inches=0.1)
