# Figure 1: precipitation difference histogram

from glob import glob
from utils import read_met
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
import seaborn as sns

sns.set_context("paper", font_scale=2.5)

def confusion_matrix(df, tcol, pcol):
    cm = np.zeros((2, 2))
    cm[0, 0] = sum((df[tcol] == 0) & (df[pcol] == 0)) / len(df) * 100
    cm[0, 1] = sum((df[tcol] == 0) & (df[pcol] > 0)) / len(df) * 100
    cm[1, 0] = sum((df[tcol] > 0) & (df[pcol] == 0)) / len(df) * 100
    cm[1, 1] = sum((df[tcol] > 0) & (df[pcol] > 0)) / len(df) * 100
    return cm

def precip_differences():
    daymet = pd.concat([read_met(f).loc["1985-10-1":"2008-9-30", :] for f in glob("data/daymet/*txt")])
    nldas = pd.concat([read_met(f).loc["1985-10-1":"2008-9-30", :] for f in glob("data/nldas/*txt")])
    maurer = pd.concat([read_met(f).loc["1985-10-1":"2008-9-30", :] for f in glob("data/maurer/*txt")])
    daymet = daymet.reset_index().set_index(['Date', 'Gauge']).Prcp
    nldas = nldas.reset_index().set_index(['Date', 'Gauge']).Prcp
    maurer = maurer.reset_index().set_index(['Date', 'Gauge']).Prcp
    df = pd.DataFrame(dict(daymet=daymet, nldas=nldas, maurer=maurer))
    dfp = df.loc[np.all(df > 0, axis=1), :]
    fig, ax = plt.subplots(2, 3, figsize=(21, 11))
    sns.histplot((dfp.daymet - dfp.nldas) / dfp.mean(axis=1) * 100, stat='percent', ax=ax[0, 0])
    sns.histplot((dfp.daymet - dfp.maurer) / dfp.mean(axis=1) * 100, stat='percent', ax=ax[0, 1])
    sns.histplot((dfp.nldas - dfp.maurer) / dfp.mean(axis=1) * 100, stat='percent', ax=ax[0, 2])
    for a in ax[0, :]:
        a.set_ylabel("Fraction")
    ax[0, 0].set_title("Daymet - NLDAS")
    ax[0, 1].set_title("Daymet - Maurer")
    ax[0, 2].set_title("NLDAS - Maurer")
    ax[0, 0].set_xlabel("Difference (%)")
    ax[0, 1].set_xlabel("Difference (%)")
    ax[0, 2].set_xlabel("Difference (%)")
    labels = ["P=0", "P>0"]
    cm_bnds = (0.0, 60.0)
    sns.heatmap(confusion_matrix(df, "daymet", "nldas"), annot=True, cbar=False, xticklabels=labels, yticklabels=labels,
                ax=ax[1, 0], vmin=cm_bnds[0], vmax=cm_bnds[1], cmap="coolwarm")
    sns.heatmap(confusion_matrix(df, "daymet", "maurer"), annot=True, cbar=False, xticklabels=labels, yticklabels=labels,
                ax=ax[1, 1], vmin=cm_bnds[0], vmax=cm_bnds[1], cmap="coolwarm")
    sns.heatmap(confusion_matrix(df, "nldas", "maurer"), annot=True, cbar=False, xticklabels=labels, yticklabels=labels,
                ax=ax[1, 2], vmin=cm_bnds[0], vmax=cm_bnds[1], cmap="coolwarm")
    ax[1, 0].set_ylabel("Daymet")
    ax[1, 0].set_xlabel("NLDAS")
    ax[1, 1].set_ylabel("Daymet")
    ax[1, 1].set_xlabel("Maurer")
    ax[1, 2].set_ylabel("NLDAS")
    ax[1, 2].set_xlabel("Maurer")
    labels = ["a)", "b)", "c)", "d)", "e)", "f)"]
    for i, a in enumerate(ax.flatten()):
        a.text(0.0, 1.0, labels[i], transform=(a.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)), fontsize='large', va='bottom', fontfamily='serif')
    fig.tight_layout()
    fig.savefig("fig1_precip_diffs.png", dpi=300, bbox_inches=0, pad_inches=0.1)
