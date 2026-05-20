# Figure 2: baseline performance boxplot

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import kge, nse

sns.set_context("paper", font_scale=2.5)

def model_performance_with_different_datasets():
    data = []
    for forcing in ["daymet", "nldas", "maurer"]:
        obs = pd.read_csv(f"outputs/ealstm_{forcing}_valid_observations.csv.gz", index_col=0, parse_dates=True)
        lstm = pd.read_csv(f"outputs/ealstm_{forcing}_valid_predictions.csv.gz", index_col=0, parse_dates=True).loc[:, obs.columns]
        vic = pd.read_csv(f"outputs/vic_{forcing}_valid_predictions.csv.gz", index_col=0, parse_dates=True).loc[:, obs.columns]
        df = pd.DataFrame(dict(NSE=nse(obs, lstm), KGE=kge(obs, lstm)))
        strfun = str.upper if forcing == "nldas" else str.capitalize
        df["Model"] = f"LSTM-{strfun(forcing)}"
        data.append(df)
        df = pd.DataFrame(dict(NSE=nse(obs, vic), KGE=kge(obs, vic)))
        df["Model"] = f"VIC-{strfun(forcing)}"
        data.append(df)
    data = pd.melt(pd.concat(data), id_vars="Model", var_name="Metric", value_name="Value")
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    sns.boxplot(data=data.loc[data.Value >= -1.0, :].sort_values(by="Model"), x="Model", y="Value", hue="Metric", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig("lstm_vic_baseline.png", dpi=150, bbox_inches=0, pad_inches=0.1)
