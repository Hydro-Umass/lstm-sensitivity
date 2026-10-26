"""Compute seed-aggregated per-basin median statistics for perturbation experiments.

Outputs two LaTeX tables:
  Table S2 — random error (NSE, KGE, r, α, β, FHV) for LSTM and VIC
  Table S3 — systematic bias (NSE, KGE, r, α, β, FHV, FLV, rMAE) for LSTM and VIC

Each LSTM value is the across-seed median of the per-seed, per-basin median
(utils.aggregate_seeds); metrics are never computed on the ensemble-mean
hydrograph. VIC runs are single members. The across-seed spread of every
entry is written to a CSV next to the tables (--spread-csv), which supports
the blanket bound quoted in the text.

Usage: python figures/compute_perturbation_metrics.py --output-dir outputs [--forcing daymet]
"""

import argparse
import pandas as pd
import numpy as np

from utils import ensemble_summary, METRICS

STD_LEVELS = [0.1, 0.25, 0.5]
BIAS_LEVELS = [-0.5, -0.25, -0.1, 0.1, 0.25, 0.5]
MODELS = [("LSTM", "ealstm"), ("VIC", "vic")]


def summarize(output_dir, forcing, model, variant):
    """Seed-aggregated value and spread for every metric, or None if the run is missing."""
    try:
        agg, _ = ensemble_summary(output_dir, forcing, variant=variant, model=model)
    except (FileNotFoundError, ValueError):
        return None
    return agg


def collect(output_dir, forcing, conditions, level_name):
    """
    conditions: list of (level_label, [(condition_label, variant), ...]).
    Returns (table rows with values, long DataFrame of spreads).
    """
    rows, spreads = [], []
    for model_label, model in MODELS:
        for level, conds in conditions:
            for cond_label, variant in conds:
                agg = summarize(output_dir, forcing, model, variant)
                vals = {m: (None if agg is None else agg.loc[m, "value"]) for m in METRICS}
                rows.append(dict(Model=model_label, Condition=cond_label, **{level_name: level}, **vals))
                if agg is not None:
                    spreads.append(agg.assign(Model=model_label, Condition=cond_label,
                                              level=level, variant=variant or "baseline"))
                else:
                    print(f"  missing: {model}_{forcing}_{variant}")
    spreads = pd.concat(spreads).rename_axis("metric").reset_index() if spreads else pd.DataFrame()
    return pd.DataFrame(rows), spreads


def random_conditions():
    conds = [("---", [("Baseline", "")])]
    for std in STD_LEVELS:
        lvl = f"{std:g}"
        conds.append((f"{int(std*100)}\\%", [
            ("Train \\& eval", f"random_error_std{lvl}"),
            ("Train only",     f"random_error_std{lvl}_noperturb"),
            ("Infer only",     f"infer_random_error_std{lvl}"),
        ]))
    return conds


def bias_conditions():
    conds = [("---", [("Baseline", "")])]
    for b in BIAS_LEVELS:
        conds.append((f"{int(b*100):+d}\\%", [
            ("Train \\& eval", f"bias{b:g}"),
            ("Train only",     f"bias{b:g}_noperturb"),
            ("Infer only",     f"infer_bias{b:g}"),
        ]))
    return conds


def fmt(v, decimals=2, signed=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "---"
    fmt_str = f"{{:+.{decimals}f}}" if signed else f"{{:.{decimals}f}}"
    return fmt_str.format(v)


def latex_row_rand(row, is_lstm_baseline=False):
    vals = " & ".join([
        fmt(row["NSE"]),
        fmt(row["KGE"]),
        fmt(row["r"], 3),
        fmt(row["alpha"], 3),
        fmt(row["beta"], 3),
        fmt(row["FHV"], 1, signed=True),
    ])
    return vals


def print_table_s2(df):
    lstm = df[df.Model == "LSTM"].reset_index(drop=True)
    vic  = df[df.Model == "VIC"].reset_index(drop=True)
    assert len(lstm) == len(vic)

    lines = []
    lines.append(r"\begin{table}")
    lines.append(r"  \settablenum{S2}")
    lines.append(r"  \caption{Per-basin median statistics for random error perturbation experiments")
    lines.append(r"    (Daymet forcing, validation period 2000--2008). Condition indicates whether")
    lines.append(r"    the perturbation was applied during training, evaluation, or both.")
    lines.append(r"    Metrics: NSE = Nash--Sutcliffe efficiency; KGE = Kling--Gupta efficiency;")
    lines.append(r"    $r$ = Pearson correlation; $\alpha$ = variability ratio ($\sigma_\mathrm{sim}/\sigma_\mathrm{obs}$);")
    lines.append(r"    $\beta$ = bias ratio ($\mu_\mathrm{sim}/\mu_\mathrm{obs}$);")
    lines.append(r"    FHV = high-flow volume error (\% error in total flow above the 98th-percentile threshold).}")
    lines.append(r"  \label{tab:s2}")
    lines.append(r"  \footnotesize\centering")
    lines.append(r"  \begin{tabular}{llrrrrrrrrrrrr}")
    lines.append(r"    \hline")
    lines.append(r"    & & \multicolumn{6}{c}{LSTM} & \multicolumn{6}{c}{VIC} \\")
    lines.append(r"    \cline{3-8}\cline{9-14}")
    lines.append(r"    Condition & $\sigma$ & NSE & KGE & $r$ & $\alpha$ & $\beta$ & FHV(\%) & NSE & KGE & $r$ & $\alpha$ & $\beta$ & FHV(\%) \\")
    lines.append(r"    \hline")

    prev_cond = None
    for i in range(len(lstm)):
        lr, vr = lstm.iloc[i], vic.iloc[i]
        cond = lr["Condition"]
        sig = lr["sigma"]
        if cond == "Baseline":
            lines.append(
                f"    {cond} & {sig} & {latex_row_rand(lr)} & {latex_row_rand(vr)} \\\\"
            )
            lines.append(r"    \hline")
        else:
            if cond != prev_cond and prev_cond != "Baseline":
                lines.append(r"    \hline")
            lines.append(
                f"    {cond} & {sig} & {latex_row_rand(lr)} & {latex_row_rand(vr)} \\\\"
            )
        prev_cond = cond

    lines.append(r"    \hline")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def latex_row_bias(row):
    vals = " & ".join([
        fmt(row["NSE"]),
        fmt(row["KGE"]),
        fmt(row["r"], 3),
        fmt(row["alpha"], 3),
        fmt(row["beta"], 3),
        fmt(row["FHV"], 1, signed=True),
        fmt(row["FLV"], 1, signed=True),
        fmt(row["rMAE"], 1),
    ])
    return vals


def print_table_s3(df):
    lstm = df[df.Model == "LSTM"].reset_index(drop=True)
    vic  = df[df.Model == "VIC"].reset_index(drop=True)
    assert len(lstm) == len(vic)

    lines = []
    lines.append(r"\begin{table}")
    lines.append(r"  \settablenum{S3}")
    lines.append(r"  \caption{Per-basin median statistics for systematic bias perturbation experiments")
    lines.append(r"    (Daymet forcing, validation period 2000--2008). Bias levels are stated as signed")
    lines.append(r"    percentage changes relative to the baseline precipitation (e.g., $-50\%$ corresponds")
    lines.append(r"    to $P \times 0.5$; $+50\%$ to $P \times 1.5$). Condition indicates whether the")
    lines.append(r"    perturbation was applied during training, evaluation, or both.")
    lines.append(r"    Metrics: NSE, KGE, $r$, $\alpha$, $\beta$, FHV as in Table~\ref{tab:s2};")
    lines.append(r"    FLV = low-flow volume error (\% error in total flow below the 30th-percentile threshold);")
    lines.append(r"    rMAE = relative mean absolute error (MAE normalized by mean observed flow, \%).}")
    lines.append(r"  \label{tab:s3}")
    lines.append(r"  \footnotesize\centering")
    lines.append(r"  \begin{tabular}{llrrrrrrrrrrrrrrrr}")
    lines.append(r"    \hline")
    lines.append(r"    & & \multicolumn{8}{c}{LSTM} & \multicolumn{8}{c}{VIC} \\")
    lines.append(r"    \cline{3-10}\cline{11-18}")
    lines.append(r"    Condition & Bias & NSE & KGE & $r$ & $\alpha$ & $\beta$ & FHV(\%) & FLV(\%) & rMAE(\%) &"
                 r" NSE & KGE & $r$ & $\alpha$ & $\beta$ & FHV(\%) & FLV(\%) & rMAE(\%) \\")
    lines.append(r"    \hline")

    prev_cond = None
    for i in range(len(lstm)):
        lr, vr = lstm.iloc[i], vic.iloc[i]
        cond = lr["Condition"]
        bval = lr["bias"]
        if cond == "Baseline":
            lines.append(
                f"    {cond} & {bval} & {latex_row_bias(lr)} & {latex_row_bias(vr)} \\\\"
            )
            lines.append(r"    \hline")
        else:
            if cond != prev_cond and prev_cond != "Baseline":
                lines.append(r"    \hline")
            lines.append(
                f"    {cond} & {bval} & {latex_row_bias(lr)} & {latex_row_bias(vr)} \\\\"
            )
        prev_cond = cond

    lines.append(r"    \hline")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--forcing", default="daymet")
    parser.add_argument("--spread-csv", default="perturbation_seed_spread.csv")
    args = parser.parse_args()

    rand_df, rand_spread = collect(args.output_dir, args.forcing, random_conditions(), "sigma")
    bias_df, bias_spread = collect(args.output_dir, args.forcing, bias_conditions(), "bias")

    print(print_table_s2(rand_df))
    print()
    print(print_table_s3(bias_df))

    spread = pd.concat([rand_spread.assign(experiment="random"),
                        bias_spread.assign(experiment="bias")], ignore_index=True)
    spread.to_csv(args.spread_csv, index=False)
    lstm = spread[(spread.Model == "LSTM") & (spread.n_seeds > 1)]
    if len(lstm):
        print()
        print(f"% LSTM across-seed spread written to {args.spread_csv}")
        print(f"% n_seeds per configuration: {sorted(lstm.n_seeds.unique())}")
        print("% largest across-seed spread of the per-basin median, any configuration:")
        print("% " + ", ".join(f"{m}={lstm.loc[lstm.metric == m, 'spread'].max():.3g}" for m in METRICS))
