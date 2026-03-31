from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import ROUND1_RESULTS_FILE, ROUND2_RESULTS_FILE, ROUND3_RESULTS_FILE
from src.figure_utils import normalize_panel_name, normalize_target_name, save_figure, set_publication_style


def main() -> None:
    set_publication_style()

    round1 = pd.read_excel(ROUND1_RESULTS_FILE, sheet_name="summary_by_split_target")
    round2 = pd.read_excel(ROUND2_RESULTS_FILE, sheet_name="summary_by_split_target")
    round3 = pd.read_excel(ROUND3_RESULTS_FILE, sheet_name="summary_by_split_target")

    for df in [round1, round2, round3]:
        df["panel_std"] = df["panel"].apply(normalize_panel_name)
        df["target_std"] = df["target"].apply(normalize_target_name)

    round1["source"] = "round1"
    round2["source"] = "round2"
    round3["source"] = "round3"

    all_results = pd.concat([round1, round2, round3], ignore_index=True)
    model_set = ["Seasonal Naive", "ETS", "Theta", "LightGBM", "LSTM"]
    all_results = all_results[all_results["model"].isin(model_set)].copy()

    main_df = all_results[
        ((all_results["panel_std"] == "Production_12c") & (all_results["target_std"].isin(["C16", "C31"])))
        | ((all_results["panel_std"] == "Trade_8c") & (all_results["target_std"].isin(["Exports", "Imports"])))
    ].copy()

    priority_map = {"round1": 1, "round2": 2, "round3": 3}
    main_df["source_priority"] = main_df["source"].map(priority_map)
    main_df = main_df.sort_values(["panel_std", "target_std", "split_id", "model", "source_priority"])
    main_df = main_df.drop_duplicates(subset=["panel_std", "target_std", "split_id", "model"], keep="last").copy()

    summary = (
        main_df.groupby(["panel_std", "target_std", "model"], as_index=False)["mean_mase"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["se"] = summary["std"] / np.sqrt(summary["count"])
    summary["lower"] = summary["mean"] - summary["se"]
    summary["upper"] = summary["mean"] + summary["se"]

    panel_specs = [
        ("A. Production: C16", "Production_12c", "C16"),
        ("B. Production: C31", "Production_12c", "C31"),
        ("C. Trade: Exports", "Trade_8c", "Exports"),
        ("D. Trade: Imports", "Trade_8c", "Imports"),
    ]

    winner_color = "#D55E00"
    other_color = "#6E6E6E"

    fig, axes = plt.subplots(2, 2, figsize=(13, 7.8), constrained_layout=True)
    axes = axes.flatten()

    for ax, (title, panel_key, target_key) in zip(axes, panel_specs):
        sub = summary[(summary["panel_std"] == panel_key) & (summary["target_std"] == target_key)].copy()
        sub = sub.sort_values("mean", ascending=True).reset_index(drop=True)
        y = np.arange(len(sub))
        winner_model = sub.loc[sub["mean"].idxmin(), "model"]

        xmin = max(0, sub["lower"].min() - 0.05 * (sub["upper"].max() - sub["lower"].min()))
        xmax = sub["upper"].max() + 0.10 * (sub["upper"].max() - sub["lower"].min())
        ax.set_xlim(xmin, xmax)
        xspan = xmax - xmin
        text_offset = 0.02 * xspan

        for i, row in sub.iterrows():
            color = winner_color if row["model"] == winner_model else other_color
            lw = 2.2 if row["model"] == winner_model else 1.8
            ms = 46 if row["model"] == winner_model else 38
            ax.hlines(y=i, xmin=row["lower"], xmax=row["upper"], linewidth=lw, color=color, zorder=2)
            ax.scatter(row["mean"], i, s=ms, color=color, zorder=3)
            ax.text(row["upper"] + text_offset, i, f"{row['mean']:.2f}", va="center", ha="left", fontsize=8.5, color=color)

        ax.set_yticks(y)
        ax.set_yticklabels(sub["model"])
        ax.invert_yaxis()
        ax.set_title(title, loc="left", pad=6)
        ax.set_xlabel("Mean MASE")
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        ax.grid(axis="y", visible=False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)

    save_figure(fig, "Figure_4_Main_model_comparison_across_benchmark_targets")
    plt.close(fig)


if __name__ == "__main__":
    main()
