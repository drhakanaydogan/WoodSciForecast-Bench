from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.config import ROUND2_RESULTS_FILE
from src.figure_utils import save_figure, set_publication_style


def normalize_trade_panel_name(value: object) -> object:
    if pd.isna(value):
        return value
    text = str(value).strip().lower()
    if "trade" in text:
        return "Trade_8c"
    return text


def normalize_trade_target_name(value: object) -> object:
    if pd.isna(value):
        return value
    text = str(value).strip().lower()
    if "export" in text:
        return "Exports"
    if "import" in text:
        return "Imports"
    return text


def main() -> None:
    set_publication_style()
    plt.rcParams["axes.titlesize"] = 10.8

    relative = pd.read_excel(ROUND2_RESULTS_FILE, sheet_name="lgbm_relative_perf")
    relative["panel_std"] = relative["panel"].apply(normalize_trade_panel_name)
    relative["target_std"] = relative["target"].apply(normalize_trade_target_name)

    trade_rel = relative[relative["panel_std"] == "Trade_8c"].copy()
    trade_rel = trade_rel[["target_std", "split_id", "lgbm_vs_snaive_pct", "lgbm_vs_ets_pct"]].copy()

    target_order = ["Exports", "Imports"]
    split_order = ["T1", "T2", "T3"]
    trade_rel["target_std"] = pd.Categorical(trade_rel["target_std"], categories=target_order, ordered=True)
    trade_rel["split_id"] = pd.Categorical(trade_rel["split_id"], categories=split_order, ordered=True)
    trade_rel = trade_rel.sort_values(["target_std", "split_id"]).copy()

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.9), sharex=True)

    color_snaive = "#4C78A8"
    color_ets = "#F58518"
    connector_color = "#9A9A9A"

    all_values = np.concatenate([trade_rel["lgbm_vs_snaive_pct"].values, trade_rel["lgbm_vs_ets_pct"].values])
    xmin = np.floor(all_values.min() - 5)
    xmax = 2

    for ax, target in zip(axes, target_order):
        sub = trade_rel[trade_rel["target_std"] == target].copy().sort_values("split_id")
        y = np.array([2, 1, 0])
        x1 = sub["lgbm_vs_snaive_pct"].values
        x2 = sub["lgbm_vs_ets_pct"].values
        labels = sub["split_id"].astype(str).tolist()

        for yi, a, b in zip(y, x1, x2):
            ax.hlines(y=yi, xmin=min(a, b), xmax=max(a, b), color=connector_color, linewidth=1.4, zorder=1)

        ax.scatter(x1, y, s=68, color=color_snaive, zorder=3)
        ax.scatter(x2, y, s=68, color=color_ets, zorder=3)

        for yi, a, b in zip(y, x1, x2):
            ax.text(a - 1.8, yi + 0.06, f"{a:.1f}", ha="right", va="bottom", fontsize=8.5, color=color_snaive)
            ax.text(b + 1.8, yi - 0.06, f"{b:.1f}", ha="left", va="top", fontsize=8.5, color=color_ets)

        ax.axvline(0, color="black", linewidth=0.9)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-0.15, 2.15)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_title(f"{'A' if target == 'Exports' else 'B'}. Trade: {target}", loc="left", pad=10)
        ax.grid(axis="x", linestyle="--", alpha=0.22)
        ax.grid(axis="y", visible=False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)

    axes[0].set_ylabel("Evaluation split")
    axes[0].set_xlabel("Percent difference in mean MASE")
    axes[1].set_xlabel("Percent difference in mean MASE")

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color_snaive, markersize=8, label="Against Seasonal Naive"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color_ets, markersize=8, label="Against ETS"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=2, frameon=False)
    plt.subplots_adjust(top=0.84, wspace=0.06)

    save_figure(fig, "Figure_6_Relative_advantage_of_LightGBM_across_trade_benchmark_splits")
    plt.close(fig)


if __name__ == "__main__":
    main()
