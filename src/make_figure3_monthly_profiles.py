from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import BENCHMARK_PROTOCOL_FILE
from src.figure_utils import save_figure, set_publication_style
from src.io import load_benchmark_panels


MONTH_ORDER = list(range(1, 13))
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def monthly_profile(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    summary = (
        df.groupby("month")[value_col]
        .agg(["mean", "std", "count"])
        .reindex(MONTH_ORDER)
        .reset_index()
    )
    summary["se"] = summary["std"] / np.sqrt(summary["count"])
    return summary


def main() -> None:
    set_publication_style()
    prod12, _, trade8 = load_benchmark_panels(BENCHMARK_PROTOCOL_FILE)

    series_specs = [
        ("A. Production: C16", prod12, "ln_sts_c16_sa_idx2021_100", "#4C78A8"),
        ("B. Production: C31", prod12, "ln_sts_c31_sa_idx2021_100", "#72B7B2"),
        ("C. Trade: Exports", trade8, "ln_trade_world_export_eur_hs4sum", "#F58518"),
        ("D. Trade: Imports", trade8, "ln_trade_world_import_eur_hs4sum", "#E45756"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 7.2), constrained_layout=True)
    axes = axes.flatten()

    for ax, (title, df, column, color) in zip(axes, series_specs):
        profile = monthly_profile(df, column)
        x = np.arange(1, 13)
        y = profile["mean"].values
        se = profile["se"].values

        ax.plot(x, y, linewidth=2.0, marker="o", markersize=4.2, color=color, zorder=3)
        ax.fill_between(x, y - se, y + se, color=color, alpha=0.12, zorder=2)
        ax.set_title(title, loc="left", pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels(MONTH_LABELS, rotation=0)
        ax.set_xlim(1, 12)
        ax.set_ylabel("Log-transformed value")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.grid(axis="x", visible=False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)

    axes[2].set_xlabel("Month")
    axes[3].set_xlabel("Month")

    save_figure(fig, "Figure_3_Monthly_profiles_of_the_benchmark_targets")
    plt.close(fig)


if __name__ == "__main__":
    main()
