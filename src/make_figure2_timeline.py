from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FormatStrFormatter

from src.figure_utils import save_figure, set_publication_style


def main() -> None:
    set_publication_style()
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 9.5
    plt.rcParams["xtick.labelsize"] = 9.5
    plt.rcParams["ytick.labelsize"] = 9.5

    fig, ax = plt.subplots(figsize=(13.5, 4.2))

    bars = [
        ("Production benchmark (12 countries)", 2005, 2024, "#4C78A8"),
        ("Trade benchmark (8 countries)", 2004, 2024, "#F58518"),
        ("Production sensitivity (8 countries)", 2005, 2024, "#54A24B"),
    ]
    ypos = [2, 1, 0]

    for y, (label, start, end, color) in zip(ypos, bars):
        ax.barh(
            y=y,
            width=end - start + 1,
            left=start,
            height=0.46,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )
        ax.text(start + 0.22, y, label, va="center", ha="left", fontsize=10.5, color="black")

    regimes = [
        (2014, 2019, "Pre-pandemic benchmark", "#DCEAF6"),
        (2020, 2021, "Pandemic shock", "#FBE4E2"),
        (2022, 2024, "Post-shock regime", "#E4F1E1"),
    ]
    for x0, x1, _, color in regimes:
        ax.axvspan(x0, x1 + 1, color=color, alpha=0.95, zorder=0)

    ax.set_xlim(2003.5, 2025.0)
    ax.set_ylim(-0.35, 2.35)
    ax.set_yticks(ypos)
    ax.set_yticklabels(["", "", ""])
    ax.tick_params(axis="y", length=0)

    year_ticks = list(range(2004, 2025, 2))
    ax.xaxis.set_major_locator(FixedLocator(year_ticks))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax.set_xlabel("Year")
    ax.grid(axis="x", linestyle="--", alpha=0.28)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    legend_handles = [Patch(facecolor=color, edgecolor="none", label=label) for _, _, label, color in regimes]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=3,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.8,
    )

    plt.tight_layout()
    save_figure(fig, "Figure_2_Benchmark_timeline_and_evaluation_regimes")
    plt.close(fig)


if __name__ == "__main__":
    main()
