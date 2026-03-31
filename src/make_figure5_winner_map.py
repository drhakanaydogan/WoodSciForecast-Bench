from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

from src.config import ROUND2_RESULTS_FILE, ROUND3_RESULTS_FILE
from src.figure_utils import (
    normalize_model_name,
    normalize_panel_name,
    normalize_target_name,
    pretty_block_label,
    save_figure,
    set_publication_style,
)


def main() -> None:
    set_publication_style()

    winners_round2 = pd.read_excel(ROUND2_RESULTS_FILE, sheet_name="block_winners")
    winners_round3 = pd.read_excel(ROUND3_RESULTS_FILE, sheet_name="block_winners")

    for df in [winners_round2, winners_round3]:
        df["panel_std"] = df["panel"].apply(normalize_panel_name)
        df["target_std"] = df["target"].apply(normalize_target_name)
        df["model_std"] = df["model"].apply(normalize_model_name)

    winners_round2["source"] = "round2"
    winners_round3["source"] = "round3"

    combined = pd.concat([winners_round2, winners_round3], ignore_index=True)
    priority_map = {"round2": 2, "round3": 3}
    combined["source_priority"] = combined["source"].map(priority_map)
    combined = combined.sort_values(["panel_std", "target_std", "split_id", "source_priority"])
    combined = combined.drop_duplicates(subset=["panel_std", "target_std", "split_id"], keep="last").copy()
    combined["block"] = combined["panel_std"] + " | " + combined["target_std"]

    prod_order = [
        "Production_12c | C16",
        "Production_12c | C31",
        "Production_8c | C16",
        "Production_8c | C31",
    ]
    trade_order = ["Trade_8c | Exports", "Trade_8c | Imports"]
    prod_splits = ["S1", "S2", "S3"]
    trade_splits = ["T1", "T2", "T3"]

    prod_df = combined[combined["block"].isin(prod_order)].copy()
    trade_df = combined[combined["block"].isin(trade_order)].copy()

    prod_pivot = prod_df.pivot(index="block", columns="split_id", values="model_std")
    trade_pivot = trade_df.pivot(index="block", columns="split_id", values="model_std")

    prod_pivot = prod_pivot.reindex(index=prod_order, columns=prod_splits)
    trade_pivot = trade_pivot.reindex(index=trade_order, columns=trade_splits)

    prod_pivot.index = [pretty_block_label(value) for value in prod_pivot.index]
    trade_pivot.index = [pretty_block_label(value) for value in trade_pivot.index]

    model_order = ["Seasonal Naive", "ETS", "Theta", "LightGBM", "LSTM"]
    model_to_num = {model: i for i, model in enumerate(model_order)}

    prod_num = prod_pivot.copy()
    trade_num = trade_pivot.copy()
    for model, code in model_to_num.items():
        prod_num = prod_num.replace(model, code)
        trade_num = trade_num.replace(model, code)

    prod_num = prod_num.apply(pd.to_numeric, errors="coerce")
    trade_num = trade_num.apply(pd.to_numeric, errors="coerce")

    palette = {
        "Seasonal Naive": "#BDBDBD",
        "ETS": "#4C78A8",
        "Theta": "#72B7B2",
        "LightGBM": "#F58518",
        "LSTM": "#E45756",
    }
    cmap = ListedColormap([palette[model] for model in model_order])

    fig, axes = plt.subplots(2, 1, figsize=(10.8, 6.4), gridspec_kw={"height_ratios": [4, 2]}, constrained_layout=True)

    sns.heatmap(
        prod_num,
        annot=prod_pivot,
        fmt="",
        cmap=cmap,
        cbar=False,
        linewidths=0.8,
        linecolor="white",
        ax=axes[0],
        vmin=0,
        vmax=len(model_order) - 1,
    )
    axes[0].set_title("A. Split-wise winners in production benchmark blocks", loc="left", pad=8)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("")
    axes[0].set_xticklabels(prod_splits, rotation=0)
    axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0)

    sns.heatmap(
        trade_num,
        annot=trade_pivot,
        fmt="",
        cmap=cmap,
        cbar=False,
        linewidths=0.8,
        linecolor="white",
        ax=axes[1],
        vmin=0,
        vmax=len(model_order) - 1,
    )
    axes[1].set_title("B. Split-wise winners in trade benchmark blocks", loc="left", pad=8)
    axes[1].set_xlabel("Evaluation split")
    axes[1].set_ylabel("")
    axes[1].set_xticklabels(trade_splits, rotation=0)
    axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0)

    for ax in axes:
        for text in ax.texts:
            label = text.get_text()
            text.set_color("white" if label in ["LightGBM", "ETS", "Theta", "LSTM"] else "black")
            text.set_fontsize(10)

    handles = [mpatches.Patch(color=palette[model], label=model) for model in model_order]
    fig.legend(handles=handles, title="Winner", loc="center left", bbox_to_anchor=(1.01, 0.82), frameon=False)

    save_figure(fig, "Figure_5_Split_wise_winning_models_across_production_and_trade_benchmark_blocks")
    plt.close(fig)


if __name__ == "__main__":
    main()
