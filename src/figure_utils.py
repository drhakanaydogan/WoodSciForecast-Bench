from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIGURES_DIR


def set_publication_style() -> None:
    plt.rcParams["figure.dpi"] = 180
    plt.rcParams["savefig.dpi"] = 600
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 11
    plt.rcParams["axes.labelsize"] = 10.5
    plt.rcParams["legend.fontsize"] = 9
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def save_figure(fig: plt.Figure, base_name: str, output_dir: Path | None = None) -> None:
    destination = output_dir or FIGURES_DIR
    destination.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination / f"{base_name}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(destination / f"{base_name}.tiff", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(destination / f"{base_name}.pdf", bbox_inches="tight", facecolor="white")


def normalize_panel_name(value: object) -> object:
    if pd.isna(value):
        return value
    text = str(value).strip()
    lower = text.lower()
    if "trade" in lower:
        return "Trade_8c"
    if "production" in lower and "8" in lower:
        return "Production_8c"
    if "production" in lower and "12" in lower:
        return "Production_12c"
    if lower == "production":
        return "Production_12c"
    return text


def normalize_target_name(value: object) -> object:
    if pd.isna(value):
        return value
    text = str(value).strip()
    lower = text.lower()
    if "c16" in lower:
        return "C16"
    if "c31" in lower:
        return "C31"
    if "export" in lower:
        return "Exports"
    if "import" in lower:
        return "Imports"
    return text


def normalize_model_name(value: object) -> object:
    if pd.isna(value):
        return value
    text = str(value).strip()
    alias_map = {
        "LightGBM_fallback": "LightGBM",
        "lightgbm_fallback": "LightGBM",
        "LightGBM fallback": "LightGBM",
        "SeasonalNaive": "Seasonal Naive",
        "seasonal_naive": "Seasonal Naive",
        "seasonal naive": "Seasonal Naive",
        "ETS_auto": "ETS",
        "Theta_auto": "Theta",
    }
    return alias_map.get(text, text)


def pretty_block_label(value: str) -> str:
    mapping = {
        "Production_12c | C16": "Production (12c) | C16",
        "Production_12c | C31": "Production (12c) | C31",
        "Production_8c | C16": "Production sensitivity (8c) | C16",
        "Production_8c | C31": "Production sensitivity (8c) | C31",
        "Trade_8c | Exports": "Trade (8c) | Exports",
        "Trade_8c | Imports": "Trade (8c) | Imports",
    }
    return mapping.get(value, value)
