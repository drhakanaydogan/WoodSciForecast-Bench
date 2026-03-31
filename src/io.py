from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from src.config import PROD12_SHEET, PROD8_SHEET, TRADE8_SHEET


def prepare_panel(df: pd.DataFrame) -> pd.DataFrame:
    panel = df.copy()
    panel['date'] = pd.to_datetime(panel['date'])
    panel['year'] = panel['date'].dt.year
    panel['month'] = panel['date'].dt.month
    return panel


def load_benchmark_panels(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prod12 = prepare_panel(pd.read_excel(path, sheet_name=PROD12_SHEET))
    prod8 = prepare_panel(pd.read_excel(path, sheet_name=PROD8_SHEET))
    trade8 = prepare_panel(pd.read_excel(path, sheet_name=TRADE8_SHEET))
    return prod12, prod8, trade8


def load_trade_panel(path: Path) -> pd.DataFrame:
    return prepare_panel(pd.read_excel(path, sheet_name=TRADE8_SHEET))


def get_fao_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column.startswith('fao_')]
