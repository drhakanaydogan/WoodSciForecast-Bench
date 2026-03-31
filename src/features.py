from __future__ import annotations

import numpy as np
import pandas as pd


def build_lightgbm_features(df: pd.DataFrame, target: str, fao_columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    data = df.copy().sort_values(['country', 'date'])

    for lag in [1, 3, 6, 12]:
        data[f'lag{lag}'] = data.groupby('country')[target].shift(lag)

    data['diff1'] = data[target] - data.groupby('country')[target].shift(1)
    data['diff12'] = data[target] - data.groupby('country')[target].shift(12)
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)

    usable_fao = [c for c in fao_columns if c in data.columns and data[c].notna().sum() > 0]

    feature_columns = [
        'country', 'year', 'month_sin', 'month_cos',
        'lag1', 'lag3', 'lag6', 'lag12',
        'diff1', 'diff12',
    ] + usable_fao

    return data, feature_columns
