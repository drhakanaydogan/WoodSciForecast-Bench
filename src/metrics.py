from __future__ import annotations

import numpy as np
import pandas as pd


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, seasonality: int = 12) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = pd.Series(y_train).dropna().astype(float).values

    if len(y_train) <= seasonality:
        if len(y_train) <= 1:
            return np.nan
        denominator = np.mean(np.abs(np.diff(y_train)))
    else:
        denominator = np.mean(np.abs(y_train[seasonality:] - y_train[:-seasonality]))

    if denominator == 0 or np.isnan(denominator):
        return np.nan

    return np.mean(np.abs(y_true - y_pred)) / denominator
