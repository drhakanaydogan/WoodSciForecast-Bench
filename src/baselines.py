from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def seasonal_naive_forecast(train_series: np.ndarray, horizon: int, season: int = 12) -> np.ndarray:
    history = list(train_series)
    predictions = []
    for _ in range(horizon):
        prediction = history[-season] if len(history) >= season else history[-1]
        predictions.append(prediction)
        history.append(prediction)
    return np.asarray(predictions)


def ets_forecast(train_series: np.ndarray, horizon: int) -> np.ndarray:
    y = pd.Series(train_series).astype(float)
    if len(y) < 24:
        return np.repeat(y.iloc[-1], horizon)

    try:
        fit = ExponentialSmoothing(
            y,
            trend='add',
            seasonal='add',
            seasonal_periods=12,
            damped_trend=True,
        ).fit(optimized=True, use_brute=False)
        return np.asarray(fit.forecast(horizon))
    except Exception:
        return np.repeat(y.iloc[-1], horizon)
