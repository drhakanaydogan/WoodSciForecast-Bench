from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.config import LIGHTGBM_PARAMS
from src.features import build_lightgbm_features


def safe_last_value_fallback(train_values: np.ndarray, test_values: np.ndarray) -> np.ndarray:
    if len(train_values) == 0:
        return np.full(len(test_values), np.nan)
    last_value = pd.Series(train_values).dropna().iloc[-1]
    return np.repeat(last_value, len(test_values))


def _build_fallback_predictions(df: pd.DataFrame, target: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows: list[pd.DataFrame] = []
    train_cache: dict[str, np.ndarray] = {}

    grouped = df[['country', 'date', target]].dropna().sort_values('date').groupby('country')
    for country, group in grouped:
        train_values = group[group['date'] < start][target].values
        test_values = group[(group['date'] >= start) & (group['date'] <= end)][target].values
        test_dates = group[(group['date'] >= start) & (group['date'] <= end)]['date'].values

        if len(train_values) == 0 or len(test_values) == 0:
            continue

        predictions = safe_last_value_fallback(train_values, test_values)
        rows.append(pd.DataFrame({
            'country': country,
            'date': pd.to_datetime(test_dates),
            'y_true': test_values,
            'y_pred': predictions,
            'model': 'LightGBM_fallback',
        }))
        train_cache[country] = train_values

    if not rows:
        return pd.DataFrame(columns=['country', 'date', 'y_true', 'y_pred', 'model']), {}

    return pd.concat(rows, ignore_index=True), train_cache


def pooled_lgbm_eval(
    df: pd.DataFrame,
    target: str,
    fao_columns: list[str],
    split_start: str,
    split_end: str,
    panel_name: str,
    split_id: str,
    logs: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    data, feature_columns = build_lightgbm_features(df, target, fao_columns)
    start = pd.Timestamp(split_start)
    end = pd.Timestamp(split_end)

    train = data[data['date'] < start].copy()
    test = data[(data['date'] >= start) & (data['date'] <= end)].copy()

    train = train[train[target].notna()].copy()
    test = test[test[target].notna()].copy()

    usable_features = []
    for column in feature_columns:
        if column == 'country':
            usable_features.append(column)
        elif column in train.columns and train[column].notna().sum() > 0:
            usable_features.append(column)

    train = train.dropna(subset=usable_features + [target]).copy()
    test = test.dropna(subset=usable_features + [target]).copy()

    if train.shape[0] == 0 or test.shape[0] == 0:
        logs.append({
            'panel': panel_name,
            'split_id': split_id,
            'target': target,
            'model': 'LightGBM',
            'status': 'fallback_empty_train_or_test',
            'train_rows': train.shape[0],
            'test_rows': test.shape[0],
        })
        return _build_fallback_predictions(df, target, start, end)

    x_train = pd.get_dummies(train[usable_features], columns=['country'], drop_first=False)
    x_test = pd.get_dummies(test[usable_features], columns=['country'], drop_first=False)
    x_train, x_test = x_train.align(x_test, join='left', axis=1, fill_value=0)

    if x_train.shape[0] == 0 or x_train.shape[1] == 0 or x_test.shape[0] == 0:
        logs.append({
            'panel': panel_name,
            'split_id': split_id,
            'target': target,
            'model': 'LightGBM',
            'status': 'fallback_empty_matrix',
            'train_rows': x_train.shape[0],
            'test_rows': x_test.shape[0],
            'n_features': x_train.shape[1] if x_train.ndim == 2 else 0,
        })
        return _build_fallback_predictions(df, target, start, end)

    try:
        model = LGBMRegressor(**LIGHTGBM_PARAMS)
        model.fit(x_train, train[target])
        predictions = model.predict(x_test)

        train_cache = {
            country: group.loc[group['date'] < start, target].dropna().values
            for country, group in df[['country', 'date', target]].groupby('country')
        }

        pred_df = test[['country', 'date']].copy()
        pred_df['y_true'] = test[target].values
        pred_df['y_pred'] = predictions
        pred_df['model'] = 'LightGBM'

        logs.append({
            'panel': panel_name,
            'split_id': split_id,
            'target': target,
            'model': 'LightGBM',
            'status': 'success',
            'train_rows': x_train.shape[0],
            'test_rows': x_test.shape[0],
            'n_features': x_train.shape[1],
        })
        return pred_df, train_cache

    except Exception as exc:
        logs.append({
            'panel': panel_name,
            'split_id': split_id,
            'target': target,
            'model': 'LightGBM',
            'status': f'fallback_exception: {str(exc)}',
            'train_rows': x_train.shape[0],
            'test_rows': x_test.shape[0],
            'n_features': x_train.shape[1],
        })
        return _build_fallback_predictions(df, target, start, end)
