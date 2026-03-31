from __future__ import annotations

import math

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.baselines import ets_forecast, seasonal_naive_forecast
from src.metrics import mase


def country_baseline_eval(df: pd.DataFrame, target: str, split_start: str, split_end: str, model_name: str) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    start = pd.Timestamp(split_start)
    end = pd.Timestamp(split_end)

    rows: list[pd.DataFrame] = []
    train_cache: dict[str, pd.Series] = {}

    grouped = df[['country', 'date', target]].dropna().sort_values('date').groupby('country')
    for country, group in grouped:
        train = group[group['date'] < start][target].values
        test = group[(group['date'] >= start) & (group['date'] <= end)][target].values
        test_dates = group[(group['date'] >= start) & (group['date'] <= end)]['date'].values

        if len(train) == 0 or len(test) == 0:
            continue

        if model_name == 'Seasonal Naive':
            pred = seasonal_naive_forecast(train, len(test), season=12)
        elif model_name == 'ETS':
            pred = ets_forecast(train, len(test))
        else:
            raise ValueError(f'Unknown baseline model: {model_name}')

        rows.append(pd.DataFrame({
            'country': country,
            'date': pd.to_datetime(test_dates),
            'y_true': test,
            'y_pred': pred,
            'model': model_name,
        }))
        train_cache[country] = train

    if not rows:
        return pd.DataFrame(columns=['country', 'date', 'y_true', 'y_pred', 'model']), {}

    return pd.concat(rows, ignore_index=True), train_cache


def compute_metrics(pred_df: pd.DataFrame, train_source: dict[str, pd.Series], split_id: str, panel_name: str, target_label: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    if pred_df.empty:
        return pd.DataFrame(columns=['panel', 'split_id', 'target', 'model', 'country', 'mae', 'rmse', 'mase', 'n_test'])

    for country, group in pred_df.groupby('country'):
        if country not in train_source or len(train_source[country]) == 0:
            continue

        y_true = group['y_true'].values
        y_pred = group['y_pred'].values
        y_train = train_source[country]

        rows.append({
            'panel': panel_name,
            'split_id': split_id,
            'target': target_label,
            'model': group['model'].iloc[0],
            'country': country,
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': math.sqrt(mean_squared_error(y_true, y_pred)),
            'mase': mase(y_true, y_pred, y_train, seasonality=12),
            'n_test': len(y_true),
        })

    return pd.DataFrame(rows)


def summarize_results(metrics: pd.DataFrame) -> dict[str, pd.DataFrame]:
    summary = (
        metrics.groupby(['panel', 'target', 'split_id', 'model'], as_index=False)
        .agg(
            mean_mae=('mae', 'mean'),
            mean_rmse=('rmse', 'mean'),
            mean_mase=('mase', 'mean'),
            countries=('country', 'nunique'),
        )
    )

    block_winners = (
        summary.sort_values(['panel', 'target', 'split_id', 'mean_mase'])
        .groupby(['panel', 'target', 'split_id'], as_index=False)
        .first()
    )

    model_win_counts = (
        block_winners.groupby(['panel', 'model'], as_index=False)
        .size()
        .rename(columns={'size': 'n_wins'})
        .sort_values(['panel', 'n_wins'], ascending=[True, False])
    )

    global_means = (
        summary.groupby(['panel', 'model'], as_index=False)['mean_mase']
        .mean()
        .sort_values(['panel', 'mean_mase'])
    )

    return {
        'summary': summary,
        'block_winners': block_winners,
        'model_win_counts': model_win_counts,
        'global_means': global_means,
    }
