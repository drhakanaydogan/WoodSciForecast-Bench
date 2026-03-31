from __future__ import annotations

import pandas as pd

from src.config import BENCHMARK_PROTOCOL_FILE, PROD_SPLITS, PROD_TARGETS, ROUND2_RESULTS_FILE, TRADE_SPLITS, TRADE_TARGETS
from src.evaluation import compute_metrics, country_baseline_eval, summarize_results
from src.io import get_fao_columns, load_benchmark_panels
from src.lightgbm_pipeline import pooled_lgbm_eval


def main() -> None:
    prod12, prod8, trade8 = load_benchmark_panels(BENCHMARK_PROTOCOL_FILE)

    all_metrics = []
    all_predictions = []
    run_logs: list[dict[str, object]] = []

    prod12_fao = get_fao_columns(prod12)
    prod8_fao = get_fao_columns(prod8)
    trade8_fao = get_fao_columns(trade8)

    for target, label in PROD_TARGETS.items():
        for split_id, start, end in PROD_SPLITS:
            for baseline in ['Seasonal Naive', 'ETS']:
                pred_df, train_cache = country_baseline_eval(prod12, target, start, end, baseline)
                pred_df['panel'] = 'Production_12c'
                pred_df['split_id'] = split_id
                pred_df['target'] = label
                all_predictions.append(pred_df)
                all_metrics.append(compute_metrics(pred_df, train_cache, split_id, 'Production_12c', label))

            pred_df, train_cache = pooled_lgbm_eval(prod12, target, prod12_fao, start, end, 'Production_12c', split_id, run_logs)
            pred_df['panel'] = 'Production_12c'
            pred_df['split_id'] = split_id
            pred_df['target'] = label
            all_predictions.append(pred_df)
            all_metrics.append(compute_metrics(pred_df, train_cache, split_id, 'Production_12c', label))

    for target, label in PROD_TARGETS.items():
        for split_id, start, end in PROD_SPLITS:
            for baseline in ['Seasonal Naive', 'ETS']:
                pred_df, train_cache = country_baseline_eval(prod8, target, start, end, baseline)
                pred_df['panel'] = 'Production_8c'
                pred_df['split_id'] = split_id
                pred_df['target'] = label
                all_predictions.append(pred_df)
                all_metrics.append(compute_metrics(pred_df, train_cache, split_id, 'Production_8c', label))

            pred_df, train_cache = pooled_lgbm_eval(prod8, target, prod8_fao, start, end, 'Production_8c', split_id, run_logs)
            pred_df['panel'] = 'Production_8c'
            pred_df['split_id'] = split_id
            pred_df['target'] = label
            all_predictions.append(pred_df)
            all_metrics.append(compute_metrics(pred_df, train_cache, split_id, 'Production_8c', label))

    for target, label in TRADE_TARGETS.items():
        for split_id, start, end in TRADE_SPLITS:
            for baseline in ['Seasonal Naive', 'ETS']:
                pred_df, train_cache = country_baseline_eval(trade8, target, start, end, baseline)
                pred_df['panel'] = 'Trade_8c'
                pred_df['split_id'] = split_id
                pred_df['target'] = label
                all_predictions.append(pred_df)
                all_metrics.append(compute_metrics(pred_df, train_cache, split_id, 'Trade_8c', label))

            pred_df, train_cache = pooled_lgbm_eval(trade8, target, trade8_fao, start, end, 'Trade_8c', split_id, run_logs)
            pred_df['panel'] = 'Trade_8c'
            pred_df['split_id'] = split_id
            pred_df['target'] = label
            all_predictions.append(pred_df)
            all_metrics.append(compute_metrics(pred_df, train_cache, split_id, 'Trade_8c', label))

    metrics = pd.concat(all_metrics, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    run_logs_df = pd.DataFrame(run_logs)

    outputs = summarize_results(metrics)
    summary = outputs['summary']
    block_winners = outputs['block_winners']
    model_win_counts = outputs['model_win_counts']
    global_means = outputs['global_means']

    relative_perf = summary.pivot_table(index=['panel', 'target', 'split_id'], columns='model', values='mean_mase').reset_index()
    if 'LightGBM' in relative_perf.columns and 'Seasonal Naive' in relative_perf.columns:
        relative_perf['lgbm_vs_snaive_pct'] = 100 * (relative_perf['LightGBM'] / relative_perf['Seasonal Naive'] - 1)
    if 'LightGBM' in relative_perf.columns and 'ETS' in relative_perf.columns:
        relative_perf['lgbm_vs_ets_pct'] = 100 * (relative_perf['LightGBM'] / relative_perf['ETS'] - 1)
    if 'LightGBM_fallback' in relative_perf.columns and 'Seasonal Naive' in relative_perf.columns:
        relative_perf['lgbm_fallback_vs_snaive_pct'] = 100 * (relative_perf['LightGBM_fallback'] / relative_perf['Seasonal Naive'] - 1)

    with pd.ExcelWriter(ROUND2_RESULTS_FILE, engine='xlsxwriter') as writer:
        summary.to_excel(writer, sheet_name='summary_by_split_target', index=False)
        block_winners.to_excel(writer, sheet_name='block_winners', index=False)
        model_win_counts.to_excel(writer, sheet_name='model_win_counts', index=False)
        relative_perf.to_excel(writer, sheet_name='lgbm_relative_perf', index=False)
        metrics.to_excel(writer, sheet_name='country_level_metrics', index=False)
        predictions.to_excel(writer, sheet_name='all_predictions', index=False)
        global_means.to_excel(writer, sheet_name='global_means', index=False)
        run_logs_df.to_excel(writer, sheet_name='run_logs', index=False)

    print(f'Round 2 results written to: {ROUND2_RESULTS_FILE}')


if __name__ == '__main__':
    main()
