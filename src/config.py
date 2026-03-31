from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
OUTPUT_DIR = PROJECT_ROOT / 'outputs'
TABLES_DIR = OUTPUT_DIR / 'tables'
FIGURES_DIR = OUTPUT_DIR / 'figures'

BENCHMARK_PROTOCOL_FILE = DATA_DIR / 'woodsciforecast_benchmark_protocol_v1.xlsx'
ROUND1_RESULTS_FILE = DATA_DIR / 'woodsciforecast_first_baseline_results_v1.xlsx'
ROUND2_RESULTS_FILE = DATA_DIR / 'woodsciforecast_second_round_results_v2.xlsx'
ROUND3_RESULTS_FILE = DATA_DIR / 'woodsciforecast_round3_trade_deep_results_v1.xlsx'

PROD_SPLITS = [
    ('S1', '2015-01-01', '2019-12-01'),
    ('S2', '2020-01-01', '2021-12-01'),
    ('S3', '2022-01-01', '2024-12-01'),
]

TRADE_SPLITS = [
    ('T1', '2014-01-01', '2019-12-01'),
    ('T2', '2020-01-01', '2021-12-01'),
    ('T3', '2022-01-01', '2024-12-01'),
]

PROD_TARGETS = {
    'ln_sts_c16_sa_idx2021_100': 'Production C16',
    'ln_sts_c31_sa_idx2021_100': 'Production C31',
}

TRADE_TARGETS = {
    'ln_trade_world_export_eur_hs4sum': 'Trade export',
    'ln_trade_world_import_eur_hs4sum': 'Trade import',
}

PROD12_SHEET = 'production_panel_12c'
PROD8_SHEET = 'production_panel_8c'
TRADE8_SHEET = 'trade_panel_8c'

LIGHTGBM_PARAMS = {
    'n_estimators': 120,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'random_state': 42,
    'objective': 'regression',
    'verbose': -1,
}

LSTM_LOOKBACK = 12
LSTM_EPOCHS = 40
LSTM_BATCH_SIZE = 32
LSTM_VALIDATION_SPLIT = 0.2
LSTM_PATIENCE = 5
LSTM_RANDOM_SEED = 42

for path in (DATA_DIR, TABLES_DIR, FIGURES_DIR):
    path.mkdir(parents=True, exist_ok=True)
