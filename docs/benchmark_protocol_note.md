# Benchmark protocol note

This repository accompanies the benchmark study built around nested, regime-aware forecasting tasks for wood-related production and trade series.

## Core benchmark structure

The benchmark is organized into three main panel blocks:

1. `Production_12c` — main 12-country production benchmark
2. `Production_8c` — production sensitivity benchmark aligned to the reduced trade-country scope
3. `Trade_8c` — main 8-country trade benchmark

## Evaluation logic

The split architecture is chronological and regime-aware.

### Production splits
- `S1`: 2015-01 to 2019-12
- `S2`: 2020-01 to 2021-12
- `S3`: 2022-01 to 2024-12

### Trade splits
- `T1`: 2014-01 to 2019-12
- `T2`: 2020-01 to 2021-12
- `T3`: 2022-01 to 2024-12

For each panel × target × split × model combination, the model is fit once using all observations dated before the split start and is then evaluated over the full split block. The protocol therefore uses fixed-origin block forecasting within split windows.

## Model families

Round 2 benchmark:
- Seasonal Naive
- ETS
- pooled LightGBM

Round 3 trade extension:
- Seasonal Naive
- ETS
- pooled LightGBM
- LSTM

## Fallback rule

The pooled LightGBM pipeline includes an explicit fallback rule. Fallback is triggered when:

- the effective train or test design becomes empty after feature construction and missing-value filtering,
- the aligned feature matrix collapses, or
- model fitting or prediction raises an exception.

In these cases, predictions default to a country-wise last-observed-value repeat rule across the test block and are labeled `LightGBM_fallback`.

## Reproducibility note

The repository is designed around processed benchmark files derived from public Eurostat and FAOSTAT sources. To reproduce the manuscript outputs, place the processed `.xlsx` files in `data/processed/` and run the scripts in `src/`.
