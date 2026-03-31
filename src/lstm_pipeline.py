from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

from src.config import (
    LSTM_BATCH_SIZE,
    LSTM_EPOCHS,
    LSTM_LOOKBACK,
    LSTM_PATIENCE,
    LSTM_RANDOM_SEED,
    LSTM_VALIDATION_SPLIT,
)


def build_lstm_sequences(df: pd.DataFrame, target: str, lookback: int, split_start: str, split_end: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    start = pd.Timestamp(split_start)
    end = pd.Timestamp(split_end)

    sequence_rows = []
    train_cache: dict[str, np.ndarray] = {}

    grouped = df[['country', 'date', target]].dropna().sort_values('date').groupby('country')
    for country, group in grouped:
        y = group[target].values
        dates = group['date'].values
        train_cache[country] = group[group['date'] < start][target].values

        for i in range(lookback, len(group)):
            sequence_rows.append({
                'country': country,
                'date': pd.Timestamp(dates[i]),
                'y_true': y[i],
                'seq': y[i - lookback:i],
            })

    sequence_df = pd.DataFrame(sequence_rows)
    train_df = sequence_df[sequence_df['date'] < start].copy()
    test_df = sequence_df[(sequence_df['date'] >= start) & (sequence_df['date'] <= end)].copy()
    return train_df, test_df, train_cache


def lstm_eval(
    df: pd.DataFrame,
    target: str,
    split_start: str,
    split_end: str,
    lookback: int = LSTM_LOOKBACK,
    epochs: int = LSTM_EPOCHS,
    batch_size: int = LSTM_BATCH_SIZE,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    train_df, test_df, train_cache = build_lstm_sequences(df, target, lookback, split_start, split_end)

    if train_df.empty or test_df.empty:
        return pd.DataFrame(columns=['country', 'date', 'y_true', 'y_pred', 'model']), {}

    x_train = np.stack(train_df['seq'].values)
    x_test = np.stack(test_df['seq'].values)
    y_train = train_df['y_true'].values

    scaler = StandardScaler()
    x_train_2d = x_train.reshape(x_train.shape[0], x_train.shape[1])
    x_test_2d = x_test.reshape(x_test.shape[0], x_test.shape[1])

    x_train_scaled = scaler.fit_transform(x_train_2d).reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test_scaled = scaler.transform(x_test_2d).reshape(x_test.shape[0], x_test.shape[1], 1)

    tf.keras.backend.clear_session()
    tf.random.set_seed(LSTM_RANDOM_SEED)
    np.random.seed(LSTM_RANDOM_SEED)

    model = Sequential([
        LSTM(32, input_shape=(lookback, 1), return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1),
    ])
    model.compile(optimizer='adam', loss='mse')

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=LSTM_PATIENCE,
        restore_best_weights=True,
    )

    model.fit(
        x_train_scaled,
        y_train,
        validation_split=LSTM_VALIDATION_SPLIT,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[early_stopping],
    )

    predictions = model.predict(x_test_scaled, verbose=0).flatten()

    pred_df = test_df[['country', 'date', 'y_true']].copy()
    pred_df['y_pred'] = predictions
    pred_df['model'] = 'LSTM'
    return pred_df, train_cache
