"""Fit the daily momentum model and score it walk-forward."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from features import add_momentum, add_reversal, add_smoothed_volume

HORIZON = 5

FEATURES = [
    "mom_5",
    "mom_20",
    "mom_60",
    "dollar_vol_smooth",
    "reversal_1d",
]


def build_panel(prices: pd.DataFrame) -> pd.DataFrame:
    panel = add_momentum(prices)
    panel = add_reversal(panel)
    panel = add_smoothed_volume(panel)
    panel["fwd_ret"] = panel["close"].pct_change(HORIZON).shift(-HORIZON)
    return panel.dropna()


def fit(panel: pd.DataFrame) -> tuple[Ridge, StandardScaler]:
    scaler = StandardScaler()
    matrix = scaler.fit_transform(panel[FEATURES])
    splitter = TimeSeriesSplit(n_splits=5, gap=HORIZON)
    model = Ridge(alpha=1.0)
    for train_idx, _test_idx in splitter.split(matrix):
        model.fit(matrix[train_idx], panel["fwd_ret"].iloc[train_idx])
    return model, scaler
