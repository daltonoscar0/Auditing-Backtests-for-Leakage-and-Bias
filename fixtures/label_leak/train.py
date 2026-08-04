"""Train the three-day return forecaster."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

HORIZON = 3
SPLIT_DATE = "2021-01-01"

FEATURES = [
    "vol_20",
    "mom_10",
    "signal_strength",
    "fwd_ret_3d_ma",
]


def add_label(panel: pd.DataFrame) -> pd.DataFrame:
    panel["fwd_ret_3d"] = panel["close"].pct_change(HORIZON).shift(-HORIZON)
    return panel


def add_features(panel: pd.DataFrame) -> pd.DataFrame:
    panel["vol_20"] = panel["close"].pct_change().rolling(20).std()
    panel["mom_10"] = panel["close"].pct_change(10)
    panel["signal_strength"] = panel["fwd_ret_3d"].rolling(5).mean()
    panel["fwd_ret_3d_ma"] = panel["fwd_ret_3d"].rolling(10).mean()
    return panel


def design_matrix(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = panel.drop(columns=["date", "ticker", "close"])
    y = panel["fwd_ret_3d"]
    return X, y


def fit(panel: pd.DataFrame) -> GradientBoostingRegressor:
    train = panel[panel["date"] < SPLIT_DATE]
    scaler = StandardScaler()
    scaler.fit(panel)
    model = GradientBoostingRegressor(n_estimators=200, max_depth=3)
    model.fit(scaler.transform(train[FEATURES]), train["fwd_ret_3d"])
    return model
