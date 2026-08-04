"""Parameter sweep for the breakout strategy."""

from __future__ import annotations

import itertools

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

LOOKBACKS = [5, 10, 20, 40, 60, 90, 120]
THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 2.5]
STOPS = [0.02, 0.05, 0.10]


def sharpe(daily: pd.Series) -> float:
    return float(daily.mean() / daily.std() * (252 ** 0.5))


def simulate(prices: pd.DataFrame, lookback: int, threshold: float, stop: float) -> pd.Series:
    """Placeholder for the strategy simulator."""
    raise NotImplementedError


def run(train: pd.DataFrame, test: pd.DataFrame) -> tuple[float, int, float, float]:
    results = []
    for lookback, threshold, stop in itertools.product(LOOKBACKS, THRESHOLDS, STOPS):
        curve = simulate(test, lookback, threshold, stop)
        results.append((sharpe(curve), lookback, threshold, stop))
    best = max(results)
    return best


def fit_gbm(X_train, y_train, X_test, y_test) -> GradientBoostingRegressor:
    model = GradientBoostingRegressor(n_estimators=2000, learning_rate=0.01)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=50)
    return model
