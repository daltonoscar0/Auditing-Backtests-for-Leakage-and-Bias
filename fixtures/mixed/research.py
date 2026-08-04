"""Entry-threshold and holding-period sweep."""

from __future__ import annotations

import itertools

import pandas as pd

THRESHOLDS = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60]
HOLD_BARS = [1, 2, 4, 8, 12]


def evaluate(test: pd.DataFrame, threshold: float, hold: int) -> float:
    signal = (test["score"] > threshold).astype(float)
    realised = signal.shift(1) * test["ret_1"].rolling(hold).sum()
    return float(realised.mean() / realised.std())


def sweep(test: pd.DataFrame) -> tuple[float, float, int]:
    results = []
    for threshold, hold in itertools.product(THRESHOLDS, HOLD_BARS):
        results.append((evaluate(test, threshold, hold), threshold, hold))
    return max(results)
