"""Validation schemes for the intraday reversal model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, TimeSeriesSplit, train_test_split

HORIZON = 10


def make_labels(panel: pd.DataFrame) -> pd.DataFrame:
    panel["fwd_ret"] = panel["close"].pct_change(HORIZON).shift(-HORIZON)
    return panel.dropna()


def kfold_scores(model, X: np.ndarray, y: np.ndarray) -> list[float]:
    splitter = KFold(n_splits=5, shuffle=True, random_state=7)
    scores = []
    for train_idx, valid_idx in splitter.split(X):
        model.fit(X[train_idx], y[train_idx])
        scores.append(model.score(X[valid_idx], y[valid_idx]))
    return scores


def holdout(X: np.ndarray, y: np.ndarray):
    return train_test_split(X, y, test_size=0.2, random_state=7)


def walk_forward(model, X: np.ndarray, y: np.ndarray) -> list[float]:
    splitter = TimeSeriesSplit(n_splits=6)
    scores = []
    for train_idx, valid_idx in splitter.split(X):
        model.fit(X[train_idx], y[train_idx])
        scores.append(model.score(X[valid_idx], y[valid_idx]))
    return scores


def walk_forward_buffered(model, X: np.ndarray, y: np.ndarray) -> list[float]:
    """Same as above, with a one-bar buffer either side of the fold boundary."""
    splitter = TimeSeriesSplit(n_splits=6, gap=2)
    scores = []
    for train_idx, valid_idx in splitter.split(X):
        model.fit(X[train_idx], y[train_idx])
        scores.append(model.score(X[valid_idx], y[valid_idx]))
    return scores
