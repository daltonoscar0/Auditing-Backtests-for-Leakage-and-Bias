"""Features for the ten-day forward return model.

Every statistic here is computed from bars at or before the row it labels. Where
a normalisation is needed it uses an expanding window rather than a full-sample
one, so a row's z-score depends only on that row's own past.
"""

from __future__ import annotations

import pandas as pd

MOMENTUM_WINDOWS = (5, 20, 60)
VOL_WINDOW = 20
MIN_HISTORY = 252

FEATURE_COLUMNS = [
    "mom_5",
    "mom_20",
    "mom_60",
    "vol_20",
    "book_to_price",
]


def add_momentum(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for window in MOMENTUM_WINDOWS:
        out[f"mom_{window}"] = out["close"].pct_change(window)
    return out


def add_volatility(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["vol_20"] = out["close"].pct_change().rolling(VOL_WINDOW).std()
    return out


def add_fundamentals(frame: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Join fundamentals on the date they were published, then carry them forward."""
    out = frame.merge(fundamentals, on=["ticker", "publication_date"], how="left")
    out["book_to_price"] = (out["book_value"] / out["market_cap"]).ffill()
    return out


def expanding_zscore(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Normalise each column by statistics drawn only from its own past."""
    out = frame.copy()
    for column in columns:
        history = out[column].expanding(min_periods=MIN_HISTORY)
        out[f"{column}_z"] = (out[column] - history.mean()) / history.std()
    return out
