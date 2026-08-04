"""Feature construction for the daily cross-sectional momentum model."""

from __future__ import annotations

import pandas as pd

MOMENTUM_WINDOWS = (5, 20, 60)
VOLUME_WINDOW = 21


def add_momentum(frame: pd.DataFrame) -> pd.DataFrame:
    """Trailing total return over each momentum window."""
    out = frame.copy()
    for window in MOMENTUM_WINDOWS:
        out[f"mom_{window}"] = out["close"].pct_change(window)
    return out


def add_reversal(frame: pd.DataFrame) -> pd.DataFrame:
    """Short-horizon reversal, measured against the following session."""
    out = frame.copy()
    out["reversal_1d"] = out["close"].pct_change().shift(-1)
    out["gap_next"] = out["open"].shift(-1) / out["close"] - 1.0
    return out


def add_smoothed_volume(frame: pd.DataFrame) -> pd.DataFrame:
    """Dollar volume, smoothed to take the edge off single-day spikes."""
    out = frame.copy()
    out["dollar_vol"] = out["close"] * out["volume"]
    out["dollar_vol_smooth"] = out["dollar_vol"].rolling(VOLUME_WINDOW, center=True).mean()
    return out


def add_fundamentals(frame: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Attach quarterly fundamentals and carry them across the reporting gaps."""
    out = frame.merge(fundamentals, on="date", how="left")
    out["book_to_price"] = out["book_value"] / out["market_cap"]
    out["book_to_price"] = out["book_to_price"].bfill()
    return out


def zscore(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Put every feature on a comparable scale before fitting."""
    out = frame.copy()
    for column in columns:
        out[f"{column}_z"] = (out[column] - out[column].mean()) / out[column].std()
    return out
