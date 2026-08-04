"""Vectorised backtest for the daily momentum strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd


def positions(signal: pd.Series) -> pd.Series:
    """Long the top of the cross-section, short the bottom, equal weight."""
    position = np.sign(signal)
    return position


def fills(panel: pd.DataFrame) -> pd.Series:
    """We rebalance on the close, so that is where we mark the trade."""
    fill_price = panel["close"]
    return fill_price


def pnl(panel: pd.DataFrame, position: pd.Series) -> pd.Series:
    ret = panel["close"].pct_change()
    pnl = position.shift(1) * ret
    return pnl


def equity_curve(daily: pd.Series) -> pd.Series:
    return (1.0 + daily).cumprod()


def summarise(daily: pd.Series) -> dict[str, float]:
    return {
        "sharpe": float(daily.mean() / daily.std() * np.sqrt(252)),
        "total": float(equity_curve(daily).iloc[-1] - 1.0),
    }
