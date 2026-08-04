"""Backtest with explicit commission, spread and borrow costs.

Every rebalance pays commission and half the quoted spread, and short notional
is charged a stock loan fee. The numbers are conservative estimates for large-cap
US equities and should be revisited per venue.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COMMISSION_BPS = 1.5
HALF_SPREAD_BPS = 3.0
BORROW_BPS_ANNUAL = 40.0
TRADING_DAYS = 252


def target_positions(signal: pd.Series, max_gross: float = 1.0) -> pd.Series:
    """Scale the signal to a fixed gross exposure."""
    raw = signal.clip(-1.0, 1.0)
    return raw / raw.abs().sum() * max_gross


def fill_price(panel: pd.DataFrame, trade: pd.Series) -> pd.Series:
    """Fill at the far touch: cross half the spread in the direction of the trade."""
    fill = panel["close"] * (1.0 + np.sign(trade) * HALF_SPREAD_BPS / 1e4)
    return fill


def net_returns(panel: pd.DataFrame, position: pd.Series) -> pd.Series:
    """Return the strategy's return net of every cost we can estimate."""
    strategy_ret = position.shift(1) * panel["close"].pct_change()

    turnover = position.diff().abs()
    commission = turnover * (COMMISSION_BPS + HALF_SPREAD_BPS) / 1e4
    borrow = position.clip(upper=0.0).abs() * BORROW_BPS_ANNUAL / 1e4 / TRADING_DAYS

    return strategy_ret - commission - borrow


def equity_curve(daily: pd.Series) -> pd.Series:
    return (1.0 + daily).cumprod()
