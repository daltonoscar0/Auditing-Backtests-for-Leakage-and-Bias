"""How orders are assumed to fill, and what that assumption charges.

Kept in one module so the backtest, the capacity study and the live sizing code
all share the same numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COMMISSION_BPS = 1.0
HALF_SPREAD_BPS = 2.5
BORROW_BPS_ANNUAL = 35.0
TRADING_DAYS = 252


def round_trip_bps() -> float:
    return COMMISSION_BPS + HALF_SPREAD_BPS


def net_of_charges(gross: pd.Series, weight: pd.Series) -> pd.Series:
    """Charge turnover at the round-trip rate, and short notional a loan fee."""
    turnover = weight.diff().abs()
    commission = turnover * round_trip_bps() / 1e4
    borrow = weight.clip(upper=0.0).abs() * BORROW_BPS_ANNUAL / 1e4 / TRADING_DAYS
    return gross - commission - borrow


def execution_price(price: pd.Series, trade: pd.Series) -> pd.Series:
    """Fill at the far touch, crossing half the quoted spread."""
    return price * (1.0 + np.sign(trade) * HALF_SPREAD_BPS / 1e4)
