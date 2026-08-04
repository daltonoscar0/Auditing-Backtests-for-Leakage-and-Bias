"""Quarterly fundamentals blended into the daily equity panel.

Nothing in this module is a mechanical red flag. Every window is trailing, every
fill carries the past forward, the splitter is a grouped one rather than a
shuffled one, and there is a cost model. The defects are in what the column names
mean and in what the join is doing, which is a reading-comprehension problem
rather than a pattern-matching one.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupKFold

COST_BPS = 2.0
VOL_WINDOW = 60


def load_fundamentals(path: str) -> pd.DataFrame:
    """Quarterly reports.

    `period_end` is the last day of the quarter the numbers describe.
    `filing_date` is the day the report was actually released, typically six to
    nine weeks later.
    """
    return pd.read_csv(path, parse_dates=["period_end", "filing_date"])


def attach_fundamentals(panel: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    out = panel.merge(
        fundamentals,
        left_on=["ticker", "date"],
        right_on=["ticker", "period_end"],
        how="left",
    )
    out["earnings_yield"] = out["net_income"] / out["market_cap"]
    return out.sort_values(["ticker", "date"]).ffill()


def attach_sector(panel: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """`reference` is one row per ticker: sector, country, listing venue."""
    return panel.merge(reference, on="ticker", how="left")


def add_label(panel: pd.DataFrame) -> pd.DataFrame:
    """Label the session by its own move, from its own open to its own close."""
    panel["intraday_move"] = panel["close"] / panel["open"] - 1.0
    return panel


def add_features(panel: pd.DataFrame) -> pd.DataFrame:
    panel["overnight_gap"] = panel["open"] / panel["close"].shift(1) - 1.0
    panel["realised_vol"] = panel["close"].pct_change().rolling(VOL_WINDOW).std()
    panel["close_to_close"] = panel["close"].pct_change()
    return panel


def split(panel: pd.DataFrame, n_splits: int = 5):
    """Keep each name whole so a ticker never appears in both folds."""
    splitter = GroupKFold(n_splits=n_splits)
    return splitter.split(panel, groups=panel["ticker"])


def size_orders(panel: pd.DataFrame, weight: pd.Series, capital: float) -> pd.Series:
    """Convert target weights into share counts at the closing price."""
    notional = weight * capital
    return notional / panel["close"]


def net_return(panel: pd.DataFrame, weight: pd.Series) -> pd.Series:
    gross = weight.shift(1) * panel["close_to_close"]
    turnover = weight.diff().abs()
    return gross - turnover * COST_BPS / 1e4
