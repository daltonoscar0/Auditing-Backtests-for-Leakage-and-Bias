"""Point-in-time universe construction.

Membership is keyed by (ticker, date) and includes names that were later
delisted or removed from the index, so the sample at any date is the sample a
person standing at that date would have had.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA = Path("data")
ADV_WINDOW = 20
MIN_ADV_USD = 5e6


def membership() -> pd.DataFrame:
    """Index membership by date, including entries that have since been removed."""
    return pd.read_csv(DATA / "constituents_pit.csv", parse_dates=["date"])


def load_panel() -> pd.DataFrame:
    """Prices for every name that ever traded, delistings included."""
    panel = pd.read_csv(DATA / "prices.csv", parse_dates=["date"])
    return panel.sort_values(["ticker", "date"])


def restrict_to_members(panel: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    """Keep each row only if that name was a member on that row's date."""
    return panel.merge(members, on=["ticker", "date"], how="inner")


def liquidity_screen(panel: pd.DataFrame) -> pd.DataFrame:
    """Screen on trailing dollar volume, lagged one bar so it is knowable at t."""
    panel["adv_20"] = panel.groupby("ticker")["dollar_volume"].transform(
        lambda s: s.rolling(ADV_WINDOW, min_periods=ADV_WINDOW).mean().shift(1)
    )
    return panel[panel["adv_20"] > MIN_ADV_USD]
