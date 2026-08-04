"""Universe construction for the cross-sectional value model."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA = Path("data")

TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "JPM", "XOM"]

MIN_DOLLAR_VOLUME = 1e9


def current_constituents() -> list[str]:
    """Index membership, read from the file shipped alongside this repo."""
    members = pd.read_csv(DATA / "sp500.csv")
    return members["ticker"].tolist()


def load_panel() -> pd.DataFrame:
    panel = pd.read_csv(DATA / "prices.csv", parse_dates=["date"])
    return panel[panel["ticker"].isin(current_constituents())]


def drop_dead_names(panel: pd.DataFrame) -> pd.DataFrame:
    """Keep only the names we can still get a quote for."""
    return panel[panel["is_active"]]


def liquid_names(panel: pd.DataFrame) -> pd.DataFrame:
    """Screen out names too small to trade at size."""
    return panel[panel.groupby("ticker")["dollar_volume"].transform("sum") > MIN_DOLLAR_VOLUME]


def fallback_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Used when the membership file is unavailable."""
    return panel[panel["ticker"].isin(TICKERS)]
