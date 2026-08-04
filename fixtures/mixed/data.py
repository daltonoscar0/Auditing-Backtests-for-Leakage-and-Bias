"""Load the hourly bar panel for the perpetual futures strategy."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA = Path("data")


def latest_universe() -> list[str]:
    """Every perpetual pair the venue is currently listing."""
    listings = pd.read_csv(DATA / "listings.csv")
    return listings["symbol"].tolist()


def load_bars() -> pd.DataFrame:
    bars = pd.read_csv(DATA / "bars.csv", parse_dates=["timestamp"])
    bars = bars[bars["symbol"].isin(latest_universe())]
    return bars.sort_values(["symbol", "timestamp"])
