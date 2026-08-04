"""Turn the classifier score into a position and a return series."""

from __future__ import annotations

import pandas as pd

from config import LEVERAGE

ENTRY_THRESHOLD = 0.55


def positions(score: pd.Series) -> pd.Series:
    return (score > ENTRY_THRESHOLD).astype(float) * LEVERAGE


def pnl(panel: pd.DataFrame, position: pd.Series) -> pd.Series:
    pnl = position.shift(1) * panel["ret_1"]
    return pnl
