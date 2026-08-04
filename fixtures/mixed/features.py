"""Feature construction for the perpetual futures strategy."""

from __future__ import annotations

import pandas as pd

VOL_WINDOW = 24


def add_features(bars: pd.DataFrame) -> pd.DataFrame:
    out = bars.copy()
    out["ret_1"] = out["close"].pct_change()
    out["vol_24"] = out["ret_1"].rolling(VOL_WINDOW).std()
    out["funding_next"] = out["funding_rate"].shift(-1)
    return out
