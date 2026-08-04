"""Daily long/short backtest.

Execution assumptions live in execution.py; nothing here duplicates them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from execution import execution_price, net_of_charges


def positions(signal: pd.Series) -> pd.Series:
    """Long the top of the cross-section, short the bottom."""
    position = np.sign(signal)
    return position


def fills(panel: pd.DataFrame, trade: pd.Series) -> pd.Series:
    fill_price = execution_price(panel["close"], trade)
    return fill_price


def pnl(panel: pd.DataFrame, position: pd.Series) -> pd.Series:
    ret = panel["close"].pct_change()
    pnl = position.shift(1) * ret
    return net_of_charges(pnl, position)
