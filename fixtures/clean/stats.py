"""Performance statistics, adjusted for the number of trials run.

A Sharpe ratio reported without reference to how many configurations were tried
is not interpretable. The deflated Sharpe below subtracts the level a search of
that size would have produced under the null of zero skill.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import pandas as pd

TRADING_DAYS = 252
EULER_MASCHERONI = 0.5772156649015329


def sharpe_ratio(daily: pd.Series) -> float:
    return float(daily.mean() / daily.std() * math.sqrt(TRADING_DAYS))


def expected_max_sharpe(trials: int, variance: float) -> float:
    """Expected maximum Sharpe across `trials` independent backtests under the null."""
    if trials < 2:
        return 0.0
    normal = NormalDist()
    z_upper = normal.inv_cdf(1.0 - 1.0 / trials)
    z_lower = normal.inv_cdf(1.0 - 1.0 / (trials * math.e))
    return math.sqrt(variance) * (
        (1.0 - EULER_MASCHERONI) * z_upper + EULER_MASCHERONI * z_lower
    )


def deflated_sharpe(observed: float, trials: int, variance: float) -> float:
    """Observed Sharpe less the level a search of this size would produce by luck."""
    return observed - expected_max_sharpe(trials, variance)
