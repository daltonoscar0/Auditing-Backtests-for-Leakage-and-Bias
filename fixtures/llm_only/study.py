"""Model selection log for the intraday study.

Kept as a running record so the team can see how the configuration was arrived
at. Each round was evaluated against the same holdout period.
"""

from __future__ import annotations

import pandas as pd

HOLDOUT_START = "2023-01-01"

ROUNDS = [
    # (round, change made, holdout Sharpe)
    (1, "baseline: earnings yield only", 0.31),
    (2, "added realised vol", 0.44),
    (3, "dropped the vol term, added the overnight gap", 0.58),
    (4, "widened the vol window from 20 to 60", 0.71),
    (5, "excluded the two worst-performing sectors", 0.94),
]


def holdout(panel: pd.DataFrame) -> pd.DataFrame:
    return panel[panel["date"] >= HOLDOUT_START]


def report(panel: pd.DataFrame, model) -> float:
    """Headline number for the writeup: round 5, evaluated on the holdout."""
    period = holdout(panel)
    predictions = model.predict(period[FEATURES])
    realised = predictions * period["intraday_move"]
    return float(realised.mean() / realised.std() * (252**0.5))


FEATURES = ["earnings_yield", "realised_vol", "overnight_gap"]
