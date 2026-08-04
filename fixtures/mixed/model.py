"""Fit the four-hour direction classifier."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HORIZON = 4

FEATURES = ["ret_1", "vol_24", "funding_next", "fwd_ret_4h"]


def add_label(panel: pd.DataFrame) -> pd.DataFrame:
    panel["fwd_ret_4h"] = panel["close"].pct_change(HORIZON).shift(-HORIZON)
    return panel.dropna()


def fit(panel: pd.DataFrame) -> RandomForestClassifier:
    scaler = StandardScaler()
    matrix = scaler.fit_transform(panel[FEATURES])
    X_train, X_test, y_train, y_test = train_test_split(
        matrix, panel["fwd_ret_4h"], test_size=0.25, random_state=11
    )
    model = RandomForestClassifier(n_estimators=300, random_state=11)
    model.fit(X_train, y_train)
    return model
