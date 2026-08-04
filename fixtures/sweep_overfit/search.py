"""Hyper-parameter search for the daily direction classifier."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

PARAM_GRID = {
    "n_estimators": [100, 300, 600],
    "max_depth": [3, 5, 8, 12],
    "min_samples_leaf": [1, 5, 20],
}


def search(X, y) -> RandomForestClassifier:
    grid = GridSearchCV(RandomForestClassifier(random_state=0), PARAM_GRID, cv=5)
    grid.fit(X, y)
    print(f"cross-validated accuracy: {grid.best_score_:.3f}")
    return grid.best_estimator_
