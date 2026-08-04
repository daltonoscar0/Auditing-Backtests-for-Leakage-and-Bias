"""Purged, embargoed walk-forward validation.

Follows Lopez de Prado, *Advances in Financial Machine Learning*, ch. 7. Two
things are enforced: training observations whose label horizon overlaps a test
fold are purged out of the training set, and a further embargo is applied after
each test fold, because serial correlation runs in both directions.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

HORIZON = 10
EMBARGO = 5
N_SPLITS = 6


def purged_walk_forward(
    n_samples: int,
    n_splits: int = N_SPLITS,
    horizon: int = HORIZON,
    embargo: int = EMBARGO,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train, test) index arrays with a purge gap and a trailing embargo."""
    fold_size = n_samples // (n_splits + 1)
    for fold in range(1, n_splits + 1):
        test_start = fold * fold_size
        test_stop = min(test_start + fold_size, n_samples)

        purge_to = max(0, test_start - horizon)
        before = np.arange(0, purge_to)
        after = np.arange(min(test_stop + embargo, n_samples), n_samples)

        yield np.concatenate([before, after]), np.arange(test_start, test_stop)


def score(model, X: np.ndarray, y: np.ndarray) -> list[float]:
    """Refit on each purged training set and score the fold it was held out from."""
    fold_scores = []
    for train_idx, valid_idx in purged_walk_forward(len(X)):
        model.fit(X[train_idx], y[train_idx])
        fold_scores.append(model.score(X[valid_idx], y[valid_idx]))
    return fold_scores
