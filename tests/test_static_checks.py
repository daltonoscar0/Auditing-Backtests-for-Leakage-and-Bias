"""Positive and negative cases for every static check.

The negative cases carry most of the weight. Anyone can write a rule that fires;
the question is whether it fires on `shift(1)`, on `center=False`, on a Sharpe
ratio, on shuffled cross-validation over genuinely cross-sectional data, or on a
backtest that does charge costs. Each of those has a row below.
"""

from __future__ import annotations

import pytest
from conftest import run_check

from hindsight.checks.base import registered_ids

# (check_id, label, source, should_fire)
CASES: list[tuple[str, str, str, bool]] = [
    # ---------------------------------------------------------------- LA001
    (
        "LA001",
        "negative shift on a feature",
        """
        df["reversal"] = df["close"].pct_change().shift(-1)
        """,
        True,
    ),
    (
        "LA001",
        "negative shift written as a keyword",
        """
        df["reversal"] = df["close"].shift(periods=-3)
        """,
        True,
    ),
    (
        "LA001",
        "negative shift through a module constant",
        """
        HORIZON = 5
        df["peek"] = df["close"].shift(-HORIZON)
        """,
        True,
    ),
    (
        "LA001",
        "positive shift is a lag, not a lookahead",
        """
        df["lagged"] = df["close"].shift(1)
        """,
        False,
    ),
    (
        "LA001",
        "negative shift building the target column is how labels are made",
        """
        df["fwd_ret"] = df["close"].pct_change(5).shift(-5)
        """,
        False,
    ),
    # ---------------------------------------------------------------- LA002
    (
        "LA002",
        "centred rolling window",
        """
        df["smooth"] = df["volume"].rolling(21, center=True).mean()
        """,
        True,
    ),
    (
        "LA002",
        "center=False is trailing",
        """
        df["smooth"] = df["volume"].rolling(21, center=False).mean()
        """,
        False,
    ),
    (
        "LA002",
        "the default is trailing",
        """
        df["smooth"] = df["volume"].rolling(21).mean()
        """,
        False,
    ),
    # ---------------------------------------------------------------- LA003
    (
        "LA003",
        "bfill on a feature",
        """
        df["book_to_price"] = df["book_to_price"].bfill()
        """,
        True,
    ),
    (
        "LA003",
        "fillna with a backfill method",
        """
        df["eps"] = df["eps"].fillna(method="backfill")
        """,
        True,
    ),
    (
        "LA003",
        "ffill carries the past forward, which is fine",
        """
        df["book_to_price"] = df["book_to_price"].ffill()
        """,
        False,
    ),
    (
        "LA003",
        "filling with a constant is not a lookahead",
        """
        df["eps"] = df["eps"].fillna(0.0)
        """,
        False,
    ),
    # ---------------------------------------------------------------- LA004
    (
        "LA004",
        "scaler fitted before the split",
        """
        from sklearn.model_selection import train_test_split
        scaler = StandardScaler()
        matrix = scaler.fit_transform(panel[FEATURES])
        X_train, X_test = train_test_split(matrix, test_size=0.2)
        """,
        True,
    ),
    (
        "LA004",
        "scaler fitted after the split",
        """
        from sklearn.model_selection import train_test_split
        X_train, X_test = train_test_split(matrix, test_size=0.2)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(X_train)
        """,
        False,
    ),
    (
        "LA004",
        "a transformer inside a Pipeline is refit per fold",
        """
        from sklearn.model_selection import train_test_split
        pipe = Pipeline([("scale", StandardScaler()), ("model", Ridge())])
        pipe.fit(X, y)
        X_train, X_test = train_test_split(X, test_size=0.2)
        """,
        False,
    ),
    (
        "LA004",
        "no split in the file means nothing to be early relative to",
        """
        scaler = StandardScaler()
        matrix = scaler.fit_transform(panel[FEATURES])
        """,
        False,
    ),
    # ---------------------------------------------------------------- LA005
    (
        "LA005",
        "full-sample z-score",
        """
        df["z"] = (df["value"] - df["value"].mean()) / df["value"].std()
        """,
        True,
    ),
    (
        "LA005",
        "full-sample min-max scaling",
        """
        df["scaled"] = (df["v"] - df["v"].min()) / (df["v"].max() - df["v"].min())
        """,
        True,
    ),
    (
        "LA005",
        "a Sharpe ratio is mean over standard deviation, not a normalisation",
        """
        sharpe = daily.mean() / daily.std() * 15.87
        """,
        False,
    ),
    (
        "LA005",
        "expanding statistics use only the past",
        """
        df["z"] = (df["v"] - df["v"].expanding().mean()) / df["v"].expanding().std()
        """,
        False,
    ),
    (
        "LA005",
        "an expanding view bound to a name is still expanding",
        """
        history = df["v"].expanding(min_periods=252)
        df["z"] = (df["v"] - history.mean()) / history.std()
        """,
        False,
    ),
    (
        "LA005",
        "a trailing rolling z-score is fine",
        """
        df["z"] = (df["v"] - df["v"].rolling(60).mean()) / df["v"].rolling(60).std()
        """,
        False,
    ),
    # ---------------------------------------------------------------- LL001
    (
        "LL001",
        "target in an explicit feature list",
        """
        FEATURES = ["mom_5", "vol_20", "fwd_ret_3d"]
        panel["fwd_ret_3d"] = panel["close"].shift(-3)
        """,
        True,
    ),
    (
        "LL001",
        "target survives a subtractive drop",
        """
        panel["fwd_ret_3d"] = panel["close"].shift(-3)
        X = panel.drop(columns=["date", "close"])
        """,
        True,
    ),
    (
        "LL001",
        "a clean allow-list",
        """
        FEATURES = ["mom_5", "vol_20"]
        panel["fwd_ret_3d"] = panel["close"].shift(-3)
        """,
        False,
    ),
    (
        "LL001",
        "a drop that removes the target",
        """
        panel["fwd_ret_3d"] = panel["close"].shift(-3)
        X = panel.drop(columns=["date", "close", "fwd_ret_3d"])
        """,
        False,
    ),
    # ---------------------------------------------------------------- LL002
    (
        "LL002",
        "feature engineered from the target",
        """
        FEATURES = ["strength"]
        panel["fwd_ret"] = panel["close"].shift(-3)
        panel["strength"] = panel["fwd_ret"].rolling(5).mean()
        """,
        True,
    ),
    (
        "LL002",
        "PnL legitimately consumes the realised forward return",
        """
        panel["fwd_ret"] = panel["close"].shift(-3)
        panel["pnl"] = position * panel["fwd_ret"]
        """,
        False,
    ),
    (
        "LL002",
        "a feature built from observable inputs",
        """
        FEATURES = ["vol_20"]
        panel["fwd_ret"] = panel["close"].shift(-3)
        panel["vol_20"] = panel["close"].pct_change().rolling(20).std()
        """,
        False,
    ),
    # ---------------------------------------------------------------- LL003
    (
        "LL003",
        "scaler fitted on the frame holding the target",
        """
        panel["fwd_ret"] = panel["close"].shift(-3)
        scaler = StandardScaler()
        scaler.fit(panel)
        """,
        True,
    ),
    (
        "LL003",
        "scaler fitted on the feature matrix only",
        """
        panel["fwd_ret"] = panel["close"].shift(-3)
        X = panel[FEATURES]
        scaler = StandardScaler()
        scaler.fit(X)
        """,
        False,
    ),
    # ---------------------------------------------------------------- SV001
    (
        "SV001",
        "a present-day membership helper",
        """
        def current_constituents():
            return pd.read_csv("members.csv")["ticker"].tolist()
        """,
        True,
    ),
    (
        "SV001",
        "a hard-coded ticker list applied to a time-indexed panel",
        """
        TICKERS = ["AAPL", "MSFT", "AMZN", "JPM"]
        panel = pd.read_csv("prices.csv", parse_dates=["date"])
        """,
        True,
    ),
    (
        "SV001",
        "a point-in-time membership table",
        """
        def membership():
            return pd.read_csv("constituents_pit.csv", parse_dates=["date"])
        """,
        False,
    ),
    (
        "SV001",
        "a ticker list in a file with no time dimension is not a historical universe",
        """
        TICKERS = ["AAPL", "MSFT", "AMZN", "JPM"]
        """,
        False,
    ),
    # ---------------------------------------------------------------- SV002
    (
        "SV002",
        "filtering the panel on a listing-status flag",
        """
        live = panel[panel["is_active"]]
        """,
        True,
    ),
    (
        "SV002",
        "a status filter written as a query string",
        """
        live = panel.query("status == 'active'")
        """,
        True,
    ),
    (
        "SV002",
        "an ordinary liquidity filter",
        """
        liquid = panel[panel["volume"] > 0]
        """,
        False,
    ),
    (
        "SV002",
        "a date filter",
        """
        train = panel[panel["date"] < "2021-01-01"]
        """,
        False,
    ),
    # ---------------------------------------------------------------- SV003
    (
        "SV003",
        "selection on a whole-history aggregate",
        """
        big = panel[panel.groupby("ticker")["volume"].transform("sum") > 1e9]
        """,
        True,
    ),
    (
        "SV003",
        "selection on a column named for a whole-sample outcome",
        """
        winners = panel[panel["total_return"] > 0.0]
        """,
        True,
    ),
    (
        "SV003",
        "selection on a trailing statistic computed beforehand",
        """
        panel["adv_20"] = panel["volume"].rolling(20).mean().shift(1)
        liquid = panel[panel["adv_20"] > 1e9]
        """,
        False,
    ),
    # ---------------------------------------------------------------- CV001
    (
        "CV001",
        "shuffled K-fold on time-ordered data",
        """
        panel = pd.read_csv("bars.csv", parse_dates=["date"])
        splitter = KFold(n_splits=5, shuffle=True)
        """,
        True,
    ),
    (
        "CV001",
        "an integer cv on a time series",
        """
        panel = pd.read_csv("bars.csv", parse_dates=["date"])
        scores = cross_val_score(model, X, y, cv=5)
        """,
        True,
    ),
    (
        "CV001",
        "shuffled K-fold on genuinely cross-sectional data",
        """
        splitter = KFold(n_splits=5, shuffle=True, random_state=0)
        scores = cross_val_score(model, X, y, cv=splitter)
        """,
        False,
    ),
    # ---------------------------------------------------------------- CV002
    (
        "CV002",
        "train_test_split on a time series",
        """
        panel = pd.read_csv("bars.csv", parse_dates=["date"])
        X_train, X_test = train_test_split(X, test_size=0.2)
        """,
        True,
    ),
    (
        "CV002",
        "shuffle=False is a contiguous split",
        """
        panel = pd.read_csv("bars.csv", parse_dates=["date"])
        X_train, X_test = train_test_split(X, test_size=0.2, shuffle=False)
        """,
        False,
    ),
    (
        "CV002",
        "train_test_split on cross-sectional data",
        """
        X_train, X_test = train_test_split(X, y, test_size=0.2)
        """,
        False,
    ),
    # ---------------------------------------------------------------- CV003
    (
        "CV003",
        "walk-forward with no gap",
        """
        panel["fwd_ret"] = panel["close"].shift(-10)
        splitter = TimeSeriesSplit(n_splits=6)
        """,
        True,
    ),
    (
        "CV003",
        "an explicit gap of zero is still no gap",
        """
        panel["fwd_ret"] = panel["close"].shift(-10)
        splitter = TimeSeriesSplit(n_splits=6, gap=0)
        """,
        True,
    ),
    (
        "CV003",
        "a positive gap",
        """
        panel["fwd_ret"] = panel["close"].shift(-10)
        splitter = TimeSeriesSplit(n_splits=6, gap=10)
        """,
        False,
    ),
    (
        "CV003",
        "purging implemented elsewhere in the file",
        """
        panel["fwd_ret"] = panel["close"].shift(-10)
        splitter = TimeSeriesSplit(n_splits=6)
        train_idx = purge_overlapping(train_idx, test_idx, embargo=5)
        """,
        False,
    ),
    # ---------------------------------------------------------------- CV004
    (
        "CV004",
        "gap shorter than the label horizon",
        """
        panel["fwd_ret"] = panel["close"].shift(-10)
        splitter = TimeSeriesSplit(n_splits=6, gap=2)
        """,
        True,
    ),
    (
        "CV004",
        "gap equal to the label horizon",
        """
        panel["fwd_ret"] = panel["close"].shift(-10)
        splitter = TimeSeriesSplit(n_splits=6, gap=10)
        """,
        False,
    ),
    (
        "CV004",
        "no gap at all belongs to CV003",
        """
        panel["fwd_ret"] = panel["close"].shift(-10)
        splitter = TimeSeriesSplit(n_splits=6)
        """,
        False,
    ),
    # ---------------------------------------------------------------- CF001
    (
        "CF001",
        "position times return with nothing subtracted",
        """
        def run(panel, position):
            ret = panel["close"].pct_change()
            pnl = position.shift(1) * ret
            return pnl
        """,
        True,
    ),
    (
        "CF001",
        "a costed PnL",
        """
        def run(panel, position, cost_bps):
            ret = panel["close"].pct_change()
            turnover = position.diff().abs()
            pnl = position.shift(1) * ret - turnover * cost_bps / 1e4
            return pnl
        """,
        False,
    ),
    (
        "CF001",
        "a cost constant defined at module level counts as in scope",
        """
        COST_BPS = 2.0

        def run(panel, position):
            pnl = position.shift(1) * panel["ret"] - position.diff().abs() * COST_BPS / 1e4
            return pnl
        """,
        False,
    ),
    # ---------------------------------------------------------------- CF002
    (
        "CF002",
        "filling at the close with no adjustment",
        """
        def execute(panel):
            fill_price = panel["close"]
            return fill_price
        """,
        True,
    ),
    (
        "CF002",
        "crossing half the spread",
        """
        HALF_SPREAD_BPS = 3.0

        def execute(panel, trade):
            fill_price = panel["close"] * (1 + np.sign(trade) * HALF_SPREAD_BPS / 1e4)
            return fill_price
        """,
        False,
    ),
    # ---------------------------------------------------------------- CF003
    (
        "CF003",
        "a cost parameter pinned to zero",
        """
        COMMISSION_BPS = 0.0
        """,
        True,
    ),
    (
        "CF003",
        "a populated cost parameter",
        """
        COMMISSION_BPS = 1.5
        """,
        False,
    ),
    (
        "CF003",
        "a non-cost parameter that happens to be zero",
        """
        MIN_HOLDING_DAYS = 0
        """,
        False,
    ),
    # ---------------------------------------------------------------- CF004
    (
        "CF004",
        "a long/short book with no borrow cost",
        """
        def build(signal):
            position = np.sign(signal)
            return position
        """,
        True,
    ),
    (
        "CF004",
        "short notional charged a loan fee",
        """
        BORROW_BPS_ANNUAL = 40.0

        def build(signal):
            position = np.sign(signal)
            borrow = position.clip(upper=0.0).abs() * BORROW_BPS_ANNUAL / 1e4
            return position, borrow
        """,
        False,
    ),
    (
        "CF004",
        "a long-only book",
        """
        def build(signal):
            weight = signal.clip(lower=0.0)
            return weight / weight.sum()
        """,
        False,
    ),
    # ---------------------------------------------------------------- SO001
    (
        "SO001",
        "a grid scored on the test set and maximised",
        """
        results = []
        for a, b in itertools.product(ALPHAS, BETAS):
            results.append((score(test, a, b), a, b))
        best = max(results)
        """,
        True,
    ),
    (
        "SO001",
        "a running best over a grid scored on the test set",
        """
        best_score = -1e9
        for a, b in itertools.product(ALPHAS, BETAS):
            candidate = score(test, a, b)
            if candidate > best_score:
                best_score = candidate
        """,
        True,
    ),
    (
        "SO001",
        "a grid scored on a validation fold",
        """
        results = []
        for a, b in itertools.product(ALPHAS, BETAS):
            results.append((score(valid, a, b), a, b))
        best = max(results)
        """,
        False,
    ),
    (
        "SO001",
        "a loop that is not a sweep",
        """
        for row in frame.itertuples():
            handle(row)
        """,
        False,
    ),
    # ---------------------------------------------------------------- SO002
    (
        "SO002",
        "best_score_ reported as the result",
        """
        grid = GridSearchCV(model, PARAM_GRID, cv=5)
        grid.fit(X, y)
        print(grid.best_score_)
        """,
        True,
    ),
    (
        "SO002",
        "the chosen model scored on a held-out set",
        """
        grid = GridSearchCV(model, PARAM_GRID, cv=5)
        grid.fit(X_train, y_train)
        print(grid.best_score_)
        print(grid.best_estimator_.score(X_test, y_test))
        """,
        False,
    ),
    # ---------------------------------------------------------------- SO003
    (
        "SO003",
        "the test set drives early stopping",
        """
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=50)
        """,
        True,
    ),
    (
        "SO003",
        "a validation fold drives early stopping",
        """
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=50)
        """,
        False,
    ),
    # ---------------------------------------------------------------- SO004
    (
        "SO004",
        "a wide grid with no correction",
        """
        PARAM_GRID = {
            "n_estimators": [100, 300, 600],
            "max_depth": [3, 5, 8, 12],
            "min_samples_leaf": [1, 5, 20],
        }
        """,
        True,
    ),
    (
        "SO004",
        "a small grid",
        """
        PARAM_GRID = {"max_depth": [3, 5], "n_estimators": [100, 300]}
        """,
        False,
    ),
    (
        "SO004",
        "a wide grid with a deflated Sharpe",
        """
        PARAM_GRID = {
            "n_estimators": [100, 300, 600],
            "max_depth": [3, 5, 8, 12],
            "min_samples_leaf": [1, 5, 20],
        }
        headline = deflated_sharpe(best, trials=36, variance=v)
        """,
        False,
    ),
]


@pytest.mark.parametrize(
    ("check_id", "label", "source", "should_fire"),
    CASES,
    ids=[f"{c[0]}-{'fires' if c[3] else 'quiet'}-{c[1]}" for c in CASES],
)
def test_check_behaviour(check_id: str, label: str, source: str, should_fire: bool) -> None:
    findings = run_check(check_id, source)
    if should_fire:
        assert findings, f"{check_id} should have fired on: {label}"
        assert all(f.check_id == check_id for f in findings)
        assert all(f.mechanism for f in findings)
        assert all(f.location.line >= 1 for f in findings)
    else:
        assert not findings, (
            f"{check_id} fired on a lookalike it must ignore ({label}): "
            f"{[str(f.location) for f in findings]}"
        )


def test_every_registered_check_has_both_a_positive_and_a_negative_case() -> None:
    """A check with no negative case is a check nobody has tested for false positives."""
    positive = {case[0] for case in CASES if case[3]}
    negative = {case[0] for case in CASES if not case[3]}
    registered = set(registered_ids())

    assert registered - positive == set(), "checks with no positive case"
    assert registered - negative == set(), "checks with no negative case"
