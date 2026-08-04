"""Machine-readable finding taxonomy.

``TAXONOMY.md`` is the prose specification; this module is the version of it that
the code actually reads. They are kept in sync by ``tests/test_taxonomy.py``,
which asserts that every id here appears in the document and that every
registered check has a spec.

Bumping :data:`TAXONOMY_VERSION` is required whenever a check id changes meaning,
because stored reports and fixture ``expected_findings.json`` files pin against it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .models import Category, Severity

TAXONOMY_VERSION = "1.0.0"

Strategy = Literal["static", "llm", "both"]


@dataclass(frozen=True, slots=True)
class CheckSpec:
    """Normative definition of a single detection rule."""

    id: str
    category: Category
    name: str
    strategy: Strategy
    severity: Severity
    mechanism: str
    """Why this biases results. Rendered into every finding; written so a reader
    who has never seen the tool can judge the claim on its merits."""
    fix: str
    """The canonical remedy, in prose. Checks that can emit a mechanical patch
    generate a unified diff on top of this."""


def _spec(
    id: str,
    category: Category,
    name: str,
    strategy: Strategy,
    severity: Severity,
    mechanism: str,
    fix: str,
) -> CheckSpec:
    return CheckSpec(id, category, name, strategy, severity, mechanism, fix)


_SPECS: tuple[CheckSpec, ...] = (
    # ---------------------------------------------------------------- LOOKAHEAD
    _spec(
        "LA001",
        Category.LOOKAHEAD,
        "Negative shift on a feature column",
        "static",
        Severity.CRITICAL,
        "A negative argument to .shift() moves data backwards in time, so the value "
        "sitting on row t was observed at t+n. Any model trained on this column is "
        "reading the future. Backtest returns will look strong and will not survive "
        "live trading, because at decision time t the value simply does not exist yet.",
        "Use a non-negative shift. Features must be built only from bars at or before "
        "t; if the intent was to build a label, assign it to the target column and "
        "keep it out of the feature matrix.",
    ),
    _spec(
        "LA002",
        Category.LOOKAHEAD,
        "Centered rolling window",
        "static",
        Severity.CRITICAL,
        "rolling(window=n, center=True) places the window symmetrically around each "
        "row, so the statistic at t is computed from roughly n/2 bars that occur after "
        "t. The feature is a smoothed view of the future and leaks the very move the "
        "model is asked to predict.",
        "Drop center=True (the default, center=False, is trailing) and widen the "
        "window if more smoothing is needed.",
    ),
    _spec(
        "LA003",
        Category.LOOKAHEAD,
        "Backward fill on a feature column",
        "static",
        Severity.HIGH,
        "Backward filling propagates a later observation into earlier, missing rows. "
        "Where the data was genuinely unavailable at t, the model now sees the value "
        "that will only be published later - a textbook lookahead in fundamentals and "
        "any vendor series with reporting lag.",
        "Forward fill instead (ffill), or leave the gap as NaN and let the model handle "
        "missingness. Restrict any fill to information available at or before t.",
    ),
    _spec(
        "LA004",
        Category.LOOKAHEAD,
        "Transformer fitted before the train/test split",
        "static",
        Severity.HIGH,
        "Calling .fit() or .fit_transform() on the full frame before splitting lets the "
        "scaler, imputer or decomposition learn the mean, variance and covariance "
        "structure of the test period. The test set is no longer out-of-sample: its "
        "distribution has already been folded into the transform.",
        "Split first, fit the transformer on the training fold only, then .transform() "
        "the test fold. Inside cross-validation, wrap the transformer in a Pipeline so "
        "it is refitted per fold.",
    ),
    _spec(
        "LA005",
        Category.LOOKAHEAD,
        "Full-sample normalisation statistics",
        "static",
        Severity.HIGH,
        "Normalising a column by its whole-history mean and standard deviation embeds "
        "the entire sample - including the future - into every row. A z-score at t "
        "silently encodes where t sits in a distribution that could not be known until "
        "the end of the backtest, which alone can manufacture a positive Sharpe.",
        "Use an expanding or trailing-rolling statistic so each row is normalised only "
        "by data available at or before it.",
    ),
    _spec(
        "LA006",
        Category.LOOKAHEAD,
        "Feature timestamped after the bar it indexes",
        "llm",
        Severity.HIGH,
        "A feature attached to bar t is derived from a record whose own timestamp, "
        "publication lag or resampling boundary falls after t. Static analysis cannot "
        "resolve the alignment; it depends on the semantics of the join keys and the "
        "vendor's release schedule.",
        "Align every input on its point-in-time availability timestamp rather than its "
        "effective date, and apply the vendor's publication lag explicitly.",
    ),
    # --------------------------------------------------------------- LABEL-LEAK
    _spec(
        "LL001",
        Category.LABEL_LEAK,
        "Target column reachable from the feature matrix",
        "static",
        Severity.CRITICAL,
        "The target, or a column derived from it, is still present in the feature "
        "matrix handed to .fit(). The model can read the answer directly, so training "
        "and test metrics both collapse to a near-perfect fit that measures nothing.",
        "Build the feature matrix from an explicit allow-list of columns rather than by "
        "dropping the target, and assert the target's name and its derivatives are "
        "absent before fitting.",
    ),
    _spec(
        "LL002",
        Category.LABEL_LEAK,
        "Feature engineered from the target column",
        "static",
        Severity.CRITICAL,
        "A feature is computed from the target series. Even smoothed or lagged, it "
        "carries the label's own signal, so cross-validated scores reflect the "
        "transform rather than any predictive relationship in the data.",
        "Derive features only from observable inputs. If a lagged realisation of the "
        "target is genuinely tradeable information, construct it from the raw price "
        "series with an explicit lag and document the availability argument.",
    ),
    _spec(
        "LL003",
        Category.LABEL_LEAK,
        "Transformer fitted on a frame that still contains the target",
        "static",
        Severity.HIGH,
        "Fitting a scaler or decomposition on a frame that includes the target column "
        "leaks label information into every transformed feature through the shared "
        "covariance structure, even if the target column itself is dropped afterwards.",
        "Fit transformers on the feature matrix only, after the target has been separated out.",
    ),
    _spec(
        "LL004",
        Category.LABEL_LEAK,
        "No gap between the feature bar and the label horizon",
        "llm",
        Severity.HIGH,
        "Features and label are drawn from the same bar or from overlapping windows, so "
        "part of the label's realisation is already embedded in the inputs. Whether a "
        "gap is needed depends on how the bar is built and when it is observable.",
        "Insert an explicit gap of at least one bar between the last observation used "
        "for features and the first used for the label, and state the execution "
        "assumption that justifies its size.",
    ),
    # -------------------------------------------------------------- SURVIVORSHIP
    _spec(
        "SV001",
        Category.SURVIVORSHIP,
        "Universe drawn from current constituents",
        "static",
        Severity.CRITICAL,
        "Applying today's index membership or ticker list to historical data tests a "
        "universe selected for having survived. Names that were delisted, acquired or "
        "removed from the index never appear, so the backtest measures the return of "
        "companies already known to have made it.",
        "Use a point-in-time constituents table keyed by date and reconstruct the "
        "universe as of each rebalance.",
    ),
    _spec(
        "SV002",
        Category.SURVIVORSHIP,
        "Delisted or inactive names filtered out of history",
        "static",
        Severity.HIGH,
        "Filtering the historical panel on a current status flag removes exactly the "
        "names whose outcomes were worst. The surviving sample has a return "
        "distribution with the left tail cut off, which inflates mean return and "
        "understates drawdown.",
        "Keep delisted names in the panel and model the delisting return explicitly "
        "(a recovery value, or the exchange's final price).",
    ),
    _spec(
        "SV003",
        Category.SURVIVORSHIP,
        "Universe selected on a whole-sample outcome",
        "static",
        Severity.CRITICAL,
        "The universe is filtered by a statistic computed over the entire history - a "
        "total return, a final price, a cumulative flag. Membership therefore depends "
        "on what the name went on to do, which is the strongest form of selection bias "
        "there is.",
        "Select on information available at the rebalance date only. If a liquidity or "
        "size screen is needed, compute it on a trailing window.",
    ),
    _spec(
        "SV004",
        Category.SURVIVORSHIP,
        "Static universe table joined across history",
        "llm",
        Severity.MEDIUM,
        "A universe or metadata table with no date dimension is joined onto a "
        "time-indexed panel. Whether this biases the sample depends on how the table "
        "was assembled and whether it is genuinely time-invariant.",
        "Join on a point-in-time table keyed by (identifier, date), or document why the "
        "attribute cannot change over the sample.",
    ),
    # -------------------------------------------------------------- UNPURGED-CV
    _spec(
        "CV001",
        Category.UNPURGED_CV,
        "Shuffled K-fold on time-ordered data",
        "static",
        Severity.CRITICAL,
        "Shuffling breaks the time ordering, so training folds contain bars that occur "
        "after the bars being predicted. With autocorrelated financial series the model "
        "interpolates rather than forecasts, and cross-validated scores bear no "
        "relation to what walk-forward trading would have produced.",
        "Replace with a walk-forward splitter that respects time order, and add a purge "
        "gap and embargo around each test fold.",
    ),
    _spec(
        "CV002",
        Category.UNPURGED_CV,
        "train_test_split with shuffling on a time series",
        "static",
        Severity.CRITICAL,
        "train_test_split shuffles by default. On a time series that scatters future "
        "bars into the training set and leaves the test set interleaved with training "
        "data from immediately before and after it.",
        "Pass shuffle=False, or better, slice the frame by date so the test period is a "
        "contiguous block strictly after the training period.",
    ),
    _spec(
        "CV003",
        Category.UNPURGED_CV,
        "Walk-forward split with no purge gap or embargo",
        "static",
        Severity.HIGH,
        "TimeSeriesSplit with no gap puts the last training bar immediately before the "
        "first test bar. Labels built over a forward horizon straddle that boundary, so "
        "training rows carry outcomes realised inside the test window. An embargo after "
        "the test fold is needed too, because serial correlation runs both ways.",
        "Set a purge gap of at least the label horizon (sklearn's TimeSeriesSplit "
        "accepts gap=), and embargo a further fraction of the sample after each test "
        "fold.",
    ),
    _spec(
        "CV004",
        Category.UNPURGED_CV,
        "Label horizon exceeds the purge gap",
        "static",
        Severity.HIGH,
        "The forward-looking label spans more bars than the gap separating train from "
        "test, so the outcome attached to the final training rows is realised inside "
        "the test period. The overlap is exactly the quantity purging exists to remove.",
        "Set the gap to at least the label horizon in bars. Where labels have variable "
        "length (triple-barrier, event-driven), purge on the actual event end times.",
    ),
    _spec(
        "CV005",
        Category.UNPURGED_CV,
        "Grouped split that ignores the time axis",
        "llm",
        Severity.MEDIUM,
        "Splitting by entity (ticker, instrument) keeps a name whole but allows a "
        "training fold to span the same calendar window as the test fold. In a market "
        "with common factor exposure that is enough to leak the period's realised "
        "factor returns across the boundary.",
        "Split on time first and on entity second, or use a combinatorial purged "
        "cross-validation scheme that respects both axes.",
    ),
    # ------------------------------------------------------------- COST-FANTASY
    _spec(
        "CF001",
        Category.COST_FANTASY,
        "PnL arithmetic with no cost term",
        "static",
        Severity.CRITICAL,
        "Returns are computed as position times market return with nothing subtracted. "
        "Every rebalance is free. For anything above weekly turnover the omitted cost "
        "is usually larger than the entire measured edge, so a strategy that looks "
        "profitable gross is loss-making net.",
        "Subtract turnover times a round-trip cost on every rebalance: "
        "pnl = position.shift(1) * ret - position.diff().abs() * cost_bps / 1e4.",
    ),
    _spec(
        "CF002",
        Category.COST_FANTASY,
        "Fills at close or mid with no spread or slippage",
        "static",
        Severity.HIGH,
        "Assuming execution at the closing print or the mid quote assumes the trade "
        "crosses no spread and moves no price. Real orders pay at least half the spread "
        "and more in size, which is the dominant cost for anything intraday or in less "
        "liquid names.",
        "Fill at the far touch, or apply an explicit slippage model to the reference "
        "price in the direction of the trade.",
    ),
    _spec(
        "CF003",
        Category.COST_FANTASY,
        "Cost parameter pinned to zero",
        "static",
        Severity.HIGH,
        "A cost, fee, commission or slippage parameter is present but set to zero. The "
        "accounting is structurally correct and the number is a placeholder, so the "
        "reported PnL is a gross figure being read as net.",
        "Populate with a defensible estimate - venue fees plus half-spread plus a market "
        "impact term - and report the strategy's break-even cost alongside its Sharpe.",
    ),
    _spec(
        "CF004",
        Category.COST_FANTASY,
        "Short positions with no borrow cost",
        "static",
        Severity.MEDIUM,
        "The strategy takes short exposure but charges no stock loan fee. For general "
        "collateral the omission is small; for the hard-to-borrow names that dominate "
        "most short-side signals it can exceed the gross return entirely, and those are "
        "precisely the names a cross-sectional short screen selects.",
        "Apply a borrow rate to short notional, sourced per name where possible, and "
        "exclude names that were not borrowable at the time.",
    ),
    _spec(
        "CF005",
        Category.COST_FANTASY,
        "Unlimited liquidity assumed",
        "llm",
        Severity.MEDIUM,
        "Order size is never checked against traded volume, so the backtest can take "
        "positions no participant could have filled. Whether this matters depends on "
        "the capital assumed and the names traded.",
        "Cap participation at a fraction of ADV, and add a market impact term that "
        "scales with participation rate.",
    ),
    # ------------------------------------------------------------- SWEEP-OVERFIT
    _spec(
        "SO001",
        Category.SWEEP_OVERFIT,
        "Parameter grid scored on the test set and maximised",
        "static",
        Severity.CRITICAL,
        "Every parameter combination is evaluated against the test set and the best "
        "result is reported. The maximum of many noisy estimates is biased upward by "
        "construction; the test set has been consumed as a training signal and there is "
        "no out-of-sample number left in the study.",
        "Select on a validation fold or on nested cross-validation, and evaluate the "
        "single chosen configuration on a test set touched exactly once.",
    ),
    _spec(
        "SO002",
        Category.SWEEP_OVERFIT,
        "Search best_score_ reported as out-of-sample",
        "static",
        Severity.HIGH,
        "best_score_ is the maximum over the search grid of a cross-validated score "
        "computed on the same data the search optimised. Reporting it as the strategy's "
        "expected performance carries the full selection bias of the sweep.",
        "Report the score of the refit best estimator on a held-out set that took no "
        "part in the search, or use nested cross-validation.",
    ),
    _spec(
        "SO003",
        Category.SWEEP_OVERFIT,
        "Test set used for early stopping",
        "static",
        Severity.CRITICAL,
        "Passing the test set as the evaluation set makes the stopping iteration a "
        "function of test performance. The model is tuned on the data meant to measure "
        "it, and the resulting score is optimistic by an amount nobody can quantify "
        "after the fact.",
        "Carve a validation fold out of the training period for early stopping and keep "
        "the test set untouched until the final evaluation.",
    ),
    _spec(
        "SO004",
        Category.SWEEP_OVERFIT,
        "Large sweep with no multiple-testing correction",
        "static",
        Severity.MEDIUM,
        "Searching a wide grid and reporting the winner's Sharpe without adjusting for "
        "the number of trials overstates significance. Under the null, the expected "
        "maximum Sharpe across N independent backtests grows roughly with the square "
        "root of log N - a large enough sweep produces an impressive result from noise "
        "alone.",
        "Report a deflated Sharpe ratio that accounts for the number of trials and the "
        "skew and kurtosis of returns, or hold out a period the sweep never saw.",
    ),
    _spec(
        "SO005",
        Category.SWEEP_OVERFIT,
        "Iterative refinement against a single holdout",
        "llm",
        Severity.MEDIUM,
        "The same holdout period is used repeatedly across successive rounds of model "
        "revision. No single run looks like a sweep, but the researcher is the search "
        "algorithm and the holdout has been fitted through them.",
        "Fix the holdout before the study begins, budget the number of times it may be "
        "queried, and record every evaluation against it.",
    ),
)

SPECS: dict[str, CheckSpec] = {spec.id: spec for spec in _SPECS}


def spec(check_id: str) -> CheckSpec:
    try:
        return SPECS[check_id]
    except KeyError:
        raise KeyError(f"unknown check id {check_id!r}; see TAXONOMY.md") from None


def specs_for(category: Category) -> list[CheckSpec]:
    return [s for s in _SPECS if s.category is category]


def static_check_ids() -> list[str]:
    return [s.id for s in _SPECS if s.strategy in ("static", "both")]


def llm_check_ids() -> list[str]:
    return [s.id for s in _SPECS if s.strategy in ("llm", "both")]
