"""UNPURGED-CV checks: information crossing the train/test boundary.

Reference standard throughout: purged, embargoed walk-forward cross-validation
(Lopez de Prado, *Advances in Financial Machine Learning*, ch. 7).

Every check here is gated on ``ctx.is_temporal``. Shuffled cross-validation on
genuinely cross-sectional data is correct practice, and firing on it would make
the tool useless in any repository that also contains non-time-series work.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from .. import astutils as A
from ..context import FileContext
from ..fixes import replace_on_line
from ..models import Confidence, Finding
from .base import Check, register

RANDOM_SPLITTERS = {
    "KFold": "folds are assigned without reference to time",
    "StratifiedKFold": "folds are assigned by label stratum, not by time",
    "ShuffleSplit": "each split is a random subsample of the whole period",
    "StratifiedShuffleSplit": "each split is a random, label-stratified subsample",
    "GroupShuffleSplit": "groups are sampled at random across the whole period",
}
SEARCH_CALLS = {"cross_val_score", "cross_validate", "GridSearchCV", "RandomizedSearchCV"}


@register
class ShuffledKFold(Check):
    """CV001 - a random-assignment splitter used on time-ordered data."""

    id = "CV001"

    def run(self, ctx: FileContext) -> Iterable[Finding]:
        if not ctx.is_temporal:
            return

        for call in A.walk_calls(ctx.tree):
            tail = A.call_tail(call)
            if tail in RANDOM_SPLITTERS:
                yield self._from_splitter(ctx, call, tail)
            elif tail in SEARCH_CALLS:
                finding = self._from_search(ctx, call, tail)
                if finding is not None:
                    yield finding

    def _from_splitter(self, ctx: FileContext, call, tail: str) -> Finding:
        shuffle = A.keyword_node(call, "shuffle")
        n_splits = A.const(A.arg_or_keyword(call, 0, "n_splits"))
        detail = f"`{tail}` is used on time-ordered data, where {RANDOM_SPLITTERS[tail]}"
        if A.is_true(shuffle):
            detail += " and `shuffle=True` scatters future bars directly into the training folds"
        else:
            detail += (
                "; even unshuffled, every fold but the last trains on bars that occur after "
                "its own test window"
            )
        return self.finding(
            ctx,
            call,
            detail=detail + ".",
            fix=self._walk_forward_fix(ctx, call, tail, n_splits),
            evidence={
                "splitter": tail,
                "shuffle": A.is_true(shuffle),
                "temporal_evidence": ctx.temporal_evidence[:6],
            },
        )

    def _from_search(self, ctx: FileContext, call, tail: str) -> Finding | None:
        cv = A.keyword_node(call, "cv")
        folds = A.const(cv)
        if not isinstance(folds, int) or isinstance(folds, bool):
            return None
        return self.finding(
            ctx,
            call,
            detail=(
                f"`{tail}(cv={folds})` expands to an unshuffled {folds}-fold split with no "
                f"regard for time ordering, which on a time series trains on the future for "
                f"every fold but the last."
            ),
            evidence={"call": tail, "cv": folds},
        )

    def _walk_forward_fix(self, ctx: FileContext, call, tail: str, n_splits) -> str | None:
        horizon = ctx.forward_horizon
        if horizon is None:
            # Without a known label horizon there is no defensible gap to propose,
            # and a diff that guesses one would be worse than no diff.
            return None
        splits = n_splits if isinstance(n_splits, int) else 5
        return replace_on_line(
            ctx,
            A.span(call)[0],
            rf"{re.escape(tail)}\s*\([^()]*\)",
            f"TimeSeriesSplit(n_splits={splits}, gap={horizon})",
        )


@register
class ShuffledTrainTestSplit(Check):
    """CV002 - ``train_test_split`` on a time series without ``shuffle=False``."""

    id = "CV002"

    def run(self, ctx: FileContext) -> Iterable[Finding]:
        if not ctx.is_temporal:
            return

        for call in A.walk_calls(ctx.tree):
            if A.call_tail(call) != "train_test_split":
                continue
            shuffle = A.keyword_node(call, "shuffle")
            if shuffle is not None and A.const(shuffle) is False:
                continue

            explicit = A.is_true(shuffle)
            detail = (
                "`shuffle=True` was passed explicitly"
                if explicit
                else "`shuffle` was not passed and defaults to True"
            )
            lineno = A.span(call)[0]
            fix = replace_on_line(
                ctx, lineno, r"shuffle\s*=\s*True", "shuffle=False"
            ) or replace_on_line(
                ctx, lineno, r"(train_test_split\([^()]*)\)", r"\1, shuffle=False)"
            )
            yield self.finding(
                ctx,
                call,
                detail=(
                    f"{detail}, so the test rows are drawn at random from the whole period and "
                    f"sit interleaved with training rows from immediately before and after them."
                ),
                fix=fix,
                evidence={"shuffle_explicit": explicit},
            )


@register
class NoPurgeGap(Check):
    """CV003 - walk-forward split with no purge gap and no embargo."""

    id = "CV003"

    def run(self, ctx: FileContext) -> Iterable[Finding]:
        if not ctx.is_temporal or ctx.has_purging:
            return

        for call in A.walk_calls(ctx.tree):
            if A.call_tail(call) != "TimeSeriesSplit":
                continue
            gap = ctx.resolve_int(A.keyword_node(call, "gap"))
            if isinstance(gap, int) and gap > 0:
                continue

            horizon = ctx.forward_horizon
            detail = (
                "`TimeSeriesSplit` is constructed with "
                + ("`gap=0`" if gap == 0 else "no `gap=` argument")
                + ", so the last training bar sits immediately before the first test bar, and "
                "nothing is embargoed after the test fold"
            )
            if horizon:
                detail += (
                    f"; the file's labels look {horizon} bar(s) forward, so at least the final "
                    f"{horizon} training rows resolve inside the test window"
                )
            fix = None
            if horizon:
                fix = replace_on_line(
                    ctx,
                    A.span(call)[0],
                    r"(TimeSeriesSplit\([^()]*?)(\s*\))",
                    rf"\1, gap={horizon}\2",
                )
            yield self.finding(
                ctx,
                call,
                detail=detail + ".",
                fix=fix,
                evidence={"gap": gap, "forward_horizon": horizon},
            )


@register
class HorizonExceedsGap(Check):
    """CV004 - a purge gap is present but shorter than the label horizon."""

    id = "CV004"

    def run(self, ctx: FileContext) -> Iterable[Finding]:
        horizon = ctx.forward_horizon
        if not ctx.is_temporal or not horizon:
            return

        for call in A.walk_calls(ctx.tree):
            if A.call_tail(call) != "TimeSeriesSplit":
                continue
            gap = ctx.resolve_int(A.keyword_node(call, "gap"))
            if not isinstance(gap, int) or isinstance(gap, bool) or gap <= 0 or gap >= horizon:
                continue

            overlap = horizon - gap
            fix = replace_on_line(ctx, A.span(call)[0], rf"(gap\s*=\s*){gap}\b", rf"\g<1>{horizon}")
            yield self.finding(
                ctx,
                call,
                confidence=Confidence.HIGH,
                detail=(
                    f"the labels look {horizon} bar(s) forward but the gap is only {gap}, so the "
                    f"last {overlap} training row(s) before each test fold carry outcomes that "
                    f"are realised inside that fold."
                ),
                fix=fix,
                evidence={"gap": gap, "forward_horizon": horizon, "overlap": overlap},
            )
