"""SWEEP-OVERFIT checks: the reported number is the maximum of many trials."""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Iterable

from .. import astutils as A
from ..context import CORRECTION_TERMS, TEST_VARIABLE, FileContext
from ..models import Confidence, Finding
from .base import Check, register

GRID_VARIABLE = re.compile(
    r"^(param_grid|params|grid|search_space|space|combos|combinations|candidates|settings|"
    r"configs|configurations|sweep)$",
    re.IGNORECASE,
)
BEST_VARIABLE = re.compile(r"best|top|winner|champion|optimal", re.IGNORECASE)
SELECTION_CALLS = frozenset({"max", "sorted", "nlargest", "idxmax", "argmax"})
METRIC_TERM = re.compile(
    r"sharpe|score|ret|pnl|profit|accuracy|auc|r2|ic|calmar|sortino|hit_rate", re.IGNORECASE
)

TRIAL_THRESHOLD = 25
"""Grid size above which SO004 fires. Documented in TAXONOMY.md; deliberately low,
because the upward bias in the maximum is already material at a few dozen trials."""


def _mentions_test_data(node: ast.AST) -> list[str]:
    return sorted({name for name in A.referenced_names(node) if TEST_VARIABLE.search(name)})


def _literal_lengths(ctx: FileContext) -> dict[str, int]:
    """Lengths of module-level literal sequences, for sizing a grid built from names."""
    lengths: dict[str, int] = {}
    for node in ast.walk(ctx.tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.List | ast.Tuple):
            continue
        for target in A.assign_targets(node):
            label = A.target_label(target)
            if label:
                lengths[label] = len(node.value.elts)
    for node in ast.walk(ctx.tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        if A.call_tail(node.value) != "range":
            continue
        bounds = [A.const(a) for a in node.value.args]
        if all(isinstance(b, int) for b in bounds) and bounds:
            try:
                size = len(range(*bounds))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            for target in A.assign_targets(node):
                label = A.target_label(target)
                if label:
                    lengths[label] = size
    return lengths


def _sequence_size(node: ast.expr, lengths: dict[str, int]) -> int | None:
    if isinstance(node, ast.List | ast.Tuple | ast.Set):
        return len(node.elts)
    if isinstance(node, ast.Name):
        return lengths.get(node.id)
    if isinstance(node, ast.Call) and A.call_tail(node) == "range":
        bounds = [A.const(a) for a in node.args]
        if bounds and all(isinstance(b, int) for b in bounds):
            try:
                return len(range(*bounds))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return None
    return None


@register
class GridMaximisedOnTest(Check):
    """SO001 - a parameter sweep evaluated against the test set and maximised."""

    id = "SO001"

    def run(self, ctx: FileContext) -> Iterable[Finding]:
        for loop in ast.walk(ctx.tree):
            if not isinstance(loop, ast.For):
                continue
            if not self._is_grid_loop(loop):
                continue
            test_names = _mentions_test_data(loop)
            if not test_names:
                continue

            selection = self._selection(ctx, loop)
            if selection is None:
                continue
            how, where = selection

            yield self.finding(
                ctx,
                loop,
                end_line=A.span(loop)[0],
                detail=(
                    f"the loop scores every parameter combination against "
                    f"{', '.join(f'`{n}`' for n in test_names)} and the result is then selected "
                    f"by {how} (line {where}). The maximum over many noisy estimates is biased "
                    f"upward by construction, and there is no untouched data left to measure it "
                    f"against."
                ),
                evidence={"test_variables": test_names, "selection": how, "selection_line": where},
            )

    @staticmethod
    def _is_grid_loop(loop: ast.For) -> bool:
        iterator = loop.iter
        if isinstance(iterator, ast.Call) and A.call_tail(iterator) == "product":
            return True
        if isinstance(iterator, ast.Name) and GRID_VARIABLE.match(iterator.id):
            return True
        if isinstance(iterator, ast.List | ast.Tuple) and len(iterator.elts) > 1:
            return True
        # `range` is deliberately not treated as a grid. Loop counters are far more
        # common than sweeps written as `for n in range(...)`, and the cost of the
        # false positives outweighs the handful of sweeps it would catch.
        if isinstance(iterator, ast.Call) and A.call_tail(iterator) == "ParameterGrid":
            return True
        return any(isinstance(child, ast.For) for child in loop.body)

    def _selection(self, ctx: FileContext, loop: ast.For) -> tuple[str, int] | None:
        """How the loop's results are reduced to a single reported number."""
        containers = {
            A.dotted(call.func).rsplit(".", 1)[0]
            for call in A.walk_calls(loop)
            if A.call_tail(call) == "append"
        }
        containers |= {
            label
            for node in ast.walk(loop)
            if isinstance(node, ast.Assign)
            for label in (A.target_label(t) for t in A.assign_targets(node))
            if label and METRIC_TERM.search(label)
        }
        containers.discard("")

        loop_start = A.span(loop)[0]
        for call in A.walk_calls(ctx.tree):
            if A.call_tail(call) not in SELECTION_CALLS:
                continue
            line = A.span(call)[0]
            if line < loop_start:
                continue
            if A.referenced_names(call) & containers:
                return f"`{A.call_tail(call)}(...)`", line

        # Running-best pattern: `if score > best_score: best_score = score`
        for node in ast.walk(loop):
            if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
                continue
            assigned = {
                label
                for stmt in node.body
                if isinstance(stmt, ast.Assign)
                for label in (A.target_label(t) for t in A.assign_targets(stmt))
                if label
            }
            if any(BEST_VARIABLE.search(label) for label in assigned):
                return "a running best", A.span(node)[0]
        return None


@register
class SearchBestScoreReported(Check):
    """SO002 - ``best_score_`` presented as an out-of-sample result."""

    id = "SO002"

    def run(self, ctx: FileContext) -> Iterable[Finding]:
        if self._has_untouched_holdout(ctx):
            return

        for node in ast.walk(ctx.tree):
            if not isinstance(node, ast.Attribute) or node.attr != "best_score_":
                continue
            stmt = ctx.enclosing_statement(node) or node
            yield self.finding(
                ctx,
                stmt,
                detail=(
                    f"`{A.dotted(node)}` is the maximum, over the search grid, of a "
                    f"cross-validated score computed on the same data the search optimised, and "
                    f"the file evaluates nothing on a set the search never saw"
                ),
                evidence={"attribute": A.dotted(node)},
            )

    @staticmethod
    def _has_untouched_holdout(ctx: FileContext) -> bool:
        """Whether some estimator is scored on a test-named dataset outside the search."""
        for call in A.walk_calls(ctx.tree):
            if A.call_tail(call) not in {"score", "predict", "predict_proba", "decision_function"}:
                continue
            if _mentions_test_data(call):
                return True
        return False


@register
class TestSetEarlyStopping(Check):
    """SO003 - the test set drives the stopping rule."""

    id = "SO003"

    def run(self, ctx: FileContext) -> Iterable[Finding]:
        for call in A.walk_calls(ctx.tree):
            if A.call_tail(call) not in {"fit", "train"}:
                continue

            for keyword in call.keywords:
                if keyword.arg not in {"eval_set", "validation_data", "validation_set"}:
                    continue
                names = _mentions_test_data(keyword.value)
                if not names:
                    continue
                yield self.finding(
                    ctx,
                    call,
                    detail=(
                        f"`{keyword.arg}=` receives {', '.join(f'`{n}`' for n in names)}, so the "
                        f"stopping iteration is chosen by performance on the very data meant to "
                        f"measure the model"
                    ),
                    evidence={"keyword": keyword.arg, "test_variables": names},
                )
                break
            else:
                if not A.has_keyword(call, "early_stopping_rounds"):
                    continue
                names = _mentions_test_data(call)
                if not names:
                    continue
                yield self.finding(
                    ctx,
                    call,
                    confidence=Confidence.MEDIUM,
                    detail=(
                        f"early stopping is configured on a call that also references "
                        f"{', '.join(f'`{n}`' for n in names)}"
                    ),
                    evidence={"keyword": "early_stopping_rounds", "test_variables": names},
                )


@register
class UncorrectedSweep(Check):
    """SO004 - a wide search reported without a multiple-testing adjustment."""

    id = "SO004"

    def run(self, ctx: FileContext) -> Iterable[Finding]:
        if ctx.matches(CORRECTION_TERMS):
            return

        lengths = _literal_lengths(ctx)
        best: tuple[int, ast.AST, str] | None = None

        for node in ast.walk(ctx.tree):
            sized = self._grid_size(node, lengths)
            if sized is None:
                continue
            size, description = sized
            if size <= TRIAL_THRESHOLD:
                continue
            if best is None or size > best[0]:
                best = (size, node, description)

        if best is None:
            return
        size, node, description = best
        expected = math.sqrt(2 * math.log(size))
        yield self.finding(
            ctx,
            ctx.enclosing_statement(node) or node,
            confidence=Confidence.MEDIUM,
            detail=(
                f"{description} enumerates {size} parameter combinations and nothing in the file "
                f"corrects for the number of trials. Under the null of zero skill the expected "
                f"maximum t-statistic across {size} independent trials is already about "
                f"{expected:.1f}, so a headline result of that size is what noise looks like"
            ),
            evidence={"trials": size, "threshold": TRIAL_THRESHOLD, "source": description},
        )

    @staticmethod
    def _grid_size(node: ast.AST, lengths: dict[str, int]) -> tuple[int, str] | None:
        if isinstance(node, ast.Call) and A.call_tail(node) == "product":
            sizes = [_sequence_size(arg, lengths) for arg in node.args]
            if not sizes or any(s is None for s in sizes):
                return None
            total = math.prod(s for s in sizes if s is not None)
            return (total, "`itertools.product(...)`")

        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            label = next(
                (lbl for lbl in (A.target_label(t) for t in A.assign_targets(node)) if lbl), None
            )
            if not label or not GRID_VARIABLE.match(label):
                return None
            sizes = [_sequence_size(v, lengths) for v in node.value.values]
            usable = [s for s in sizes if s is not None]
            if not usable or len(usable) != len(sizes):
                return None
            return (math.prod(usable), f"the grid `{label}`")

        return None
