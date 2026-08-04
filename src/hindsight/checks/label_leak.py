"""LABEL-LEAK checks: the target, or a derivative of it, reachable from the features."""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable

from .. import astutils as A
from ..context import TRANSFORMER_HINT, FileContext
from ..fixes import replace_on_line
from ..models import Confidence, Finding
from .base import Check, register

FEATURE_LIST_NAMES = re.compile(
    r"^(x|x_train|x_all|features?|feature_cols|feature_columns|predictors|inputs|cols)$",
    re.IGNORECASE,
)
ACCOUNTING_COLUMNS = re.compile(
    r"pnl|profit|equity|nav|position|weight|trade|turnover|cost|fee|slippage|"
    r"drawdown|sharpe|exposure|notional|strategy_ret|gross|net",
    re.IGNORECASE,
)


def feature_columns(ctx: FileContext) -> set[str]:
    """Columns the file explicitly treats as model inputs.

    Sourced from a literal feature-name list (``FEATURES = [...]``) and from
    subscript construction of the design matrix (``X = df[[...]]``). Empty when
    the file builds its features some other way, which callers must handle.
    """
    columns: set[str] = set()
    for node in ast.walk(ctx.tree):
        if not isinstance(node, ast.Assign | ast.AnnAssign):
            continue
        value = node.value
        if value is None:
            continue
        labels = [A.target_label(t) for t in A.assign_targets(node)]
        named_feature_list = any(label and FEATURE_LIST_NAMES.match(label) for label in labels)

        if named_feature_list and isinstance(value, ast.List | ast.Tuple):
            columns |= {
                elt.value
                for elt in value.elts
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
            }
        if isinstance(value, ast.Subscript) and isinstance(value.slice, ast.List):
            columns |= {
                elt.value
                for elt in value.slice.elts
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
            }
    return columns


def assigned_columns(ctx: FileContext) -> dict[str, ast.AST]:
    """Every ``frame["col"] = ...`` in the file, mapped to its assignment node."""
    columns: dict[str, ast.AST] = {}
    for node in ast.walk(ctx.tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in A.assign_targets(node):
            if isinstance(target, ast.Subscript):
                key = A.subscript_key(target)
                if key:
                    columns.setdefault(key, node)
    return columns


def _dropped_columns(call: ast.Call) -> set[str] | None:
    """Column names removed by a ``.drop()`` call, or None if it is not a column drop."""
    columns = A.keyword_node(call, "columns")
    if columns is None:
        axis = A.const(A.keyword_node(call, "axis"))
        if axis not in (1, "columns"):
            return None
        columns = call.args[0] if call.args else None
    if columns is None:
        return None
    if isinstance(columns, ast.Constant) and isinstance(columns.value, str):
        return {columns.value}
    if isinstance(columns, ast.List | ast.Tuple):
        return {
            elt.value
            for elt in columns.elts
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
        }
    return set()


@register
class TargetInFeatureMatrix(Check):
    """LL001 - the target or a derived column survives into the design matrix."""

    id = "LL001"

    def run(self, ctx: FileContext) -> Iterable[Finding]:
        if not ctx.target_columns:
            return
        yield from self._explicit_lists(ctx)
        yield from self._subtractive(ctx)

    def _explicit_lists(self, ctx: FileContext) -> Iterable[Finding]:
        for node in ast.walk(ctx.tree):
            if not isinstance(node, ast.Assign) or node.value is None:
                continue
            labels = [A.target_label(t) for t in A.assign_targets(node)]
            if not any(label and FEATURE_LIST_NAMES.match(label) for label in labels):
                continue

            listing = node.value
            if isinstance(listing, ast.Subscript):
                listing = listing.slice
            if not isinstance(listing, ast.List | ast.Tuple):
                continue

            leaked = [
                elt.value
                for elt in listing.elts
                if isinstance(elt, ast.Constant)
                and isinstance(elt.value, str)
                and ctx.is_target_like(elt.value)
            ]
            if not leaked:
                continue

            column = leaked[0]
            lineno = A.span(node)[0]
            fix = replace_on_line(
                ctx, lineno, rf"""\s*["']{re.escape(column)}["']\s*,?""", ""
            ) or replace_on_line(ctx, lineno, rf""",\s*["']{re.escape(column)}["']""", "")
            yield self.finding(
                ctx,
                node,
                detail=(
                    f"the feature list assigned to `{labels[0]}` contains "
                    f"{', '.join(repr(c) for c in leaked)}, which the file also uses as the "
                    f"prediction target ({', '.join(sorted(ctx.target_columns))})"
                ),
                fix=fix,
                evidence={"leaked_columns": leaked, "targets": sorted(ctx.target_columns)},
            )

    def _subtractive(self, ctx: FileContext) -> Iterable[Finding]:
        columns = assigned_columns(ctx)
        derived = {
            name for name in columns if ctx.is_target_like(name) and name not in ctx.target_columns
        }
        for call in A.walk_calls(ctx.tree):
            if A.call_tail(call) != "drop":
                continue
            dropped = _dropped_columns(call)
            if dropped is None:
                continue
            assigned = ctx.assigned_label(call)
            if assigned and not FEATURE_LIST_NAMES.match(assigned):
                continue

            survivors = sorted((ctx.target_columns | derived) - dropped)
            if not survivors:
                continue

            lineno = A.span(call)[0]
            insertion = ", ".join(f'"{c}"' for c in survivors)
            fix = replace_on_line(ctx, lineno, r"(columns\s*=\s*\[)", rf"\1{insertion}, ")
            yield self.finding(
                ctx,
                call,
                detail=(
                    f"the drop removes {sorted(dropped) or 'nothing'} but leaves "
                    f"{survivors} in the frame, and those columns are the target or derived "
                    f"from it"
                ),
                fix=fix,
                evidence={"dropped": sorted(dropped), "survivors": survivors},
            )


@register
class FeatureFromTarget(Check):
    """LL002 - a feature column is computed from the target series."""

    id = "LL002"

    def run(self, ctx: FileContext) -> Iterable[Finding]:
        targets = ctx.target_columns
        if not targets:
            return
        features = feature_columns(ctx)

        for node in ast.walk(ctx.tree):
            if not isinstance(node, ast.Assign) or node.value is None:
                continue
            column = next(
                (A.target_label(t) for t in A.assign_targets(node) if isinstance(t, ast.Subscript)),
                None,
            )
            if column is None or ctx.is_target_like(column):
                continue

            referenced = A.column_references(node.value) & targets
            if not referenced:
                continue

            if features:
                if column not in features:
                    continue
                confidence = Confidence.HIGH
                why = f"`{column}` appears in the feature list"
            else:
                if ACCOUNTING_COLUMNS.search(column):
                    # Backtest accounting legitimately consumes the realised forward
                    # return; that is the PnL calculation, not a feature.
                    continue
                confidence = Confidence.MEDIUM
                why = (
                    f"`{column}` is not obviously accounting output, and the file has no "
                    f"explicit feature list to check against"
                )

            yield self.finding(
                ctx,
                node,
                confidence=confidence,
                detail=(
                    f"`{column}` is computed from {', '.join(repr(c) for c in sorted(referenced))}"
                    f", the prediction target; {why}"
                ),
                evidence={"column": column, "from_targets": sorted(referenced)},
            )


@register
class FitOnFrameWithTarget(Check):
    """LL003 - a transformer is fitted on a frame that still holds the target column."""

    id = "LL003"

    def run(self, ctx: FileContext) -> Iterable[Finding]:
        if not ctx.target_columns:
            return
        columns = assigned_columns(ctx)
        target_bearing = {name for name in columns if ctx.is_target_like(name)}
        if not target_bearing:
            return

        frames = self._frames_holding_target(ctx, target_bearing)
        if not frames:
            return

        for call in A.walk_calls(ctx.tree):
            if A.call_tail(call) not in {"fit", "fit_transform"}:
                continue
            recv = A.receiver(call)
            if recv is None or not TRANSFORMER_HINT.search(A.dotted(recv)):
                continue
            if not call.args or not isinstance(call.args[0], ast.Name):
                continue
            frame = call.args[0].id
            if frame not in frames:
                continue

            yield self.finding(
                ctx,
                call,
                detail=(
                    f"`{A.dotted(recv)}` is fitted on `{frame}`, which still carries "
                    f"{sorted(target_bearing)}; the target's variance enters every "
                    f"transformed feature through the shared covariance structure"
                ),
                evidence={"frame": frame, "target_columns": sorted(target_bearing)},
            )

    @staticmethod
    def _frames_holding_target(ctx: FileContext, target_bearing: set[str]) -> set[str]:
        frames: set[str] = set()
        for node in ast.walk(ctx.tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in A.assign_targets(node):
                if not isinstance(target, ast.Subscript):
                    continue
                key = A.subscript_key(target)
                if key in target_bearing and isinstance(target.value, ast.Name):
                    frames.add(target.value.id)
        return frames


__all__ = ["assigned_columns", "feature_columns"]
