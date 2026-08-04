"""LOOKAHEAD checks: future information reaching a feature indexed at time t."""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable

from .. import astutils as A
from ..context import TRANSFORMER_HINT, FileContext
from ..fixes import replace_on_line
from ..models import Finding
from .base import Check, register

WINDOWED = re.compile(r"rolling|expanding|ewm|groupby|resample|shift|cumsum|cumprod", re.IGNORECASE)
AGGREGATES = frozenset({"mean", "std", "min", "max", "median", "var", "quantile"})


@register
class NegativeShift(Check):
    """LA001 - ``.shift(-n)`` on anything that is not the target column."""

    id = "LA001"

    def run(self, ctx: FileContext) -> Iterable[Finding]:
        for call in A.walk_calls(ctx.tree):
            if A.call_tail(call) != "shift":
                continue
            periods = ctx.resolve_int(A.arg_or_keyword(call, 0, "periods"))
            if not isinstance(periods, int) or isinstance(periods, bool) or periods >= 0:
                continue

            assigned = ctx.assigned_label(call)
            if ctx.is_target_like(assigned):
                # A negative shift is how a forward-return label is built. That is
                # legitimate; whether it is purged out of the training folds is
                # CV004's problem, not a lookahead in its own right.
                continue

            detail = (
                f"`.shift({periods})` produces a value on row t that was observed "
                f"{-periods} bar(s) later"
            )
            if assigned:
                detail += f", and it is assigned to `{assigned}`, which is used as a feature"
            fix = replace_on_line(
                ctx,
                A.span(call)[0],
                r"(\.shift\(\s*(?:periods\s*=\s*)?)-\s*(\d+)",
                r"\1\2",
            )
            yield self.finding(
                ctx,
                call,
                detail=detail + ".",
                fix=fix,
                evidence={"periods": periods, "assigned_to": assigned},
            )


@register
class CenteredRolling(Check):
    """LA002 - ``rolling(..., center=True)``."""

    id = "LA002"

    def run(self, ctx: FileContext) -> Iterable[Finding]:
        for call in A.walk_calls(ctx.tree):
            if A.call_tail(call) != "rolling":
                continue
            center = A.keyword_node(call, "center")
            if not A.is_true(center):
                continue

            window = ctx.resolve_int(A.arg_or_keyword(call, 0, "window"))
            detail = "the window is centred on each row"
            if isinstance(window, int):
                detail += (
                    f", so roughly {window // 2} of the {window} bars behind the statistic "
                    f"at t occur after t"
                )
            lineno = A.span(call)[0]
            fix = replace_on_line(ctx, lineno, r",\s*center\s*=\s*True", "") or replace_on_line(
                ctx, lineno, r"center\s*=\s*True\s*,\s*", ""
            )
            yield self.finding(
                ctx,
                call,
                detail=detail + ".",
                fix=fix,
                evidence={"window": window, "assigned_to": ctx.assigned_label(call)},
            )


@register
class BackwardFill(Check):
    """LA003 - backward fill on a feature column."""

    id = "LA003"

    def run(self, ctx: FileContext) -> Iterable[Finding]:
        for call in A.walk_calls(ctx.tree):
            tail = A.call_tail(call)
            method = None
            if tail in {"bfill", "backfill"}:
                method = tail
            elif tail == "fillna":
                value = A.const(A.keyword_node(call, "method"))
                if value in {"bfill", "backfill"}:
                    method = str(value)
            if method is None:
                continue

            assigned = ctx.assigned_label(call)
            if ctx.is_target_like(assigned):
                continue

            detail = f"`{method}` carries a later observation backwards into earlier rows"
            if assigned:
                detail += f" of `{assigned}`"
            lineno = A.span(call)[0]
            fix = (
                replace_on_line(ctx, lineno, r"\.bfill\(", ".ffill(")
                or replace_on_line(ctx, lineno, r"\.backfill\(", ".ffill(")
                or replace_on_line(
                    ctx, lineno, r"(method\s*=\s*[\"'])(bfill|backfill)([\"'])", r"\1ffill\3"
                )
            )
            yield self.finding(
                ctx,
                call,
                detail=detail + ".",
                fix=fix,
                evidence={"method": method, "assigned_to": assigned},
            )


@register
class FitBeforeSplit(Check):
    """LA004 - a transformer is fitted at a line preceding the first split."""

    id = "LA004"

    def run(self, ctx: FileContext) -> Iterable[Finding]:
        split_line = ctx.first_split_line
        if split_line is None:
            return

        for call in A.walk_calls(ctx.tree):
            if A.call_tail(call) not in {"fit", "fit_transform"}:
                continue
            lineno = A.span(call)[0]
            if lineno >= split_line:
                continue

            recv = A.receiver(call)
            if recv is None:
                continue
            recv_name = A.dotted(recv)
            if not TRANSFORMER_HINT.search(recv_name):
                continue
            if id(call) in ctx.pipeline_transformers:
                continue
            if recv_name.split(".")[0] in ctx.pipeline_variables:
                continue

            split_text = ctx.line_text(split_line).strip()
            yield self.finding(
                ctx,
                call,
                detail=(
                    f"`{recv_name}` is fitted on line {lineno}, before the split on line "
                    f"{split_line} (`{split_text}`), so it has already seen the test period"
                ),
                evidence={"transformer": recv_name, "split_line": split_line},
            )


@register
class FullSampleNormalisation(Check):
    """LA005 - z-scoring or range-scaling with whole-history statistics."""

    id = "LA005"

    def run(self, ctx: FileContext) -> Iterable[Finding]:
        for node in ast.walk(ctx.tree):
            if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
                continue
            centre = self._centring_aggregate(ctx, node.left)
            if centre is None:
                continue
            scale = self._scale_aggregate(ctx, node.right)
            if scale is None:
                continue

            assigned = ctx.assigned_label(node)
            detail = (
                f"the centring uses `.{centre}()` and the scaling `.{scale}()` over the whole "
                f"column, so every row is normalised by statistics that include the future"
            )
            if assigned:
                detail += f"; the result is stored as `{assigned}`"
            yield self.finding(
                ctx,
                node,
                detail=detail + ".",
                evidence={"centre": centre, "scale": scale, "assigned_to": assigned},
            )

    @staticmethod
    def _whole_sample(ctx: FileContext, node: ast.AST) -> str | None:
        """Aggregate name if `node` is a whole-column aggregate, else None."""
        if not isinstance(node, ast.Call):
            return None
        tail = A.call_tail(node)
        if tail not in AGGREGATES:
            return None
        chain = A.dotted(node.func)
        if WINDOWED.search(chain):
            return None
        # `history.mean()` where `history = series.expanding()` is a trailing
        # statistic even though nothing in the call chain says so.
        if chain.split(".")[0] in ctx.windowed_names:
            return None
        return tail

    def _centring_aggregate(self, ctx: FileContext, node: ast.AST) -> str | None:
        """The numerator must be an explicit ``x - x.mean()`` style subtraction.

        Requiring the subtraction is what keeps a Sharpe ratio - ``ret.mean() /
        ret.std()`` - from being read as a normalisation.
        """
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Sub):
            return None
        return self._whole_sample(ctx, node.right)

    def _scale_aggregate(self, ctx: FileContext, node: ast.AST) -> str | None:
        direct = self._whole_sample(ctx, node)
        if direct is not None:
            return direct
        # Range scaling: (x - x.min()) / (x.max() - x.min())
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
            high = self._whole_sample(ctx, node.left)
            low = self._whole_sample(ctx, node.right)
            if high and low:
                return f"{high}()-.{low}"
        return None
