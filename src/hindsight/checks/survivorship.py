"""SURVIVORSHIP checks: the sample was chosen using knowledge of what happened to it."""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable

from .. import astutils as A
from ..context import FileContext
from ..models import Confidence, Finding
from .base import Check, register

CURRENT_UNIVERSE = re.compile(
    r"current[_-]?(constituents|universe|members|membership|holdings|list|tickers|sp500|"
    r"index|names)|"
    r"(latest|todays?|now)[_-]?(constituents|universe|holdings|members|tickers)|"
    r"(sp500|spx|russell\d*|nasdaq\d*|universe|tickers|constituents)[_-]?(current|today|latest|now)|"
    r"active[_-]?constituents|as[_-]?of[_-]?today",
    re.IGNORECASE,
)
UNIVERSE_VARIABLE = re.compile(
    r"^(universe|tickers|symbols|constituents|members|names|securities|instruments)$",
    re.IGNORECASE,
)
STATUS_COLUMN = re.compile(
    r"^(is_)?(active|inactive|listed|delisted|alive|dead|live|still_trading|trading|status|"
    r"currently_listed|in_index)$",
    re.IGNORECASE,
)
OUTCOME_COLUMN = re.compile(
    r"^(total|cum|cumulative|final|lifetime|overall|full|end|ever|all_time)[_-]|"
    r"[_-](total|cumulative|lifetime|ever|to_date)$|"
    r"total_return|final_price|survived|survivor",
    re.IGNORECASE,
)
WHOLE_SAMPLE_AGGREGATES = frozenset({"sum", "mean", "max", "min", "last", "count", "prod"})
TICKER_LITERAL = re.compile(r"^[A-Z][A-Z0-9.\-]{0,5}$")


def _is_mask(node: ast.expr) -> bool:
    """Whether a subscript slice looks like a boolean row filter rather than a column pick."""
    if isinstance(node, ast.Subscript):
        return A.subscript_key(node) is not None
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return True
    return any(
        isinstance(child, ast.Compare | ast.BoolOp)
        or (isinstance(child, ast.UnaryOp) and isinstance(child.op, ast.Not))
        or (isinstance(child, ast.BinOp) and isinstance(child.op, ast.BitAnd | ast.BitOr))
        for child in ast.walk(node)
    )


def _masks(ctx: FileContext) -> Iterable[tuple[ast.Subscript, ast.expr]]:
    for node in ast.walk(ctx.tree):
        if isinstance(node, ast.Subscript) and _is_mask(node.slice):
            yield node, node.slice


@register
class CurrentConstituents(Check):
    """SV001 - a present-day universe applied to historical data."""

    id = "SV001"

    def run(self, ctx: FileContext) -> Iterable[Finding]:
        # Report once per distinct offending name. A helper called
        # `current_constituents` matches at its definition and again at every call
        # site, and three citations of one mistake is two too many.
        reported: set[str] = set()
        seen: set[int] = set()

        for node in ast.walk(ctx.tree):
            token = self._token(node)
            if token is None or not CURRENT_UNIVERSE.search(token):
                continue
            if token.lower() in reported:
                continue
            reported.add(token.lower())
            stmt = ctx.enclosing_statement(node) or node
            seen.add(id(stmt))
            yield self.finding(
                ctx,
                stmt,
                confidence=Confidence.MEDIUM,
                detail=(
                    f"`{token}` names a present-day universe, and it is being used to select "
                    f"rows from a historical panel; names that were delisted or removed from "
                    f"the index never enter the sample"
                ),
                evidence={"token": token, "match": "name"},
            )

        yield from self._hardcoded_lists(ctx, seen)

    @staticmethod
    def _token(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            return node.name
        return None

    def _hardcoded_lists(self, ctx: FileContext, seen: set[int]) -> Iterable[Finding]:
        if not ctx.is_temporal:
            return
        for node in ast.walk(ctx.tree):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.List | ast.Tuple):
                continue
            label = next(
                (lbl for lbl in (A.target_label(t) for t in A.assign_targets(node)) if lbl), None
            )
            if not label or not UNIVERSE_VARIABLE.match(label):
                continue
            tickers = [
                elt.value
                for elt in node.value.elts
                if isinstance(elt, ast.Constant)
                and isinstance(elt.value, str)
                and TICKER_LITERAL.match(elt.value)
            ]
            if len(tickers) < 3 or id(node) in seen:
                continue
            seen.add(id(node))
            yield self.finding(
                ctx,
                node,
                confidence=Confidence.MEDIUM,
                detail=(
                    f"`{label}` is a hard-coded list of {len(tickers)} tickers "
                    f"({', '.join(tickers[:5])}{'...' if len(tickers) > 5 else ''}) applied to a "
                    f"time-indexed panel; a list written today contains only names that still "
                    f"exist today"
                ),
                evidence={"variable": label, "tickers": tickers, "match": "literal-list"},
            )


@register
class DelistedFiltered(Check):
    """SV002 - the historical panel is filtered on a listing-status flag."""

    id = "SV002"

    def run(self, ctx: FileContext) -> Iterable[Finding]:
        for node, mask in _masks(ctx):
            columns = A.column_references(mask)
            hits = sorted(c for c in columns if STATUS_COLUMN.match(c))
            if not hits:
                continue
            yield self.finding(
                ctx,
                node,
                confidence=Confidence.MEDIUM,
                detail=(
                    f"rows are filtered on {', '.join(repr(c) for c in hits)}, a listing-status "
                    f"flag; applied across history this drops precisely the names whose outcomes "
                    f"were worst"
                ),
                evidence={"status_columns": hits},
            )

        for call in A.walk_calls(ctx.tree):
            if A.call_tail(call) != "query" or not call.args:
                continue
            expression = A.const(call.args[0])
            if not isinstance(expression, str):
                continue
            hits = sorted(
                {
                    token
                    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expression)
                    if STATUS_COLUMN.match(token)
                }
            )
            if not hits:
                continue
            yield self.finding(
                ctx,
                call,
                confidence=Confidence.MEDIUM,
                detail=(
                    f"the query string filters on {', '.join(repr(c) for c in hits)}, a "
                    f"listing-status flag evaluated as of today"
                ),
                evidence={"status_columns": hits, "query": expression},
            )


@register
class OutcomeSelectedUniverse(Check):
    """SV003 - membership decided by a statistic spanning the whole sample."""

    id = "SV003"

    def run(self, ctx: FileContext) -> Iterable[Finding]:
        for node, mask in _masks(ctx):
            columns = A.column_references(mask)
            named = sorted(c for c in columns if OUTCOME_COLUMN.search(c))
            aggregate = self._whole_sample_aggregate(mask)

            if named:
                yield self.finding(
                    ctx,
                    node,
                    confidence=Confidence.MEDIUM,
                    detail=(
                        f"membership depends on {', '.join(repr(c) for c in named)}, which "
                        f"summarises a name's entire history; a row's presence in the sample "
                        f"therefore depends on what it went on to do"
                    ),
                    evidence={"outcome_columns": named},
                )
            elif aggregate is not None:
                yield self.finding(
                    ctx,
                    node,
                    confidence=Confidence.HIGH,
                    detail=(
                        f"the filter is built from `{aggregate}`, an aggregate over each group's "
                        f"full history, so selection uses information that only exists at the "
                        f"end of the backtest"
                    ),
                    evidence={"aggregate": aggregate},
                )

    @staticmethod
    def _whole_sample_aggregate(mask: ast.expr) -> str | None:
        for call in A.walk_calls(mask):
            chain = A.dotted(call.func)
            if "groupby" not in chain:
                continue
            tail = A.call_tail(call)
            if tail == "transform":
                inner = A.const(call.args[0]) if call.args else None
                if isinstance(inner, str) and inner in WHOLE_SAMPLE_AGGREGATES:
                    return f"{chain}('{inner}')"
                return f"{chain}(...)"
            if tail in WHOLE_SAMPLE_AGGREGATES:
                return f"{chain}()"
        return None
