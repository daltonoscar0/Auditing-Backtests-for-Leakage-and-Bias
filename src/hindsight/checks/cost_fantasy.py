"""COST-FANTASY checks: the backtest charges less than trading would.

Every check in this module is suppressed by the presence of a cost-like token
anywhere in the enclosing scope. That is deliberate. Hindsight can tell whether a
cost model exists; it cannot statically tell whether the number in it is
defensible. Flagging research that forgot costs entirely is a claim the tool can
support - grading someone's slippage assumption is not.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable

from .. import astutils as A
from ..context import BORROW_TERMS, COST_TERMS, FileContext
from ..fixes import replace_on_line
from ..models import Confidence, Finding
from .base import Check, register

PNL_TARGET = re.compile(
    r"^(pnl|pl|profit|profits|equity|strategy_ret\w*|strat_ret\w*|portfolio_ret\w*|"
    r"net_ret\w*|ret_net|gross_ret\w*|daily_ret\w*|returns?)$",
    re.IGNORECASE,
)
POSITION_TERM = re.compile(
    r"position|signal|weight|holding|exposure|allocation|book|^pos$|^w$", re.IGNORECASE
)
RETURN_TERM = re.compile(r"ret|return|pct_change|log_ret|chg|move|fwd_", re.IGNORECASE)

FILL_TARGET = re.compile(
    r"^(fill|fills|fill_price|exec_price|execution_price|entry_price|exit_price|"
    r"trade_price|px_fill|traded_price)$",
    re.IGNORECASE,
)
REFERENCE_PRICE = re.compile(
    r"^(close|adj_close|mid|midpoint|mid_price|last|px_last)$", re.IGNORECASE
)
SLIPPAGE_TERM = re.compile(r"spread|slippage|slip|impact|touch|bid|ask|tick_size", re.IGNORECASE)

SHORT_TERM = re.compile(r"short|sell_side", re.IGNORECASE)


def _tokens(node: ast.AST) -> set[str]:
    return A.referenced_names(node) | A.string_constants(node)


def _has_product(node: ast.AST) -> bool:
    return any(
        isinstance(child, ast.BinOp) and isinstance(child.op, ast.Mult) for child in ast.walk(node)
    )


@register
class UncostedPnL(Check):
    """CF001 - position times return, with nothing subtracted."""

    id = "CF001"

    def run(self, ctx: FileContext) -> Iterable[Finding]:
        for node in ast.walk(ctx.tree):
            if not isinstance(node, ast.Assign) or node.value is None:
                continue
            label = next(
                (lbl for lbl in (A.target_label(t) for t in A.assign_targets(node)) if lbl), None
            )
            if not label or not PNL_TARGET.match(label):
                continue
            if not _has_product(node.value):
                continue

            tokens = _tokens(node.value)
            positions = sorted(t for t in tokens if POSITION_TERM.search(t))
            returns = sorted(t for t in tokens if RETURN_TERM.search(t))
            if not positions or not returns:
                continue
            if ctx.scope_matches(node, COST_TERMS):
                continue

            yield self.finding(
                ctx,
                node,
                detail=(
                    f"`{label}` is `{positions[0]}` times `{returns[0]}` with no cost term "
                    f"anywhere in scope, so every rebalance in the backtest is free"
                ),
                evidence={"target": label, "positions": positions, "returns": returns},
            )


@register
class FrictionlessFill(Check):
    """CF002 - execution assumed at the closing print or the mid."""

    id = "CF002"

    def run(self, ctx: FileContext) -> Iterable[Finding]:
        for node in ast.walk(ctx.tree):
            if not isinstance(node, ast.Assign) or node.value is None:
                continue
            label = next(
                (lbl for lbl in (A.target_label(t) for t in A.assign_targets(node)) if lbl), None
            )
            if not label or not FILL_TARGET.match(label):
                continue

            references = sorted(t for t in _tokens(node.value) if REFERENCE_PRICE.match(t))
            if not references:
                continue
            if ctx.scope_matches(node, SLIPPAGE_TERM) or ctx.scope_matches(node, COST_TERMS):
                continue

            yield self.finding(
                ctx,
                node,
                detail=(
                    f"`{label}` is taken straight from `{references[0]}` with no spread or "
                    f"slippage adjustment in scope, so the backtest crosses no spread and moves "
                    f"no price"
                ),
                evidence={"target": label, "reference_price": references},
            )


@register
class ZeroCostParameter(Check):
    """CF003 - a cost parameter exists and is set to zero."""

    id = "CF003"

    def run(self, ctx: FileContext) -> Iterable[Finding]:
        for node in ast.walk(ctx.tree):
            if not isinstance(node, ast.Assign | ast.AnnAssign) or node.value is None:
                continue
            label = next(
                (lbl for lbl in (A.target_label(t) for t in A.assign_targets(node)) if lbl), None
            )
            if not label or not COST_TERMS.search(label):
                continue
            value = A.const(node.value)
            if not isinstance(value, int | float) or isinstance(value, bool) or value != 0:
                continue

            in_bps = bool(re.search(r"bps|basis", label, re.IGNORECASE))
            suggestion = "5.0" if in_bps else "0.0005"
            unit = "basis points, round trip" if in_bps else "fraction of notional, round trip"
            fix = replace_on_line(
                ctx,
                A.span(node)[0],
                rf"(\b{re.escape(label)}\b\s*(?::[^=]+)?=\s*)0(?:\.0*)?\b",
                rf"\g<1>{suggestion}",
            )
            yield self.finding(
                ctx,
                node,
                detail=(
                    f"`{label}` is a cost parameter pinned to {value}; the accounting around it "
                    f"is structurally correct, so the reported PnL is a gross figure being read "
                    f"as net. The suggested {suggestion} is a placeholder ({unit}) and must be "
                    f"replaced with a venue-specific estimate"
                ),
                fix=fix,
                evidence={"parameter": label, "value": value},
            )


@register
class UnborrowedShort(Check):
    """CF004 - short exposure with no stock-loan cost."""

    id = "CF004"

    def run(self, ctx: FileContext) -> Iterable[Finding]:
        if ctx.matches(BORROW_TERMS):
            return

        evidence = self._short_evidence(ctx)
        if evidence is None:
            return
        node, description = evidence

        yield self.finding(
            ctx,
            node,
            confidence=Confidence.MEDIUM,
            detail=(
                f"{description}, and no borrow, loan or financing term appears anywhere in the "
                f"file; the short book is being carried for free"
            ),
            evidence={"short_construction": description},
        )

    def _short_evidence(self, ctx: FileContext) -> tuple[ast.AST, str] | None:
        for node in ast.walk(ctx.tree):
            if isinstance(node, ast.Call):
                tail = A.call_tail(node)
                if tail == "clip" and A.const(A.keyword_node(node, "upper")) == 0:
                    return node, "positions are clipped to at most zero, making the book short-only"
                if tail == "sign":
                    label = ctx.assigned_label(node)
                    if label and POSITION_TERM.search(label):
                        return (
                            node,
                            f"`{label}` is built from `sign(...)`, so it takes -1 positions",
                        )
                if tail == "where":
                    values = {A.const(a) for a in node.args[1:]}
                    label = ctx.assigned_label(node)
                    if -1 in values and label and POSITION_TERM.search(label):
                        return node, f"`{label}` is assigned -1 for the short leg"

            if isinstance(node, ast.Assign) and node.value is not None:
                label = next(
                    (lbl for lbl in (A.target_label(t) for t in A.assign_targets(node)) if lbl),
                    None,
                )
                if not label:
                    continue
                if POSITION_TERM.search(label) and A.const(node.value) == -1:
                    return node, f"`{label}` is set to -1, an outright short"
                if SHORT_TERM.search(label):
                    return node, f"`{label}` names a short leg"
        return None
