"""Small helpers over the stdlib ``ast`` module.

Deliberately boring. Every check in ``hindsight.checks`` is written against these
so that the awkward parts of the AST - dotted call names, keyword lookup,
constant folding of unary minus - are handled once and identically.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from typing import Any

_MISSING = object()


def dotted(node: ast.AST) -> str:
    """Render an attribute/name chain as a dotted string.

    ``df.close.rolling`` -> ``"df.close.rolling"``. Anything that is not a pure
    chain of names and attributes yields ``""`` for the unresolvable part, so
    ``df["c"].shift`` becomes ``".shift"`` - the tail is still usable.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{dotted(node.value)}.{node.attr}"
    if isinstance(node, ast.Call):
        return dotted(node.func)
    if isinstance(node, ast.Subscript):
        return dotted(node.value)
    return ""


def call_name(node: ast.Call) -> str:
    """Full dotted name of what is being called."""
    return dotted(node.func)


def call_tail(node: ast.Call) -> str:
    """Final segment of the callee: ``scaler.fit_transform`` -> ``fit_transform``."""
    return call_name(node).rsplit(".", 1)[-1]


def receiver(node: ast.Call) -> ast.expr | None:
    """The object a method is called on, i.e. ``df`` in ``df.shift(1)``."""
    if isinstance(node.func, ast.Attribute):
        return node.func.value
    return None


def keyword_node(node: ast.Call, name: str) -> ast.expr | None:
    for kw in node.keywords:
        if kw.arg == name:
            return kw.value
    return None


def has_keyword(node: ast.Call, name: str) -> bool:
    return any(kw.arg == name for kw in node.keywords)


def const(node: ast.AST | None) -> Any:
    """Fold a literal, including a negated numeric literal. Returns ``_MISSING`` sentinel-free
    ``None`` only when the node genuinely is ``None``; use :func:`is_const` to disambiguate."""
    value = _const_or_missing(node)
    return None if value is _MISSING else value


def is_const(node: ast.AST | None) -> bool:
    return _const_or_missing(node) is not _MISSING


def _const_or_missing(node: ast.AST | None) -> Any:
    if node is None:
        return _MISSING
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _const_or_missing(node.operand)
        if isinstance(inner, int | float) and not isinstance(inner, bool):
            return -inner
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        inner = _const_or_missing(node.operand)
        if isinstance(inner, int | float) and not isinstance(inner, bool):
            return +inner
    return _MISSING


def arg_or_keyword(node: ast.Call, position: int, name: str) -> ast.expr | None:
    """Fetch an argument that may be given positionally or by keyword."""
    kw = keyword_node(node, name)
    if kw is not None:
        return kw
    if len(node.args) > position and not isinstance(node.args[position], ast.Starred):
        return node.args[position]
    return None


def is_true(node: ast.AST | None) -> bool:
    return _const_or_missing(node) is True


def string_constants(node: ast.AST) -> set[str]:
    """Every string literal appearing anywhere inside `node`."""
    return {
        child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    }


def referenced_names(node: ast.AST) -> set[str]:
    """Identifiers and attribute names appearing anywhere inside `node`."""
    found: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            found.add(child.id)
        elif isinstance(child, ast.Attribute):
            found.add(child.attr)
    return found


def column_references(node: ast.AST) -> set[str]:
    """Names that plausibly refer to dataframe columns inside `node`.

    Union of string literals (``df["ret"]``, ``columns=["ret"]``) and attribute
    accesses on a bare name (``df.ret``). Over-inclusive by design: callers pair
    it with a specific column name they are looking for.
    """
    found = string_constants(node)
    for child in ast.walk(node):
        if isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name):
            found.add(child.attr)
    return found


def subscript_key(node: ast.Subscript) -> str | None:
    """The literal string key of ``df["close"]``, or None if it is not a string literal."""
    value = _const_or_missing(node.slice)
    return value if isinstance(value, str) else None


def assign_targets(node: ast.Assign | ast.AnnAssign | ast.AugAssign) -> list[ast.expr]:
    if isinstance(node, ast.Assign):
        return list(node.targets)
    return [node.target]


def target_label(node: ast.expr) -> str | None:
    """A readable name for an assignment target.

    ``pnl`` for ``pnl = ...``, ``pnl`` for ``df["pnl"] = ...``, ``pnl`` for
    ``df.pnl = ...``. Tuple targets yield None; checks that care handle them
    element-wise.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        key = subscript_key(node)
        if key is not None:
            return key
        return None
    return None


def walk_calls(tree: ast.AST) -> Iterator[ast.Call]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            yield node


def span(node: ast.AST) -> tuple[int, int]:
    """1-based inclusive (start, end) line span of `node`."""
    start = getattr(node, "lineno", 1)
    end = getattr(node, "end_lineno", None) or start
    return (start, end)


def attach_parents(tree: ast.Module) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    return parents


def ancestors(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> Iterator[ast.AST]:
    current = parents.get(node)
    while current is not None:
        yield current
        current = parents.get(current)
