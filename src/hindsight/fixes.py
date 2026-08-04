"""Unified-diff generation for the checks whose remedy is mechanical.

Hindsight never applies a patch - it is read-only with respect to the audited
tree. These diffs exist so a reviewer can see the exact edit being proposed and
apply it themselves with ``git apply``.

A fix is only offered where the edit is unambiguous and local: a keyword flipped,
a literal replaced, an argument added. Anything requiring the author to decide a
number (a cost estimate, an embargo length) is left to the prose remedy in the
finding, with a placeholder in the diff where a judgement call is needed.
"""

from __future__ import annotations

import difflib
import re

from .context import FileContext


def unified(ctx: FileContext, replacements: dict[int, str], *, context_lines: int = 2) -> str:
    """Render a unified diff replacing the given 1-based lines with new text."""
    if not replacements or ctx.is_notebook:
        # A notebook's lines on disk are JSON string fragments, so a diff against
        # the flattened source would not apply to anything. The prose remedy on
        # the finding still stands.
        return ""
    old = list(ctx.lines)
    new = list(ctx.lines)
    for lineno, text in replacements.items():
        if not 1 <= lineno <= len(new):
            return ""
        new[lineno - 1] = text
    if old == new:
        return ""
    diff = difflib.unified_diff(
        old,
        new,
        fromfile=f"a/{ctx.path}",
        tofile=f"b/{ctx.path}",
        n=context_lines,
        lineterm="",
    )
    return "\n".join(diff)


def replace_on_line(ctx: FileContext, lineno: int, pattern: str, replacement: str) -> str | None:
    """Regex-replace the first match on one line and return the resulting diff.

    Returns None when the pattern does not match, which happens whenever the
    construct spans lines or is written in a form the fix was not designed for.
    A missing fix is fine; a wrong one is not.
    """
    text = ctx.line_text(lineno)
    if text == "":
        return None
    new_text, count = re.subn(pattern, replacement, text, count=1)
    if count == 0 or new_text == text:
        return None
    return unified(ctx, {lineno: new_text}) or None
