"""Jupyter notebook support.

A lot of trading research lives in ``.ipynb``, and a notebook is where the worst
of it lives, because cells run in whatever order someone happened to press the
button in and nothing is ever refactored.

Cells are concatenated into one synthetic module so analysis works across them -
a target defined in cell 2 and used in cell 7 is the normal case, and treating
each cell as its own file would miss all of it. A line map records where every
synthetic line came from, so a finding cites ``cell 3, line 12``: the only
coordinates a reader looking at the notebook can act on.

Cells that do not parse - IPython magics, shell escapes, incomplete drafts - are
replaced by blank lines rather than dropped, which keeps every other cell's line
numbers honest.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass

MAGIC_PREFIXES = ("%", "!", "?")


@dataclass(frozen=True, slots=True)
class Origin:
    """Where a synthetic line came from."""

    cell: int
    line: int


def _cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    return source if isinstance(source, str) else "".join(source)


def _neutralise(line: str) -> str:
    """Blank out a line Python cannot parse, preserving the line count.

    IPython magics and shell escapes are not Python and never will be. Replacing
    them with a comment keeps every subsequent line at the number a reader sees
    in the notebook.
    """
    stripped = line.lstrip()
    if stripped.startswith(MAGIC_PREFIXES):
        return "#" + line[1:] if line.startswith(("%", "!")) else f"# {line}"
    return line


def flatten(payload: str) -> tuple[str, list[Origin]] | None:
    """Turn notebook JSON into a synthetic module plus a per-line origin map.

    Returns None if the payload is not a notebook at all.
    """
    try:
        document = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(document, dict) or "cells" not in document:
        return None

    lines: list[str] = []
    origins: list[Origin] = []

    for index, cell in enumerate(document.get("cells", []), start=1):
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        source = _cell_source(cell)
        if not source.strip():
            continue

        cell_lines = [_neutralise(line) for line in source.splitlines()]
        block = "\n".join(cell_lines)
        if not _parses(block):
            # Keep the shape of the notebook without letting a broken cell take
            # the rest of it down.
            cell_lines = ["" for _ in cell_lines]

        for offset, line in enumerate(cell_lines, start=1):
            lines.append(line)
            origins.append(Origin(cell=index, line=offset))

        # A blank separator so the last statement of one cell and the first of
        # the next are not read as a continuation.
        lines.append("")
        origins.append(Origin(cell=index, line=len(cell_lines) + 1))

    return "\n".join(lines) + "\n", origins


def _parses(source: str) -> bool:
    try:
        ast.parse(source)
    except SyntaxError:
        return False
    return True
