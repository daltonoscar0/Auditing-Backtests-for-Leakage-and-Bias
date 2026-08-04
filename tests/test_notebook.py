"""Notebook support: cells analysed together, cited separately."""

from __future__ import annotations

import json
from pathlib import Path

from hindsight import audit
from hindsight.collector import collect
from hindsight.notebook import flatten


def _notebook(*cells: tuple[str, str]) -> str:
    document = {
        "cells": [
            {
                "cell_type": kind,
                "metadata": {},
                "source": source.splitlines(keepends=True),
                **({"execution_count": None, "outputs": []} if kind == "code" else {}),
            }
            for kind, source in cells
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(document)


def test_flatten_maps_every_line_back_to_its_cell() -> None:
    source, origins = flatten(
        _notebook(("code", "a = 1\nb = 2"), ("markdown", "# ignored"), ("code", "c = 3"))
    )
    lines = source.splitlines()

    assert lines[0] == "a = 1"
    assert (origins[0].cell, origins[0].line) == (1, 1)
    assert lines[1] == "b = 2"
    assert (origins[1].cell, origins[1].line) == (1, 2)

    # Markdown is skipped, so the next code cell is cell 3, its own line 1.
    index = lines.index("c = 3")
    assert (origins[index].cell, origins[index].line) == (3, 1)


def test_magics_do_not_break_the_parse() -> None:
    source, _ = flatten(_notebook(("code", "%matplotlib inline\nimport pandas as pd")))
    assert "import pandas as pd" in source
    assert "%matplotlib" not in source


def test_an_unparseable_cell_is_blanked_not_dropped() -> None:
    """Blanking preserves every later cell's line numbers."""
    source, origins = flatten(
        _notebook(("code", "this is not python at all ((("), ("code", "x = 1"))
    )
    lines = source.splitlines()
    assert lines[0] == ""
    index = lines.index("x = 1")
    assert (origins[index].cell, origins[index].line) == (2, 1)


def test_non_notebook_json_is_rejected() -> None:
    assert flatten('{"not": "a notebook"}') is None
    assert flatten("not json at all") is None


def test_findings_are_cited_by_cell(fixtures_dir: Path) -> None:
    result = audit(fixtures_dir / "notebook")
    assert result.files_scanned == 1
    assert not result.errors

    by_check = {f.check_id: f.location for f in result.findings}
    assert (by_check["LA001"].cell, by_check["LA001"].line) == (4, 2)
    assert (by_check["LA002"].cell, by_check["LA002"].line) == (4, 3)
    assert by_check["CV002"].cell == 6
    assert str(by_check["LA001"]) == "research.ipynb:cell 4:2"


def test_analysis_spans_cells(fixtures_dir: Path) -> None:
    """The target is defined in one cell and the feature list in another.

    A per-cell analysis would see neither, which is why cells are concatenated
    rather than treated as separate files.
    """
    contexts, _ = collect(fixtures_dir / "notebook")
    ctx = contexts[0]
    assert ctx.is_notebook
    assert "fwd_ret" in ctx.target_columns
    assert ctx.is_temporal


def test_no_diff_is_offered_for_a_notebook(fixtures_dir: Path) -> None:
    """A patch against flattened cells would not apply to the JSON on disk."""
    result = audit(fixtures_dir / "notebook")
    assert result.findings
    assert all(f.fix is None for f in result.findings)


def test_the_audited_notebook_is_not_modified(fixtures_dir: Path) -> None:
    path = fixtures_dir / "notebook" / "research.ipynb"
    before = path.read_bytes()
    audit(fixtures_dir / "notebook")
    assert path.read_bytes() == before
