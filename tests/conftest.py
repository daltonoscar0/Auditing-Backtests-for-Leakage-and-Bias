"""Shared test helpers."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from hindsight import checks  # noqa: F401  - registers every check
from hindsight.checks.base import get_check
from hindsight.context import FileContext, parse
from hindsight.models import Finding

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def context(source: str, *, path: str = "strategy.py") -> FileContext:
    """Parse a dedented snippet into a FileContext."""
    text = textwrap.dedent(source).strip("\n") + "\n"
    ctx = parse(Path(path), text, path)
    assert ctx is not None, "snippet does not parse"
    return ctx


def run_check(check_id: str, source: str, *, path: str = "strategy.py") -> list[Finding]:
    """Run a single check over a snippet, in isolation from the others."""
    return list(get_check(check_id).run(context(source, path=path)))


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES
