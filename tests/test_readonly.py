"""The audited tree must be byte-identical before and after an audit.

This is the one property the tool cannot be allowed to get wrong. Someone points
Hindsight at a repository they have not committed yet; if it writes anything -
a cache, a report, a stray `__pycache__` - that is a bug with real consequences.
"""

from __future__ import annotations

import pytest

from hindsight import audit
from hindsight.llm import MockProvider
from hindsight.readonly import ReadOnlyViolation, read_text, tree_digest


def _fixture_names(fixtures_dir):
    return sorted(p.name for p in fixtures_dir.iterdir() if p.is_dir())


def test_audit_does_not_modify_the_target(fixtures_dir) -> None:
    for name in _fixture_names(fixtures_dir):
        target = fixtures_dir / name
        before = tree_digest(target)
        audit(target)
        assert tree_digest(target) == before, f"audit modified {name}"


def test_audit_with_the_model_pass_does_not_modify_the_target(fixtures_dir) -> None:
    target = fixtures_dir / "mixed"
    before = tree_digest(target)
    audit(target, provider=MockProvider())
    assert tree_digest(target) == before


def test_read_text_refuses_a_writable_mode(tmp_path) -> None:
    path = tmp_path / "sample.py"
    path.write_text("x = 1\n", encoding="utf-8")

    assert read_text(path) == "x = 1\n"
    for mode in ("w", "a", "r+", "w+", "x"):
        with pytest.raises(ReadOnlyViolation):
            read_text(path, mode=mode)


def test_tree_digest_notices_a_change(tmp_path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    before = tree_digest(tmp_path)

    (tmp_path / "a.py").write_text("x = 2\n", encoding="utf-8")
    assert tree_digest(tmp_path) != before

    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    assert tree_digest(tmp_path) == before

    (tmp_path / "b.py").write_text("y = 1\n", encoding="utf-8")
    assert tree_digest(tmp_path) != before
