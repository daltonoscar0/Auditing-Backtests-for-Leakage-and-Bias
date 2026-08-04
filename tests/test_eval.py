"""The eval harness is part of the build, not a script someone runs by hand."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
sys.path.insert(0, str(EVAL_DIR))

import run_eval  # noqa: E402


def test_eval_meets_its_targets(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = run_eval.main([])
    output = capsys.readouterr().out
    assert exit_code == 0, output
    assert "PASS" in output


def test_clean_fixture_produces_nothing() -> None:
    result = run_eval.score_fixture("clean", tolerance=0)
    assert result.findings == [], (
        "the clean fixture must stay clean; these are false positives: "
        f"{[(f.check_id, str(f.location)) for f in result.findings]}"
    )


def test_recall_holds_at_zero_tolerance() -> None:
    """Every check should cite the right line, not merely the right neighbourhood."""
    for name in run_eval.discover_fixtures():
        result = run_eval.score_fixture(name, tolerance=0)
        assert not result.missed, (
            f"{name} loses findings at zero line tolerance: "
            f"{[(e.check_id, e.file, e.line) for e in result.missed]}"
        )


def test_check_ids_match_the_planted_bugs() -> None:
    """Category matching is what the eval scores; the check id should agree too."""
    for name in run_eval.discover_fixtures():
        result = run_eval.score_fixture(name, tolerance=run_eval.DEFAULT_TOLERANCE)
        assert not result.id_mismatches, (
            f"{name}: {[(e.check_id, f.check_id) for e, f in result.id_mismatches]}"
        )


def test_json_output_is_machine_readable(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = run_eval.main(["--json"])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["false_positives_on_clean"] == 0
    assert payload["overall_recall"] == 1.0
    assert set(payload["by_category"]) == {
        "LOOKAHEAD",
        "LABEL-LEAK",
        "SURVIVORSHIP",
        "UNPURGED-CV",
        "COST-FANTASY",
        "SWEEP-OVERFIT",
    }
    for category, counts in payload["by_category"].items():
        assert counts["expected"] > 0, f"no planted bugs for {category}"
