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


def _static_fixtures() -> list[str]:
    return [
        name
        for name in run_eval.discover_fixtures()
        if run_eval.score_fixture(name, tolerance=0).layer == "static"
    ]


def test_static_recall_holds_at_zero_tolerance() -> None:
    """Every static check cites the exact line, not merely the right neighbourhood.

    Scoped to the static layer deliberately. A model chooses its own anchor within
    a defect that spans several statements, and holding it to a line the fixture
    author picked would measure agreement about citation style rather than whether
    the defect was found. The eval reports those as ANCHOR.
    """
    for name in _static_fixtures():
        result = run_eval.score_fixture(name, tolerance=0)
        assert not result.missed, (
            f"{name} loses findings at zero line tolerance: "
            f"{[(e.check_id, e.file, e.line) for e in result.missed]}"
        )


def test_static_check_ids_match_the_planted_bugs() -> None:
    """Category matching is what the eval scores; the check id should agree too."""
    for name in _static_fixtures():
        result = run_eval.score_fixture(name, tolerance=run_eval.DEFAULT_TOLERANCE)
        assert not result.id_mismatches, (
            f"{name}: {[(e.check_id, f.check_id) for e, f in result.id_mismatches]}"
        )


def test_the_model_layer_is_measured_but_does_not_gate_the_build() -> None:
    """A Layer 2 shortfall against a model-generated cassette is a result, not a
    regression, so it is reported without failing the build."""
    result = run_eval.score_fixture("llm_only", tolerance=run_eval.DEFAULT_TOLERANCE)
    assert result.layer == "llm"
    assert result.provenance in {"subagent", "live"}
    assert len(result.matched) + len(result.missed) == len(result.expectations)
    # Every expectation was either matched or anchored elsewhere: the model found
    # all six defects, and disagreed with the fixture about where to cite two.
    assert len(result.matched) + len(result.anchor_mismatches) == len(result.expectations)


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
