"""End-to-end behaviour of the `hindsight` command."""

from __future__ import annotations

import json

import pytest

from hindsight.cli import EXIT_ERROR, EXIT_FINDINGS, EXIT_OK, main


def test_markdown_report_names_the_findings(
    fixtures_dir, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["audit", str(fixtures_dir / "mixed")])
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert "# Hindsight audit" in out
    assert "## Summary" in out
    for check_id in ("LA001", "LL001", "CV002", "CF001", "SO001", "SV001"):
        assert check_id in out
    assert "Why this matters" in out
    assert "```diff" in out, "the mixed fixture should yield at least one mechanical patch"


def test_json_report_is_valid_and_pinned(fixtures_dir, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["audit", str(fixtures_dir / "unpurged_cv"), "--format", "json"])
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_OK
    assert payload["tool"] == "hindsight"
    assert payload["taxonomy_version"]
    assert payload["summary"]["total"] == 4
    assert payload["summary"]["max_severity"] == "critical"
    assert {f["check_id"] for f in payload["findings"]} == {"CV001", "CV002", "CV003", "CV004"}
    for finding in payload["findings"]:
        assert finding["file"] and finding["line"] >= 1
        assert finding["mechanism"]


def test_clean_repo_reports_nothing(fixtures_dir, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["audit", str(fixtures_dir / "clean")])
    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "## No findings" in out
    assert "does not mean" not in out.lower() or "not a statement" in out


def test_fail_on_gates_the_exit_code(fixtures_dir, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["audit", str(fixtures_dir / "mixed"), "--fail-on", "critical"]) == EXIT_FINDINGS
    capsys.readouterr()

    assert main(["audit", str(fixtures_dir / "clean"), "--fail-on", "low"]) == EXIT_OK
    capsys.readouterr()

    # cost_fantasy's worst finding is critical, so a medium threshold trips too
    assert main(["audit", str(fixtures_dir / "cost_fantasy"), "--fail-on", "medium"]) == (
        EXIT_FINDINGS
    )
    capsys.readouterr()


def test_only_restricts_the_checks(fixtures_dir, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["audit", str(fixtures_dir / "mixed"), "--format", "json", "--only", "CF001"])
    payload = json.loads(capsys.readouterr().out)
    assert code == EXIT_OK
    assert payload["checks_run"] == ["CF001"]
    assert {f["check_id"] for f in payload["findings"]} == {"CF001"}


def test_unknown_check_id_is_rejected(fixtures_dir, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["audit", str(fixtures_dir / "mixed"), "--only", "XX999"]) == EXIT_ERROR
    assert "XX999" in capsys.readouterr().err


def test_missing_path_is_rejected(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["audit", str(tmp_path / "nope")]) == EXIT_ERROR
    assert capsys.readouterr().err


def test_out_writes_the_report(fixtures_dir, tmp_path) -> None:
    destination = tmp_path / "report.md"
    assert main(["audit", str(fixtures_dir / "label_leak"), "--out", str(destination)]) == EXIT_OK
    assert "LL001" in destination.read_text(encoding="utf-8")


def test_checks_subcommand_lists_the_registry(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["checks"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "LA001" in out and "SO004" in out

    assert main(["checks", "--format", "json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["checks"]) == 23


def test_llm_replay_runs_offline_and_reports_usage(
    fixtures_dir, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "audit",
            str(fixtures_dir / "llm_only"),
            "--llm-replay",
            str(fixtures_dir / "llm_only" / "cassette.json"),
            "--format",
            "json",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert code == EXIT_OK
    assert payload["llm_enabled"] is True
    assert payload["summary"]["total"] == 11
    assert {"LA006", "LL004", "SV004", "CV005", "CF005", "SO005"} <= {
        f["check_id"] for f in payload["findings"]
    }
    assert payload["llm_usage"]["requests"] == 10
    assert payload["llm_usage"]["skipped_requests"] == 2
    assert "model pass:" in captured.err


def test_llm_record_writes_a_cassette(fixtures_dir, tmp_path, capsys) -> None:
    destination = tmp_path / "recorded.json"
    code = main(
        [
            "audit",
            str(fixtures_dir / "llm_only"),
            "--llm",
            "--provider",
            "mock",
            "--llm-record",
            str(destination),
            "--format",
            "json",
        ]
    )
    capsys.readouterr()

    assert code == EXIT_OK
    recorded = json.loads(destination.read_text(encoding="utf-8"))
    assert recorded["provenance"] == "live"
    assert len(recorded["responses"]) == 10
    assert "pipeline.py::LOOKAHEAD" in recorded["responses"]


def test_a_missing_cassette_is_an_error(fixtures_dir, tmp_path, capsys) -> None:
    assert (
        main(["audit", str(fixtures_dir / "mixed"), "--llm-replay", str(tmp_path / "nope.json")])
        == EXIT_ERROR
    )
    assert capsys.readouterr().err


def test_llm_all_categories_disables_the_gate(fixtures_dir, capsys) -> None:
    main(
        [
            "audit",
            str(fixtures_dir / "llm_only"),
            "--llm",
            "--provider",
            "mock",
            "--llm-all-categories",
            "--format",
            "json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["llm_usage"]["requests"] == 12
    assert payload["llm_usage"]["skipped_requests"] == 0


def test_baseline_round_trip(fixtures_dir, tmp_path, capsys) -> None:
    destination = tmp_path / "baseline.json"

    assert (
        main(["audit", str(fixtures_dir / "mixed"), "--write-baseline", str(destination)])
        == EXIT_OK
    )
    assert "9 finding(s)" in capsys.readouterr().err

    code = main(
        ["audit", str(fixtures_dir / "mixed"), "--baseline", str(destination), "--format", "json"]
    )
    captured = capsys.readouterr()
    assert code == EXIT_OK
    assert json.loads(captured.out)["summary"]["total"] == 0
    assert "9 finding(s) suppressed" in captured.err


def test_a_baseline_survives_edits_elsewhere_in_the_file(fixtures_dir, tmp_path, capsys) -> None:
    """Fingerprints key on the cited text, so inserting a line above a finding
    must not resurrect it."""
    import shutil

    target = tmp_path / "repo"
    shutil.copytree(fixtures_dir / "mixed", target)
    destination = tmp_path / "baseline.json"
    main(["audit", str(target), "--write-baseline", str(destination)])
    capsys.readouterr()

    features = target / "features.py"
    features.write_text("# a new comment at the top\n" + features.read_text(), encoding="utf-8")

    main(["audit", str(target), "--baseline", str(destination), "--format", "json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["total"] == 0, (
        "a line-numbered baseline would have gone stale here; a fingerprinted one does not"
    )


def test_a_new_finding_survives_the_baseline(fixtures_dir, tmp_path, capsys) -> None:
    import shutil

    target = tmp_path / "repo"
    shutil.copytree(fixtures_dir / "mixed", target)
    destination = tmp_path / "baseline.json"
    main(["audit", str(target), "--write-baseline", str(destination)])
    capsys.readouterr()

    (target / "extra.py").write_text(
        'import pandas as pd\n\npanel["peek"] = panel["close"].shift(-1)\n', encoding="utf-8"
    )

    main(["audit", str(target), "--baseline", str(destination), "--format", "json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["total"] == 1
    assert payload["findings"][0]["file"] == "extra.py"


def test_an_unreadable_baseline_is_an_error(fixtures_dir, tmp_path, capsys) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert main(["audit", str(fixtures_dir / "mixed"), "--baseline", str(broken)]) == EXIT_ERROR
    assert capsys.readouterr().err
