"""The model pass, its provider abstraction, and the citation validator.

None of these tests reach the network. The mock provider is the default so that
`pytest` and `make eval` run with no credentials and no egress.
"""

from __future__ import annotations

import json

import pytest
from conftest import context

from hindsight import audit
from hindsight.llm import FINDINGS_SCHEMA, LLMPass, MockProvider, validate
from hindsight.llm.prompts import SYSTEM, build_user_prompt, numbered_source
from hindsight.llm.provider import ProviderError, build_provider
from hindsight.merge import merge
from hindsight.models import Category, Confidence, Finding, Location, Severity, Source

SOURCE = """
import pandas as pd

def build(panel):
    panel["feature"] = panel["price"].rolling(20).mean()
    return panel
"""


def _finding_payload(**overrides):
    payload = {
        "check_id": "LA006",
        "category": "LOOKAHEAD",
        "file": "strategy.py",
        "line": 4,
        "end_line": 4,
        "title": "Feature timestamped after the bar it indexes",
        "severity": "high",
        "confidence": "medium",
        "mechanism": "The rolling mean is joined on an effective date, not an availability date.",
        "quoted_line": 'panel["feature"] = panel["price"].rolling(20).mean()',
    }
    payload.update(overrides)
    return payload


# --------------------------------------------------------------------- prompts


def test_prompt_carries_the_file_marker_and_line_numbers() -> None:
    ctx = context(SOURCE)
    prompt = build_user_prompt(ctx, Category.LOOKAHEAD)

    assert prompt.startswith("FILE: strategy.py")
    assert "LA006" in prompt
    assert "1 |" in prompt
    assert "cite them" in prompt


def test_prompt_declares_truncation_rather_than_hiding_it() -> None:
    ctx = context("\n".join(f"x{i} = {i}" for i in range(50)))
    prompt = build_user_prompt(ctx, Category.LOOKAHEAD, max_lines=10)
    body, shown = numbered_source(ctx, max_lines=10)

    assert shown == 10
    assert "you are seeing the first 10" in prompt
    assert body in prompt


def test_system_prompt_forbids_uncited_findings() -> None:
    assert "Cite a real line" in SYSTEM
    assert "empty findings list" in SYSTEM


# ------------------------------------------------------------------- provider


def test_mock_provider_returns_nothing_by_default() -> None:
    provider = MockProvider()
    raw = provider.complete(system="s", user="FILE: a.py\n", schema=FINDINGS_SCHEMA)
    assert json.loads(raw) == {"findings": []}
    assert provider.calls


def test_mock_provider_keys_on_the_file_path() -> None:
    provider = MockProvider(
        responses={"strategy.py": json.dumps({"findings": [_finding_payload()]})}
    )
    raw = provider.complete(system="s", user="FILE: strategy.py\n", schema=FINDINGS_SCHEMA)
    assert json.loads(raw)["findings"]

    other = provider.complete(system="s", user="FILE: elsewhere.py\n", schema=FINDINGS_SCHEMA)
    assert json.loads(other) == {"findings": []}


def test_build_provider_rejects_an_unknown_name() -> None:
    assert build_provider("mock").name == "mock"
    with pytest.raises(ProviderError):
        build_provider("gpt")


# ------------------------------------------------------------------ validation


def test_a_well_formed_finding_is_accepted() -> None:
    ctx = context(SOURCE)
    finding, rejection = validate(_finding_payload(), ctx, Category.LOOKAHEAD)

    assert rejection is None
    assert finding is not None
    assert finding.check_id == "LA006"
    assert finding.source is Source.LLM
    assert finding.location.path == "strategy.py"
    assert finding.location.line == 4


@pytest.mark.parametrize(
    ("overrides", "reason_fragment"),
    [
        ({"check_id": "ZZ001"}, "unknown check id"),
        ({"check_id": "CF001"}, "belongs to"),
        ({"file": "other.py"}, "not the file under audit"),
        ({"line": 9999, "end_line": 9999}, "outside the file"),
        ({"line": 1, "end_line": 1}, "quoted text does not match"),
        ({"quoted_line": "something the model invented"}, "quoted text does not match"),
        ({"mechanism": "  "}, "no mechanism"),
        ({"line": "not a number"}, "non-integer"),
    ],
)
def test_bad_citations_are_discarded(overrides: dict, reason_fragment: str) -> None:
    ctx = context(SOURCE)
    finding, rejection = validate(_finding_payload(**overrides), ctx, Category.LOOKAHEAD)

    assert finding is None
    assert rejection is not None
    assert reason_fragment in rejection.reason


def test_a_blank_cited_line_is_rejected() -> None:
    ctx = context(SOURCE)
    payload = _finding_payload(line=2, end_line=2, quoted_line="")
    finding, rejection = validate(payload, ctx, Category.LOOKAHEAD)

    assert finding is None
    assert "blank" in rejection.reason


# ------------------------------------------------------------------- the pass


def test_pass_collects_valid_findings_and_records_rejections() -> None:
    good = json.dumps({"findings": [_finding_payload()]})
    provider = MockProvider(responses={"strategy.py": good})
    pass_ = LLMPass(provider=provider, categories=(Category.LOOKAHEAD,))

    findings = pass_.run([context(SOURCE)])
    assert len(findings) == 1
    assert not pass_.rejected and not pass_.errors


def test_pass_survives_a_non_json_response() -> None:
    provider = MockProvider(responses={"strategy.py": "not json at all"})
    pass_ = LLMPass(provider=provider, categories=(Category.LOOKAHEAD,))

    assert pass_.run([context(SOURCE)]) == []
    assert any("not JSON" in error for error in pass_.errors)


def test_pass_survives_a_provider_failure() -> None:
    class Broken:
        name = "broken"

        def complete(self, *, system, user, schema):
            raise ProviderError("no credentials")

    pass_ = LLMPass(provider=Broken(), categories=(Category.LOOKAHEAD,))
    assert pass_.run([context(SOURCE)]) == []
    assert any("no credentials" in error for error in pass_.errors)


# ----------------------------------------------------------------------- merge


def _static(check_id: str, line: int, category: Category, severity=Severity.HIGH) -> Finding:
    return Finding(
        check_id=check_id,
        category=category,
        title="t",
        severity=severity,
        confidence=Confidence.MEDIUM,
        location=Location("a.py", line),
        snippet="",
        mechanism="m",
    )


def test_a_confirmed_static_finding_is_promoted() -> None:
    static = _static("LA001", 10, Category.LOOKAHEAD)
    llm = Finding(
        check_id="LA006",
        category=Category.LOOKAHEAD,
        title="t",
        severity=Severity.HIGH,
        confidence=Confidence.HIGH,
        location=Location("a.py", 11),
        snippet="",
        mechanism="m",
        source=Source.LLM,
    )

    merged = merge([static], [llm])
    assert len(merged) == 1
    assert merged[0].confidence is Confidence.HIGH
    assert merged[0].source is Source.BOTH


def test_an_uncorroborated_model_finding_is_kept_but_capped() -> None:
    llm = Finding(
        check_id="LA006",
        category=Category.LOOKAHEAD,
        title="t",
        severity=Severity.HIGH,
        confidence=Confidence.HIGH,
        location=Location("a.py", 40),
        snippet="",
        mechanism="m",
        source=Source.LLM,
    )
    merged = merge([_static("CF001", 10, Category.COST_FANTASY)], [llm])

    assert len(merged) == 2
    promoted = next(f for f in merged if f.source is Source.LLM)
    assert promoted.confidence is Confidence.MEDIUM


def test_duplicate_static_hits_collapse() -> None:
    merged = merge([_static("LA001", 10, Category.LOOKAHEAD)] * 3)
    assert len(merged) == 1


def test_findings_sort_by_severity_then_location() -> None:
    merged = merge(
        [
            _static("CF004", 90, Category.COST_FANTASY, Severity.MEDIUM),
            _static("LA001", 5, Category.LOOKAHEAD, Severity.CRITICAL),
        ]
    )
    assert [f.severity for f in merged] == [Severity.CRITICAL, Severity.MEDIUM]


# ------------------------------------------------------------------ end to end


def test_audit_with_a_mock_provider_runs_offline(fixtures_dir) -> None:
    result = audit(fixtures_dir / "clean", provider=MockProvider())
    assert result.llm_enabled
    assert result.findings == []
