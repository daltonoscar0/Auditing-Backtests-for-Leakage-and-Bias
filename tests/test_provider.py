"""The live provider's request shape, cassettes, the relevance gate and cost accounting.

The Anthropic provider is exercised against a fake client rather than the API.
That verifies everything the wire format depends on - model id, structured
output config, cached system prompt, refusal handling, usage accounting - without
needing credentials, which is what makes it runnable in CI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from conftest import context

from hindsight import audit
from hindsight.llm.cassette import (
    HAND_AUTHORED,
    LIVE,
    Cassette,
    CassetteProvider,
    RecordingProvider,
)
from hindsight.llm.prompts import prompt_key
from hindsight.llm.provider import (
    DEFAULT_MODEL,
    AnthropicProvider,
    MockProvider,
    ProviderError,
    Usage,
)
from hindsight.llm.reasoning import LLMPass, relevant_categories
from hindsight.llm.schema import FINDINGS_SCHEMA
from hindsight.models import Category

SOURCE = """
import pandas as pd

def build(panel):
    panel["feature"] = panel["price"].rolling(20).mean()
    return panel
"""


# --------------------------------------------------------------- fake client


@dataclass
class FakeUsage:
    input_tokens: int = 1200
    output_tokens: int = 300
    cache_read_input_tokens: int = 1000


@dataclass
class FakeBlock:
    type: str
    text: str


@dataclass
class FakeResponse:
    content: list[FakeBlock]
    stop_reason: str = "end_turn"
    stop_details: Any = None
    usage: FakeUsage = field(default_factory=FakeUsage)


@dataclass
class FakeMessages:
    payload: str = '{"findings": []}'
    stop_reason: str = "end_turn"
    stop_details: Any = None
    captured: dict = field(default_factory=dict)

    def create(self, **kwargs: Any) -> FakeResponse:
        self.captured = kwargs
        return FakeResponse(
            content=[FakeBlock("text", self.payload)],
            stop_reason=self.stop_reason,
            stop_details=self.stop_details,
        )


@dataclass
class FakeClient:
    messages: FakeMessages = field(default_factory=FakeMessages)


def _provider(**kwargs: Any) -> tuple[AnthropicProvider, FakeClient]:
    provider = AnthropicProvider(**kwargs)
    client = FakeClient()
    provider._client = client
    return provider, client


# ------------------------------------------------------------ request shape


def test_request_uses_the_documented_model_and_structured_output() -> None:
    provider, client = _provider()
    provider.complete(system="SYS", user="FILE: a.py\n", schema=FINDINGS_SCHEMA)
    sent = client.messages.captured

    assert sent["model"] == DEFAULT_MODEL == "claude-opus-5"
    assert sent["max_tokens"] > 0
    assert sent["messages"] == [{"role": "user", "content": "FILE: a.py\n"}]
    assert sent["output_config"]["effort"] == "high"
    assert sent["output_config"]["format"] == {
        "type": "json_schema",
        "schema": FINDINGS_SCHEMA,
    }
    # Sampling parameters and budget_tokens are rejected on this model.
    for removed in ("temperature", "top_p", "top_k", "thinking"):
        assert removed not in sent


def test_the_system_prompt_is_cached() -> None:
    """It is byte-identical across every request in a run and it is the longest
    fixed prefix, so it is the one thing worth a cache breakpoint."""
    provider, client = _provider()
    provider.complete(system="SYS", user="u", schema=FINDINGS_SCHEMA)

    system = client.messages.captured["system"]
    assert isinstance(system, list)
    assert system[0]["text"] == "SYS"
    assert system[0]["cache_control"] == {"type": "ephemeral"}


def test_usage_is_accumulated_across_requests() -> None:
    provider, _ = _provider()
    for _ in range(3):
        provider.complete(system="SYS", user="u", schema=FINDINGS_SCHEMA)

    assert provider.usage.requests == 3
    assert provider.usage.input_tokens == 3600
    assert provider.usage.output_tokens == 900
    assert provider.usage.cached_input_tokens == 3000
    assert provider.usage.estimated_usd == pytest.approx(3600 * 5 / 1e6 + 900 * 25 / 1e6)


def test_a_refusal_is_raised_before_content_is_read() -> None:
    provider, client = _provider()
    client.messages.stop_reason = "refusal"
    client.messages.stop_details = type("D", (), {"category": "cyber"})()

    with pytest.raises(ProviderError) as excinfo:
        provider.complete(system="SYS", user="u", schema=FINDINGS_SCHEMA)

    assert "declined" in str(excinfo.value)
    assert "cyber" in str(excinfo.value)
    assert "--no-llm" in str(excinfo.value)


def test_a_response_with_no_text_block_is_an_error() -> None:
    provider, client = _provider()
    client.messages.create = lambda **kw: FakeResponse(content=[FakeBlock("thinking", "")])

    with pytest.raises(ProviderError):
        provider.complete(system="SYS", user="u", schema=FINDINGS_SCHEMA)


def test_usage_totals_serialise() -> None:
    usage = Usage(requests=2, input_tokens=1_000_000, output_tokens=100_000)
    assert usage.to_dict()["estimated_usd"] == pytest.approx(5.0 + 2.5)


# ----------------------------------------------------------------- cassettes


def test_prompt_key_identifies_file_and_category() -> None:
    ctx = context(SOURCE)
    from hindsight.llm.prompts import build_user_prompt

    key = prompt_key(build_user_prompt(ctx, Category.COST_FANTASY))
    assert key == "strategy.py::COST-FANTASY"


def test_record_then_replay_round_trips(tmp_path: Path) -> None:
    payload = json.dumps({"findings": []})
    inner = MockProvider(responses={"strategy.py::LOOKAHEAD": payload})
    recorder = RecordingProvider(inner=inner, cassette=Cassette(provenance=LIVE, model="m"))

    pass_ = LLMPass(provider=recorder, categories=(Category.LOOKAHEAD,), workers=1)
    pass_.run([context(SOURCE)])

    destination = tmp_path / "cassette.json"
    recorder.save(destination)

    replayed = CassetteProvider.from_path(destination)
    assert replayed.cassette.is_live
    assert "strategy.py::LOOKAHEAD" in replayed.cassette.responses

    second = LLMPass(provider=replayed, categories=(Category.LOOKAHEAD,), workers=1)
    assert second.run([context(SOURCE)]) == []
    assert not second.errors


def test_a_cassette_miss_is_loud_by_default(tmp_path: Path) -> None:
    """A cassette that has drifted out of sync must fail, not silently score zero."""
    destination = tmp_path / "cassette.json"
    Cassette(responses={"other.py::LOOKAHEAD": {"findings": []}}).dump(destination)

    pass_ = LLMPass(
        provider=CassetteProvider.from_path(destination),
        categories=(Category.LOOKAHEAD,),
        workers=1,
    )
    pass_.run([context(SOURCE)])
    assert any("no recorded response" in error for error in pass_.errors)


def test_a_lenient_cassette_returns_nothing_on_a_miss(tmp_path: Path) -> None:
    destination = tmp_path / "cassette.json"
    Cassette(responses={}).dump(destination)
    provider = CassetteProvider.from_path(destination, strict=False)

    assert json.loads(provider.complete(system="s", user="FILE: a.py\n", schema={})) == {
        "findings": []
    }
    assert provider.misses == ["a.py"]


def test_provenance_survives_a_round_trip(tmp_path: Path) -> None:
    destination = tmp_path / "cassette.json"
    Cassette(provenance=HAND_AUTHORED, note="n").dump(destination)
    assert not Cassette.load(destination).is_live


# ------------------------------------------------------------ relevance gate


def test_the_gate_skips_categories_with_no_foothold() -> None:
    ctx = context(
        """
        def add(a, b):
            return a + b
        """
    )
    assert relevant_categories(ctx, list(Category)) == []


def test_the_gate_keeps_categories_the_file_could_trip() -> None:
    ctx = context(SOURCE)
    kept = relevant_categories(ctx, list(Category))
    assert Category.LOOKAHEAD in kept


def test_the_gate_never_drops_a_category_a_fixture_expects(fixtures_dir) -> None:
    """The gate is a cost lever, and a skipped category is a guaranteed miss.

    This caught the gate silently skipping SWEEP-OVERFIT on a selection log whose
    defect is precisely that it does not look like a search.
    """
    from hindsight.collector import collect

    for directory in sorted(p for p in fixtures_dir.iterdir() if p.is_dir()):
        expectations = json.loads(
            (directory / "expected_findings.json").read_text(encoding="utf-8")
        )["expected"]
        if not expectations:
            continue

        contexts, _ = collect(directory)
        by_path = {c.path: c for c in contexts}
        for expectation in expectations:
            ctx = by_path[expectation["file"]]
            category = Category(expectation["category"])
            assert category in relevant_categories(ctx, list(Category)), (
                f"the relevance gate would skip {category.value} on "
                f"{directory.name}/{expectation['file']}, where a bug is planted"
            )


def test_the_gate_can_be_turned_off() -> None:
    ctx = context("def add(a, b):\n    return a + b\n")
    pass_ = LLMPass(provider=MockProvider(), gate=False, workers=1)
    assert len(pass_.plan([ctx])) == len(Category)
    assert pass_.skipped == []


# ------------------------------------------------------------- orchestration


def test_audit_reports_model_pass_usage(fixtures_dir) -> None:
    result = audit(fixtures_dir / "mixed", provider=MockProvider(), llm_workers=2)
    assert result.llm_usage is not None
    assert result.llm_usage["requests"] > 0
    assert result.llm_usage["skipped_requests"] > 0


def test_parallel_and_serial_runs_agree(fixtures_dir) -> None:
    serial = audit(fixtures_dir / "llm_only", provider=_cassette(fixtures_dir), llm_workers=1)
    parallel = audit(fixtures_dir / "llm_only", provider=_cassette(fixtures_dir), llm_workers=4)

    assert [f.to_dict() for f in serial.findings] == [f.to_dict() for f in parallel.findings]


def _cassette(fixtures_dir):
    return CassetteProvider.from_path(fixtures_dir / "llm_only" / "cassette.json")


def test_the_llm_only_fixture_is_invisible_to_the_static_layer(fixtures_dir) -> None:
    """If the static checks can see these defects, the fixture is not testing Layer 2."""
    assert audit(fixtures_dir / "llm_only").findings == []


def test_the_llm_only_fixture_yields_every_planted_finding(fixtures_dir) -> None:
    result = audit(fixtures_dir / "llm_only", provider=_cassette(fixtures_dir), llm_workers=1)
    assert {f.check_id for f in result.findings} == {
        "LA006",
        "LL004",
        "SV004",
        "CV005",
        "CF005",
        "SO005",
    }
    assert all(f.source.value == "llm" for f in result.findings)
    assert all(f.confidence.value in {"medium", "low"} for f in result.findings), (
        "an uncorroborated model finding must not claim high confidence"
    )
