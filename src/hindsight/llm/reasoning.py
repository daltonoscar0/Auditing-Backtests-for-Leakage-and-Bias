"""The Layer 2 model pass: structured findings, validated against the source.

The validation in :func:`validate` is the load-bearing part. A model asked to
audit code will occasionally produce a finding that reads well and points at
nothing. Every such finding is dropped here rather than shown to a reader, on
the principle that an audit tool's credibility is spent one false positive at a
time.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from ..context import FileContext
from ..models import Category, Confidence, Finding, Location, Severity, Source
from ..taxonomy import SPECS
from .prompts import SYSTEM, build_user_prompt
from .provider import Provider, ProviderError
from .schema import FINDINGS_SCHEMA


@dataclass
class RejectedFinding:
    """A model finding that failed validation, kept for diagnostics."""

    reason: str
    payload: dict


@dataclass
class LLMPass:
    """Runs one structured request per (file, category) and validates the results."""

    provider: Provider
    categories: Sequence[Category] = tuple(Category)
    max_lines: int = 400
    rejected: list[RejectedFinding] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def run(self, contexts: Iterable[FileContext]) -> list[Finding]:
        findings: list[Finding] = []
        for ctx in contexts:
            for category in self.categories:
                findings.extend(self.run_one(ctx, category))
        return findings

    def run_one(self, ctx: FileContext, category: Category) -> list[Finding]:
        prompt = build_user_prompt(ctx, category, max_lines=self.max_lines)
        try:
            raw = self.provider.complete(system=SYSTEM, user=prompt, schema=FINDINGS_SCHEMA)
        except ProviderError as exc:
            self.errors.append(f"{ctx.path} [{category.value}]: {exc}")
            return []

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            self.errors.append(f"{ctx.path} [{category.value}]: response was not JSON ({exc})")
            return []

        results: list[Finding] = []
        for item in payload.get("findings", []):
            finding, rejection = validate(item, ctx, category)
            if finding is not None:
                results.append(finding)
            elif rejection is not None:
                self.rejected.append(rejection)
        return results


def validate(
    item: dict, ctx: FileContext, category: Category
) -> tuple[Finding | None, RejectedFinding | None]:
    """Turn one model finding into a :class:`Finding`, or reject it with a reason."""
    check_id = str(item.get("check_id", "")).strip().upper()
    if check_id not in SPECS:
        return None, RejectedFinding(f"unknown check id {check_id!r}", item)
    spec = SPECS[check_id]

    if spec.category is not category:
        return None, RejectedFinding(
            f"{check_id} belongs to {spec.category.value}, not the requested {category.value}",
            item,
        )
    # A model finding against a purely-static check id is not rejected here. It may
    # be a genuine instance the AST pass could not resolve, and if it duplicates a
    # static hit the merge layer folds the two together rather than reporting both.

    path = str(item.get("file", "")).strip()
    if path != ctx.path:
        return None, RejectedFinding(f"cited {path!r}, which is not the file under audit", item)

    try:
        line = int(item["line"])
        end_line = int(item.get("end_line", line))
    except (KeyError, TypeError, ValueError):
        return None, RejectedFinding("line numbers missing or non-integer", item)

    if not 1 <= line <= len(ctx.lines):
        return None, RejectedFinding(f"line {line} is outside the file", item)
    end_line = max(line, min(end_line, len(ctx.lines)))

    quoted = str(item.get("quoted_line", ""))
    actual = ctx.line_text(line)
    if quoted.strip() and quoted.strip() != actual.strip():
        return None, RejectedFinding(
            f"quoted text does not match line {line}: quoted {quoted.strip()!r}, "
            f"file has {actual.strip()!r}",
            item,
        )
    if not actual.strip():
        return None, RejectedFinding(f"line {line} is blank", item)

    mechanism = str(item.get("mechanism", "")).strip()
    if not mechanism:
        return None, RejectedFinding("no mechanism given", item)

    return (
        Finding(
            check_id=check_id,
            category=spec.category,
            title=str(item.get("title") or spec.name),
            severity=_enum(Severity, item.get("severity"), spec.severity),
            confidence=_enum(Confidence, item.get("confidence"), Confidence.MEDIUM),
            location=Location(ctx.path, line, end_line),
            snippet="\n".join(ctx.lines[line - 1 : min(end_line, line + 2)]).strip(),
            mechanism=mechanism,
            fix=None,
            source=Source.LLM,
            evidence={"reported_by": "llm"},
        ),
        None,
    )


def _enum(enum_cls, value, default):  # type: ignore[no-untyped-def]
    try:
        return enum_cls(value)
    except ValueError:
        return default
