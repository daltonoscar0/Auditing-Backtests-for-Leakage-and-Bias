"""Core value types shared by the static passes, the model passes and the renderers."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field, replace
from typing import Any


class Category(enum.StrEnum):
    """Top-level finding taxonomy. See TAXONOMY.md for the normative definitions."""

    LOOKAHEAD = "LOOKAHEAD"
    LABEL_LEAK = "LABEL-LEAK"
    SURVIVORSHIP = "SURVIVORSHIP"
    UNPURGED_CV = "UNPURGED-CV"
    COST_FANTASY = "COST-FANTASY"
    SWEEP_OVERFIT = "SWEEP-OVERFIT"


class Severity(enum.StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

    @property
    def rank(self) -> int:
        return _SEVERITY_RANK[self]

    def __ge__(self, other: object) -> bool:  # type: ignore[override]
        if not isinstance(other, Severity):
            return NotImplemented
        return self.rank >= other.rank


_SEVERITY_RANK: dict[Severity, int] = {
    Severity.INFO: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}


class Confidence(enum.StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @property
    def rank(self) -> int:
        return {Confidence.LOW: 0, Confidence.MEDIUM: 1, Confidence.HIGH: 2}[self]


class Source(enum.StrEnum):
    STATIC = "static"
    LLM = "llm"
    BOTH = "static+llm"


@dataclass(frozen=True, slots=True)
class Location:
    """A citation into the audited repository. `path` is repo-relative, POSIX-style."""

    path: str
    line: int
    end_line: int | None = None

    def __post_init__(self) -> None:
        if self.line < 1:
            raise ValueError(f"line numbers are 1-based, got {self.line}")

    @property
    def span(self) -> tuple[int, int]:
        return (self.line, self.end_line if self.end_line is not None else self.line)

    def overlaps(self, other: Location, tolerance: int = 0) -> bool:
        if self.path != other.path:
            return False
        a_start, a_end = self.span
        b_start, b_end = other.span
        return a_start - tolerance <= b_end and b_start - tolerance <= a_end

    def __str__(self) -> str:
        start, end = self.span
        return f"{self.path}:{start}" if start == end else f"{self.path}:{start}-{end}"


@dataclass(frozen=True, slots=True)
class Finding:
    """One audit result.

    `check_id` identifies the detection rule (e.g. ``LA001``); `category` is the
    taxonomy bucket it rolls up to. Everything a reader needs to accept or reject
    the finding without re-deriving it lives on this object.
    """

    check_id: str
    category: Category
    title: str
    severity: Severity
    confidence: Confidence
    location: Location
    snippet: str
    mechanism: str
    fix: str | None = None
    source: Source = Source.STATIC
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        start, end = self.location.span
        return {
            "check_id": self.check_id,
            "category": self.category.value,
            "title": self.title,
            "severity": self.severity.value,
            "confidence": self.confidence.value,
            "file": self.location.path,
            "line": start,
            "end_line": end,
            "snippet": self.snippet,
            "mechanism": self.mechanism,
            "fix": self.fix,
            "source": self.source.value,
            "evidence": self.evidence,
        }

    def with_confidence(self, confidence: Confidence, source: Source) -> Finding:
        return replace(self, confidence=confidence, source=source)

    @property
    def sort_key(self) -> tuple[int, str, int, str]:
        start, _ = self.location.span
        return (-self.severity.rank, self.location.path, start, self.check_id)


@dataclass(frozen=True, slots=True)
class AuditResult:
    root: str
    findings: list[Finding]
    files_scanned: int
    checks_run: list[str]
    llm_enabled: bool = False
    errors: list[str] = field(default_factory=list)
    llm_usage: dict[str, Any] | None = None
    """Requests and tokens consumed by the model pass, when one ran."""

    def counts_by_severity(self) -> dict[str, int]:
        counts = {s.value: 0 for s in Severity}
        for finding in self.findings:
            counts[finding.severity.value] += 1
        return counts

    def counts_by_category(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for finding in self.findings:
            counts[finding.category.value] = counts.get(finding.category.value, 0) + 1
        return counts

    def max_severity(self) -> Severity | None:
        if not self.findings:
            return None
        return max((f.severity for f in self.findings), key=lambda s: s.rank)
