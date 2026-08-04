"""Deduplication and confidence scoring across the static and model passes.

Two passes looking at the same code will often describe the same defect. Reporting
it twice is worse than reporting it once, because a reader who dismisses a
duplicate learns to skim. The rules here are deliberately simple and are the only
place confidence is allowed to change after a check emits.
"""

from __future__ import annotations

from .models import Confidence, Finding, Source

OVERLAP_TOLERANCE = 2
"""Lines of slack when deciding whether two findings point at the same code. The
model pass cites the line it considers most relevant, which is not always the line
the AST node starts on."""


def _same_defect(a: Finding, b: Finding, tolerance: int) -> bool:
    if a.category is not b.category:
        return False
    return a.location.overlaps(b.location, tolerance=tolerance)


def dedupe(findings: list[Finding]) -> list[Finding]:
    """Collapse identical check hits at overlapping spans, keeping the widest citation."""
    kept: list[Finding] = []
    for finding in findings:
        for index, existing in enumerate(kept):
            if existing.check_id != finding.check_id:
                continue
            if not existing.location.overlaps(finding.location):
                continue
            if _width(finding) > _width(existing):
                kept[index] = finding
            break
        else:
            kept.append(finding)
    return kept


def _width(finding: Finding) -> int:
    start, end = finding.location.span
    return end - start


def merge(
    static: list[Finding],
    llm: list[Finding] | None = None,
    *,
    tolerance: int = OVERLAP_TOLERANCE,
) -> list[Finding]:
    """Combine both layers into the reported set.

    A static hit corroborated by the model pass is promoted to high confidence and
    marked ``static+llm``. A model finding with no static counterpart is kept, but
    capped at medium confidence: one pass asserting something is weaker evidence
    than two, and the report should say so.
    """
    result = dedupe(static)
    if not llm:
        return sorted(result, key=lambda f: f.sort_key)

    for candidate in dedupe(llm):
        absorbed = False
        for index, existing in enumerate(result):
            if not _same_defect(existing, candidate, tolerance):
                continue
            if existing.source is Source.STATIC:
                # Corroboration: the two passes independently found the same thing.
                result[index] = existing.with_confidence(Confidence.HIGH, Source.BOTH)
                absorbed = True
                break
            # Category-level matching is right for static/model corroboration and
            # wrong between two model findings: a category is a bucket, not a
            # defect. A liquidity finding and a fill-price finding on the same
            # statement are two findings, and collapsing them silently loses
            # whichever the model happened to report second.
            if existing.check_id == candidate.check_id:
                absorbed = True
                break
        if absorbed:
            continue
        result.append(
            candidate
            if candidate.confidence.rank <= Confidence.MEDIUM.rank
            else candidate.with_confidence(Confidence.MEDIUM, Source.LLM)
        )

    return sorted(result, key=lambda f: f.sort_key)
