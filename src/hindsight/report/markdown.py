"""Markdown report renderer - the human-facing output."""

from __future__ import annotations

from ..models import AuditResult, Category, Finding, Severity
from ..taxonomy import TAXONOMY_VERSION, spec

SEVERITY_MARK = {
    Severity.CRITICAL: "critical",
    Severity.HIGH: "high",
    Severity.MEDIUM: "medium",
    Severity.LOW: "low",
    Severity.INFO: "info",
}

CATEGORY_ORDER = list(Category)


def render(result: AuditResult, *, show_fixes: bool = True) -> str:
    lines: list[str] = []
    lines.append("# Hindsight audit")
    lines.append("")
    lines.append(f"**Target:** `{result.root}`")
    lines.append(
        f"**Scanned:** {result.files_scanned} Python file"
        f"{'' if result.files_scanned == 1 else 's'} · "
        f"{len(result.checks_run)} checks · taxonomy {TAXONOMY_VERSION} · "
        f"model pass {'on' if result.llm_enabled else 'off'}"
    )
    if result.llm_usage and result.llm_usage.get("requests"):
        usage = result.llm_usage
        lines.append(
            f"**Model pass:** {usage['requests']} request(s), "
            f"{usage['skipped_requests']} skipped as irrelevant, "
            f"{usage['input_tokens']:,} in / {usage['output_tokens']:,} out "
            f"(~${usage['estimated_usd']:.3f})"
        )
    lines.append("")

    if not result.findings:
        lines.append("## No findings")
        lines.append("")
        lines.append(
            "None of the patterns in the taxonomy were found. This is not a statement that "
            "the research is correct - see the limits section of `TAXONOMY.md` for what is "
            "outside the tool's scope."
        )
        lines.extend(_errors_section(result))
        return "\n".join(lines) + "\n"

    lines.extend(_summary(result))
    lines.extend(_findings(result, show_fixes=show_fixes))
    lines.extend(_errors_section(result))
    return "\n".join(lines) + "\n"


def _summary(result: AuditResult) -> list[str]:
    severity_counts = result.counts_by_severity()
    category_counts = result.counts_by_category()

    lines = ["## Summary", ""]
    total = len(result.findings)
    lines.append(f"**{total} finding{'' if total == 1 else 's'}.**")
    lines.append("")

    lines.append("| Severity | Count |")
    lines.append("| --- | ---: |")
    for severity in (
        Severity.CRITICAL,
        Severity.HIGH,
        Severity.MEDIUM,
        Severity.LOW,
        Severity.INFO,
    ):
        count = severity_counts.get(severity.value, 0)
        if count:
            lines.append(f"| {severity.value} | {count} |")
    lines.append("")

    lines.append("| Category | Count |")
    lines.append("| --- | ---: |")
    for category in CATEGORY_ORDER:
        count = category_counts.get(category.value, 0)
        if count:
            lines.append(f"| {category.value} | {count} |")
    lines.append("")
    return lines


def _findings(result: AuditResult, *, show_fixes: bool) -> list[str]:
    lines = ["## Findings", ""]
    for index, finding in enumerate(result.findings, start=1):
        lines.extend(_one(index, finding, show_fixes=show_fixes))
    return lines


def _one(index: int, finding: Finding, *, show_fixes: bool) -> list[str]:
    lines = [
        f"### {index}. {finding.check_id} — {finding.title}",
        "",
        f"**{SEVERITY_MARK[finding.severity]}** · {finding.category.value} · "
        f"confidence {finding.confidence.value} · detected by {finding.source.value}",
        "",
        f"`{finding.location}`",
        "",
    ]
    if finding.snippet:
        lines.extend(["```python", finding.snippet, "```", ""])

    lines.append("**Why this matters**")
    lines.append("")
    for paragraph in finding.mechanism.split("\n\n"):
        lines.append(paragraph.strip())
        lines.append("")

    remedy = spec(finding.check_id).fix
    lines.append("**Fix**")
    lines.append("")
    lines.append(remedy)
    lines.append("")

    if show_fixes and finding.fix:
        lines.append("<details><summary>Proposed patch</summary>")
        lines.append("")
        lines.append("```diff")
        lines.append(finding.fix)
        lines.append("```")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    return lines


def _errors_section(result: AuditResult) -> list[str]:
    if not result.errors:
        return []
    lines = ["## Notes", ""]
    lines.append("Files or model responses that were skipped during this audit:")
    lines.append("")
    lines.extend(f"- {error}" for error in result.errors)
    lines.append("")
    return lines
