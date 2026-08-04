"""JSON report renderer - the CI-facing output.

The shape is pinned by the taxonomy version so that a consumer can detect a
breaking change rather than silently misread a renamed field.
"""

from __future__ import annotations

import json
from typing import Any

from ..models import AuditResult
from ..taxonomy import TAXONOMY_VERSION


def to_dict(result: AuditResult) -> dict[str, Any]:
    return {
        "tool": "hindsight",
        "taxonomy_version": TAXONOMY_VERSION,
        "root": result.root,
        "files_scanned": result.files_scanned,
        "checks_run": result.checks_run,
        "llm_enabled": result.llm_enabled,
        "summary": {
            "total": len(result.findings),
            "by_severity": result.counts_by_severity(),
            "by_category": result.counts_by_category(),
            "max_severity": result.max_severity().value if result.max_severity() else None,
        },
        "findings": [finding.to_dict() for finding in result.findings],
        "errors": result.errors,
    }


def render(result: AuditResult, *, indent: int = 2) -> str:
    return json.dumps(to_dict(result), indent=indent, sort_keys=False)
