"""Hindsight - audit trading-research repositories for leakage and bias.

Public surface:

    from hindsight import audit
    result = audit(Path("./some-strategy-repo"))
    for finding in result.findings:
        print(finding.check_id, finding.location, finding.title)

The audited tree is only ever read. See TAXONOMY.md for what is detected and,
just as importantly, what is not.
"""

from __future__ import annotations

from .audit import audit
from .models import (
    AuditResult,
    Category,
    Confidence,
    Finding,
    Location,
    Severity,
    Source,
)
from .taxonomy import TAXONOMY_VERSION, CheckSpec, spec

__version__ = "0.1.0"

__all__ = [
    "TAXONOMY_VERSION",
    "AuditResult",
    "Category",
    "CheckSpec",
    "Confidence",
    "Finding",
    "Location",
    "Severity",
    "Source",
    "__version__",
    "audit",
    "spec",
]
