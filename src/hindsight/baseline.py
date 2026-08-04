"""Suppression of already-known findings.

Pointing an auditor at an established repository produces a wall of findings
nobody is going to work through, and a wall of findings is indistinguishable
from no findings: both get ignored. A baseline records what was already there so
the next run reports only what is new.

Entries are keyed on a fingerprint of the check, the file and the *text* of the
cited code - never the line number. A line-numbered baseline goes stale the
first time somebody adds an import at the top of the file, and a stale
suppression file is worse than none, because it silently stops suppressing the
thing it named and starts suppressing whatever moved into that line.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import Finding

FORMAT_VERSION = 1


def fingerprint(finding: Finding) -> str:
    """Stable identity of a finding across edits elsewhere in the file."""
    snippet = " ".join(finding.snippet.split())
    payload = "\x00".join(
        [
            finding.check_id,
            finding.location.path,
            str(finding.location.cell or ""),
            snippet,
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class Baseline:
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)
    created_at: str = ""

    @classmethod
    def from_findings(cls, findings: list[Finding], *, created_at: str = "") -> Baseline:
        return cls(
            created_at=created_at,
            entries={
                fingerprint(f): {
                    "check_id": f.check_id,
                    "file": f.location.path,
                    "line": f.location.span[0],
                    "title": f.title,
                }
                for f in findings
            },
        )

    @classmethod
    def load(cls, path: Path) -> Baseline:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        version = payload.get("format_version", FORMAT_VERSION)
        if version != FORMAT_VERSION:
            raise ValueError(
                f"{path} is a version {version} baseline; this build writes version "
                f"{FORMAT_VERSION}. Regenerate it with --write-baseline."
            )
        return cls(entries=payload.get("entries", {}), created_at=payload.get("created_at", ""))

    def dump(self, path: Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "format_version": FORMAT_VERSION,
                    "created_at": self.created_at,
                    "note": (
                        "Findings suppressed as pre-existing. Keyed on the text of the cited "
                        "code, not its line number, so edits elsewhere in the file do not stale "
                        "the entry. Delete an entry to start reporting that finding again."
                    ),
                    "entries": self.entries,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def __len__(self) -> int:
        return len(self.entries)

    def filter(self, findings: list[Finding]) -> tuple[list[Finding], int]:
        """Drop findings the baseline already knows about."""
        kept = [f for f in findings if fingerprint(f) not in self.entries]
        return kept, len(findings) - len(kept)
