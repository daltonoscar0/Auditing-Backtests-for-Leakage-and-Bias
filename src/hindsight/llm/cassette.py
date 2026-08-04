"""Record and replay model responses.

The model pass is non-deterministic, costs money and needs network access. None
of those belong in a test suite or in ``make eval``. A cassette is a JSON file of
recorded responses keyed by ``<path>::<CATEGORY>``, so a run can be captured once
against the live API and replayed forever afterwards.

Provenance is recorded in the file and surfaced by the eval harness, because a
hand-authored cassette and a live-recorded one support very different claims. A
hand-authored one exercises the pipeline - prompt, schema, validation, merge,
render - and proves the fixture is statically clean. Only a live one says
anything about whether the model can actually find the defect.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .prompts import prompt_key
from .provider import Provider, ProviderError, Usage

HAND_AUTHORED = "hand-authored"
LIVE = "live"


@dataclass
class Cassette:
    """The on-disk format."""

    provenance: str = HAND_AUTHORED
    model: str = ""
    recorded_at: str = ""
    note: str = ""
    responses: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> Cassette:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            provenance=payload.get("provenance", HAND_AUTHORED),
            model=payload.get("model", ""),
            recorded_at=payload.get("recorded_at", ""),
            note=payload.get("note", ""),
            responses=payload.get("responses", {}),
        )

    def dump(self, path: Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "provenance": self.provenance,
                    "model": self.model,
                    "recorded_at": self.recorded_at,
                    "note": self.note,
                    "responses": self.responses,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    @property
    def is_live(self) -> bool:
        return self.provenance == LIVE


@dataclass
class CassetteProvider:
    """Replays a cassette. Never touches the network."""

    cassette: Cassette
    name: str = "cassette"
    strict: bool = True
    """When strict, a prompt with no recorded response is an error rather than a
    silent empty result. A cassette that has drifted out of sync with the prompt
    builder should fail loudly, not quietly score zero."""
    usage: Usage = field(default_factory=Usage)
    misses: list[str] = field(default_factory=list)

    @classmethod
    def from_path(cls, path: Path, *, strict: bool = True) -> CassetteProvider:
        return cls(cassette=Cassette.load(Path(path)), strict=strict)

    def complete(self, *, system: str, user: str, schema: dict[str, Any]) -> str:
        key = prompt_key(user)
        self.usage.requests += 1
        if key in self.cassette.responses:
            return json.dumps(self.cassette.responses[key])
        self.misses.append(key)
        if self.strict:
            raise ProviderError(
                f"no recorded response for {key!r}. Re-record the cassette with "
                f"`hindsight audit <path> --llm --llm-record <file>`."
            )
        return json.dumps({"findings": []})


@dataclass
class RecordingProvider:
    """Wraps a live provider and writes everything it returns to a cassette."""

    inner: Provider
    cassette: Cassette
    name: str = "recording"

    @property
    def usage(self) -> Usage:
        return getattr(self.inner, "usage", Usage())

    def complete(self, *, system: str, user: str, schema: dict[str, Any]) -> str:
        raw = self.inner.complete(system=system, user=user, schema=schema)
        try:
            self.cassette.responses[prompt_key(user)] = json.loads(raw)
        except json.JSONDecodeError:
            # Keep the unparseable text so the failure is reproducible offline.
            self.cassette.responses[prompt_key(user)] = {"__raw__": raw}
        return raw

    def save(self, path: Path) -> None:
        self.cassette.dump(Path(path))
