"""Provider abstraction for the model pass.

Everything above this file is provider-agnostic, which is what lets the static
layer run with no network, no credentials and no third-party import. The default
provider is :class:`MockProvider`; the live one is constructed only when the CLI
is given ``--llm``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_MAX_TOKENS = 8000


class ProviderError(RuntimeError):
    """Raised when a provider cannot produce a response."""


@runtime_checkable
class Provider(Protocol):
    """Anything that can answer a structured-output request."""

    name: str

    def complete(self, *, system: str, user: str, schema: dict[str, Any]) -> str:
        """Return the model's response as a JSON string conforming to `schema`."""
        ...


@dataclass
class MockProvider:
    """Deterministic stand-in used by the tests and by ``--llm --provider mock``.

    `responses` maps a key - by default the repo-relative file path - to the JSON
    string the provider should return for it. Anything unmapped yields an empty
    finding list, which is the behaviour that keeps the eval harness honest: a
    mocked model pass must not be able to inflate recall.
    """

    name: str = "mock"
    responses: dict[str, str] = field(default_factory=dict)
    key_for: Callable[[str, str], str] | None = None
    calls: list[tuple[str, str]] = field(default_factory=list)

    def complete(self, *, system: str, user: str, schema: dict[str, Any]) -> str:
        self.calls.append((system, user))
        key = self.key_for(system, user) if self.key_for else _first_path_in(user)
        return self.responses.get(key, json.dumps({"findings": []}))


def _first_path_in(user: str) -> str:
    """Extract the ``FILE: <path>`` marker the prompt builder emits."""
    for line in user.splitlines():
        if line.startswith("FILE: "):
            return line[len("FILE: ") :].strip()
    return ""


@dataclass
class AnthropicProvider:
    """Live provider backed by the Claude API.

    The ``anthropic`` package is imported lazily so that installing Hindsight
    without the ``llm`` extra still works, and so that a static-only audit never
    pays for the import.
    """

    name: str = "anthropic"
    model: str = DEFAULT_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    effort: str = "high"
    api_key: str | None = None
    _client: Any = field(default=None, init=False, repr=False)

    def client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise ProviderError(
                "the anthropic package is required for --llm; "
                "install it with `pip install 'hindsight[llm]'`"
            ) from exc
        key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        # A bare client also resolves an `ant auth login` profile, so an unset
        # ANTHROPIC_API_KEY is not by itself an error.
        self._client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
        return self._client

    def complete(self, *, system: str, user: str, schema: dict[str, Any]) -> str:
        client = self.client()
        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": schema},
            },
        )
        if getattr(response, "stop_reason", None) == "refusal":
            raise ProviderError(
                "the model declined this request; the audited file may contain content that "
                "trips a safety classifier. Re-run with --no-llm to audit it statically."
            )
        for block in response.content:
            if getattr(block, "type", None) == "text":
                return block.text
        raise ProviderError("model returned no text block")


def build_provider(name: str) -> Provider:
    if name == "mock":
        return MockProvider()
    if name == "anthropic":
        return AnthropicProvider()
    raise ProviderError(f"unknown provider {name!r}; expected 'anthropic' or 'mock'")
