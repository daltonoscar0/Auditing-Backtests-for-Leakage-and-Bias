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

from .prompts import prompt_key

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_MAX_TOKENS = 8000

# Published rates for claude-opus-5, US dollars per million tokens. Only used to
# print an estimate; the authoritative number is on the invoice.
INPUT_USD_PER_MTOK = 5.00
OUTPUT_USD_PER_MTOK = 25.00


class ProviderError(RuntimeError):
    """Raised when a provider cannot produce a response."""


@dataclass
class Usage:
    """What a run of the model pass consumed."""

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0

    def add(self, usage: Any) -> None:
        self.requests += 1
        self.input_tokens += getattr(usage, "input_tokens", 0) or 0
        self.output_tokens += getattr(usage, "output_tokens", 0) or 0
        self.cached_input_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0

    @property
    def estimated_usd(self) -> float:
        return (
            self.input_tokens * INPUT_USD_PER_MTOK + self.output_tokens * OUTPUT_USD_PER_MTOK
        ) / 1_000_000

    def to_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "estimated_usd": round(self.estimated_usd, 4),
        }


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
    usage: Usage = field(default_factory=Usage)

    def complete(self, *, system: str, user: str, schema: dict[str, Any]) -> str:
        self.calls.append((system, user))
        self.usage.requests += 1
        if self.key_for is not None:
            return self.responses.get(self.key_for(system, user), _EMPTY)
        key = prompt_key(user)
        if key in self.responses:
            return self.responses[key]
        # Fall back to a bare path key, so a caller that does not care which
        # category produced the finding can map one response to a whole file.
        return self.responses.get(key.split("::")[0], _EMPTY)


_EMPTY = json.dumps({"findings": []})


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
    usage: Usage = field(default_factory=Usage)
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
            # The system prompt is byte-identical across every request in a run,
            # and it is the longest fixed prefix, so caching it turns the per-file
            # cost into little more than the file itself.
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": schema},
            },
        )

        if getattr(response, "usage", None) is not None:
            self.usage.add(response.usage)

        # A refusal is a successful HTTP response with an empty or partial body,
        # so stop_reason has to be checked before content is touched.
        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise ProviderError(
                "the model declined this request"
                + (f" (category: {category})" if category else "")
                + "; the audited file may contain content that trips a safety classifier. "
                "Re-run with --no-llm to audit it statically."
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
