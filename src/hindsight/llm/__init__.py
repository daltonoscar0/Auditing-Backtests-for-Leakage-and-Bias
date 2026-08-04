"""Layer 2: model-backed reasoning passes.

Off by default. The static layer is complete without it, and an audit that cannot
run offline is one nobody will run in CI.
"""

from __future__ import annotations

from .cassette import Cassette, CassetteProvider, RecordingProvider
from .provider import AnthropicProvider, MockProvider, Provider, ProviderError, build_provider
from .reasoning import LLMPass, RejectedFinding, validate
from .schema import FINDINGS_SCHEMA

__all__ = [
    "FINDINGS_SCHEMA",
    "AnthropicProvider",
    "Cassette",
    "CassetteProvider",
    "LLMPass",
    "MockProvider",
    "Provider",
    "ProviderError",
    "RecordingProvider",
    "RejectedFinding",
    "build_provider",
    "validate",
]
