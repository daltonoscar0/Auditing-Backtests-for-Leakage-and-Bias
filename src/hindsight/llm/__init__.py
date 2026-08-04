"""Layer 2: model-backed reasoning passes.

Off by default. The static layer is complete without it, and an audit that cannot
run offline is one nobody will run in CI.
"""

from __future__ import annotations

from .provider import AnthropicProvider, MockProvider, Provider, ProviderError, build_provider
from .reasoning import LLMPass, RejectedFinding, validate
from .schema import FINDINGS_SCHEMA

__all__ = [
    "FINDINGS_SCHEMA",
    "AnthropicProvider",
    "LLMPass",
    "MockProvider",
    "Provider",
    "ProviderError",
    "RejectedFinding",
    "build_provider",
    "validate",
]
