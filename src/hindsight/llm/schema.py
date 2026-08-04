"""JSON schema the model pass must answer with.

The schema is enforced at the provider boundary via structured outputs, so a
malformed response is the provider's problem rather than something the parser
has to defend against. What the parser still has to defend against is a
*well-formed* finding that cites a line the model invented - see
``hindsight.llm.reasoning.validate``.
"""

from __future__ import annotations

from typing import Any

from ..models import Category, Confidence, Severity

FINDINGS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "check_id": {
                        "type": "string",
                        "description": (
                            "The taxonomy check id this finding instantiates, e.g. LA006. "
                            "Must be one of the ids listed in the prompt."
                        ),
                    },
                    "category": {
                        "type": "string",
                        "enum": [c.value for c in Category],
                    },
                    "file": {
                        "type": "string",
                        "description": "Repository-relative path exactly as given in the prompt.",
                    },
                    "line": {
                        "type": "integer",
                        "description": (
                            "1-based line number of the single most relevant line. Must be a "
                            "line that exists in the excerpt shown."
                        ),
                    },
                    "end_line": {"type": "integer"},
                    "title": {"type": "string"},
                    "severity": {
                        "type": "string",
                        "enum": [s.value for s in Severity],
                    },
                    "confidence": {
                        "type": "string",
                        "enum": [c.value for c in Confidence],
                    },
                    "mechanism": {
                        "type": "string",
                        "description": (
                            "Why this biases the result, in plain English, specific to this "
                            "code rather than the general pattern."
                        ),
                    },
                    "quoted_line": {
                        "type": "string",
                        "description": (
                            "The exact text of the cited line, copied verbatim. Used to verify "
                            "the citation points where the finding claims it does."
                        ),
                    },
                },
                "required": [
                    "check_id",
                    "category",
                    "file",
                    "line",
                    "end_line",
                    "title",
                    "severity",
                    "confidence",
                    "mechanism",
                    "quoted_line",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["findings"],
    "additionalProperties": False,
}
