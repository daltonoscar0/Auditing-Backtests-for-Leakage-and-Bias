"""Report renderers."""

from __future__ import annotations

from . import json_report, markdown

RENDERERS = {"md": markdown.render, "json": json_report.render}

__all__ = ["RENDERERS", "json_report", "markdown"]
