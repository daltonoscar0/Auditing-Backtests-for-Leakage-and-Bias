"""Static detection rules, one module per taxonomy category.

Importing this package registers every check. The submodule imports below look
unused and are not: each one populates the registry in ``base`` as a side effect
of class definition.
"""

from __future__ import annotations

from . import cost_fantasy, label_leak, lookahead, survivorship, sweep_overfit, unpurged_cv
from .base import Check, all_checks, get_check, registered_ids, run_all

__all__ = [
    "Check",
    "all_checks",
    "cost_fantasy",
    "get_check",
    "label_leak",
    "lookahead",
    "registered_ids",
    "run_all",
    "survivorship",
    "sweep_overfit",
    "unpurged_cv",
]
