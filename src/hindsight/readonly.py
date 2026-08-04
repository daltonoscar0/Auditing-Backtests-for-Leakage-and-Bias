"""Read-only access to the audited tree.

Hindsight must never modify the repository it is pointed at. Every read of an
audited file goes through :func:`read_text` here, which refuses any mode that
could create or truncate a file. This is a guard rail, not decoration: the
integration test in ``tests/test_readonly.py`` hashes the target tree before and
after a full audit and asserts it is byte-identical.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_MAX_BYTES = 2_000_000
"""Files above this size are skipped. Research repos occasionally check in large
generated modules; parsing them is slow and they are never hand-written logic."""


class ReadOnlyViolation(RuntimeError):
    """Raised when code attempts to open an audited path for anything but reading."""


def read_text(path: Path, *, mode: str = "r") -> str:
    """Read `path` as UTF-8 text. Any mode other than a plain read is rejected."""
    if set(mode) - {"r", "b", "t"}:
        raise ReadOnlyViolation(f"refusing to open {path} with mode {mode!r}")
    with open(path, encoding="utf-8", errors="replace") as handle:
        return handle.read()


def too_large(path: Path) -> bool:
    try:
        return path.stat().st_size > _MAX_BYTES
    except OSError:
        return True


def tree_digest(root: Path) -> str:
    """Stable digest of every file under `root`, used to prove nothing was written."""
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()
