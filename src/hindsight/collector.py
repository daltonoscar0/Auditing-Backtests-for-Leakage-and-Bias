"""Discovery of auditable source files inside a target repository."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from .context import FileContext, parse
from .notebook import flatten
from .readonly import read_text, too_large

SKIP_DIRECTORIES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        ".env",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".ipynb_checkpoints",
        ".tox",
        ".nox",
        "site-packages",
        "build",
        "dist",
        ".eggs",
    }
)

SKIP_SUFFIXES = frozenset({".pyi"})


def is_skipped(path: Path, root: Path) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return True
    if any(part in SKIP_DIRECTORIES for part in parts):
        return True
    return any(part.endswith(".egg-info") for part in parts)


AUDITED_SUFFIXES = (".py", ".ipynb")


def discover(root: Path) -> Iterator[Path]:
    """Yield every source file worth auditing under `root`, in stable order."""
    candidates = sorted(
        (p for suffix in AUDITED_SUFFIXES for p in root.rglob(f"*{suffix}")),
        key=lambda p: p.as_posix(),
    )
    for path in candidates:
        if not path.is_file() or path.is_symlink():
            continue
        if path.suffix in SKIP_SUFFIXES or is_skipped(path, root):
            continue
        if too_large(path):
            continue
        yield path


def collect(root: Path) -> tuple[list[FileContext], list[str]]:
    """Parse every discovered file. Returns the contexts and any non-fatal errors.

    A file that fails to parse is reported rather than raising: audited repos are
    other people's code and may target a different Python version than the one
    Hindsight runs under. Skipping one module is better than refusing to audit.
    """
    contexts: list[FileContext] = []
    errors: list[str] = []
    for path in discover(root):
        rel = path.relative_to(root).as_posix()
        try:
            source = read_text(path)
        except OSError as exc:
            errors.append(f"{rel}: could not read ({exc})")
            continue
        origins = None
        if path.suffix == ".ipynb":
            flattened = flatten(source)
            if flattened is None:
                errors.append(f"{rel}: skipped, not a readable notebook")
                continue
            source, origins = flattened

        context = parse(path, source, rel)
        if context is None:
            errors.append(f"{rel}: skipped, does not parse as Python")
            continue
        context.origins = origins
        contexts.append(context)
    return contexts, errors
