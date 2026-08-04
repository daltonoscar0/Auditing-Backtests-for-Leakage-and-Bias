"""Repository-level index, so a check can see past the file it is reading.

The single largest source of false positives in the static layer was that every
check saw one file at a time. A cost model defined in ``costs.py`` and applied in
``backtest.py`` looked, from inside ``backtest.py``, exactly like a backtest that
had forgotten costs.

This module resolves imports one level deep: for each file it works out which
names it pulls in from elsewhere in the same repository, and what vocabulary
those definitions contribute. That vocabulary then joins the file's own when a
check asks "is there a cost term in scope".

One level, deliberately. Transitive resolution would let a single mention of the
word `fee` anywhere in a repository suppress every cost finding in it, which
trades a false-positive problem for a worse false-negative one.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from . import astutils as A
from .context import FileContext


def module_name(rel_path: str) -> str:
    """``pkg/mod.py`` -> ``pkg.mod``; ``pkg/__init__.py`` -> ``pkg``."""
    stem = rel_path[:-3] if rel_path.endswith(".py") else rel_path
    parts = [p for p in stem.split("/") if p]
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _definition_vocabulary(node: ast.stmt) -> set[str]:
    return A.referenced_names(node) | A.string_constants(node)


@dataclass
class ModuleIndex:
    """What one module defines at the top level, and the vocabulary of each."""

    name: str
    path: str
    definitions: dict[str, set[str]] = field(default_factory=dict)

    @property
    def vocabulary(self) -> set[str]:
        merged: set[str] = set()
        for tokens in self.definitions.values():
            merged |= tokens
        return merged


def index_module(ctx: FileContext) -> ModuleIndex:
    index = ModuleIndex(name=module_name(ctx.path), path=ctx.path)
    for stmt in ctx.tree.body:
        tokens = _definition_vocabulary(stmt)
        if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            index.definitions[stmt.name] = tokens | {stmt.name}
        elif isinstance(stmt, ast.Assign | ast.AnnAssign):
            for target in A.assign_targets(stmt):
                label = A.target_label(target)
                if label:
                    index.definitions[label] = tokens | {label}
    return index


@dataclass
class ProjectIndex:
    """Every module in the audited repository, keyed by dotted module name."""

    modules: dict[str, ModuleIndex] = field(default_factory=dict)

    @classmethod
    def build(cls, contexts: list[FileContext]) -> ProjectIndex:
        index = cls()
        for ctx in contexts:
            module = index_module(ctx)
            index.modules[module.name] = module
        return index

    def _candidates(self, target: str, importer: str) -> list[ModuleIndex]:
        """Modules a bare import name might refer to.

        Absolute (`from costs import ...`), package-relative
        (`from .costs import ...`), and the flat-script case where `costs.py`
        sits beside the importer, are all resolved the same way: by dotted name,
        then by trailing segment.
        """
        if target in self.modules:
            return [self.modules[target]]

        package = importer.rsplit(".", 1)[0] if "." in importer else ""
        if package:
            qualified = f"{package}.{target}"
            if qualified in self.modules:
                return [self.modules[qualified]]

        tail = target.rsplit(".", 1)[-1]
        return [m for name, m in self.modules.items() if name.rsplit(".", 1)[-1] == tail]

    def imported_vocabulary(self, ctx: FileContext) -> set[str]:
        """Vocabulary reachable from `ctx` through its imports, one level deep."""
        importer = module_name(ctx.path)
        tokens: set[str] = set()

        for node in ast.walk(ctx.tree):
            if isinstance(node, ast.ImportFrom):
                if node.module is None and not node.names:
                    continue

                if node.module:
                    # The bound name counts whether or not the module resolves:
                    # `from vendor.costs import COST_BPS` tells a reader there is
                    # a cost model even though vendor/ is outside the audited tree.
                    for alias in node.names:
                        if alias.name != "*":
                            tokens.add(alias.asname or alias.name)
                    for module in self._candidates(node.module, importer):
                        for alias in node.names:
                            if alias.name == "*":
                                tokens |= module.vocabulary
                            elif alias.name in module.definitions:
                                tokens |= module.definitions[alias.name]
                else:
                    # `from . import costs` names modules, not attributes.
                    for alias in node.names:
                        tokens.add(alias.asname or alias.name)
                        for module in self._candidates(alias.name, importer):
                            tokens |= module.vocabulary

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    tokens.add((alias.asname or alias.name).split(".")[0])
                    for module in self._candidates(alias.name, importer):
                        tokens |= module.vocabulary

        return tokens


def link(contexts: list[FileContext]) -> ProjectIndex:
    """Build the index and attach each file's imported vocabulary to its context."""
    index = ProjectIndex.build(contexts)
    for ctx in contexts:
        ctx.external_vocabulary = index.imported_vocabulary(ctx)
    return index
