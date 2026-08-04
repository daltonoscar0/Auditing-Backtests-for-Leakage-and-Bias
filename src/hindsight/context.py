"""Per-file analysis context.

Most checks need the same handful of facts about a file before they can decide
anything: is this time-ordered data, which column is the target, how far forward
does the label look, does anything in scope resemble a transaction cost. Deriving
those once per file - rather than once per check - is what keeps the static layer
both fast and consistent, since every check then agrees on the answer.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

from . import astutils as A

# --------------------------------------------------------------------------- #
# Lexical vocabularies.
#
# These are heuristics and are treated as such: they gate findings and suppress
# them, and every finding whose only evidence is a name match is emitted at
# reduced confidence.
# --------------------------------------------------------------------------- #

COST_TERMS = re.compile(
    r"cost|fee|commission|slippage|spread|bps|borrow|impact|turnover_cost|financing",
    re.IGNORECASE,
)
BORROW_TERMS = re.compile(r"borrow|loan|stock_loan|hard_to_borrow|htb|financing", re.IGNORECASE)
PURGE_TERMS = re.compile(r"purge|purged|embargo", re.IGNORECASE)
CORRECTION_TERMS = re.compile(
    r"deflated|dsr|bonferroni|holm|multiple_test|family_wise|fdr|white_reality|spa_test",
    re.IGNORECASE,
)
TARGET_NAMES = re.compile(r"^(y|y_true|target|labels?)$", re.IGNORECASE)
TARGET_COLUMN_PREFIXES = re.compile(
    r"^(y|target|label|fwd_|forward_|future_|ret_fwd|next_)", re.IGNORECASE
)
DATE_COLUMN = re.compile(r"^(date|datetime|timestamp|time|dt|bar_time|asof|as_of)$", re.IGNORECASE)
TEST_VARIABLE = re.compile(r"(^|_)(test|holdout|oos)($|_)", re.IGNORECASE)

STRONG_TEMPORAL = {
    "to_datetime",
    "DatetimeIndex",
    "date_range",
    "Timestamp",
    "resample",
    "parse_dates",
    "tz_localize",
    "asfreq",
    "bdate_range",
    "TimeSeriesSplit",
}
WEAK_TEMPORAL = {"shift", "rolling", "ewm", "expanding", "freq", "diff", "pct_change"}

SPLIT_CALLS = {
    "train_test_split",
    "TimeSeriesSplit",
    "KFold",
    "StratifiedKFold",
    "GroupKFold",
    "ShuffleSplit",
    "StratifiedShuffleSplit",
    "GroupShuffleSplit",
    "PurgedKFold",
}

WINDOWING_CALLS = frozenset({"rolling", "expanding", "ewm", "groupby", "resample"})

TRANSFORMER_HINT = re.compile(
    r"scaler|standardscaler|minmaxscaler|robustscaler|normalizer|imputer|pca|"
    r"truncatedsvd|encoder|quantiletransformer|powertransformer|selector|vectorizer",
    re.IGNORECASE,
)


@dataclass
class FileContext:
    """A parsed source file plus the facts checks reason over.

    Deliberately not slotted: every derived fact below is a ``cached_property``,
    which needs an instance ``__dict__`` to memoise into.
    """

    path: str
    """Repo-relative POSIX path, used verbatim in citations."""
    abs_path: Path
    source: str
    tree: ast.Module
    lines: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.lines:
            self.lines = self.source.splitlines()

    # ------------------------------------------------------------------ text
    def line_text(self, lineno: int) -> str:
        if 1 <= lineno <= len(self.lines):
            return self.lines[lineno - 1]
        return ""

    def snippet(self, node: ast.AST, *, max_lines: int = 3) -> str:
        start, end = A.span(node)
        end = min(end, start + max_lines - 1)
        return "\n".join(self.lines[start - 1 : end]).strip()

    def indent_of(self, lineno: int) -> str:
        text = self.line_text(lineno)
        return text[: len(text) - len(text.lstrip())]

    # ------------------------------------------------------------------ shape
    @cached_property
    def parents(self) -> dict[ast.AST, ast.AST]:
        return A.attach_parents(self.tree)

    def enclosing_scope(self, node: ast.AST) -> ast.AST:
        """Nearest enclosing function, else the module."""
        for parent in A.ancestors(node, self.parents):
            if isinstance(parent, ast.FunctionDef | ast.AsyncFunctionDef):
                return parent
        return self.tree

    def enclosing_statement(self, node: ast.AST) -> ast.stmt | None:
        for parent in A.ancestors(node, self.parents):
            if isinstance(parent, ast.stmt):
                return parent
        return None

    def enclosing_assign(self, node: ast.AST) -> ast.Assign | ast.AnnAssign | ast.AugAssign | None:
        for parent in A.ancestors(node, self.parents):
            if isinstance(parent, ast.Assign | ast.AnnAssign | ast.AugAssign):
                return parent
            if isinstance(parent, ast.FunctionDef | ast.AsyncFunctionDef | ast.Module):
                return None
        return None

    def assigned_label(self, node: ast.AST) -> str | None:
        """Name of the column or variable the expression containing `node` is assigned to."""
        assign = self.enclosing_assign(node)
        if assign is None:
            return None
        for target in A.assign_targets(assign):
            label = A.target_label(target)
            if label is not None:
                return label
        return None

    # ------------------------------------------------------------- vocabulary
    @cached_property
    def all_names(self) -> set[str]:
        return A.referenced_names(self.tree) | {
            node.name
            for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        }

    @cached_property
    def all_strings(self) -> set[str]:
        return A.string_constants(self.tree)

    @cached_property
    def all_keywords(self) -> set[str]:
        return {kw.arg for call in A.walk_calls(self.tree) for kw in call.keywords if kw.arg}

    @cached_property
    def vocabulary(self) -> set[str]:
        """Every identifier, attribute, keyword and string literal in the file."""
        return self.all_names | self.all_strings | self.all_keywords

    def matches(self, pattern: re.Pattern[str]) -> bool:
        return any(pattern.search(token) for token in self.vocabulary)

    def scope_vocabulary(self, node: ast.AST) -> set[str]:
        """Vocabulary of the enclosing function, plus module-level code.

        Module level is included because cost and horizon constants are
        conventionally defined at the top of a research script and used inside
        functions further down.
        """
        scope = self.enclosing_scope(node)
        tokens = A.referenced_names(scope) | A.string_constants(scope)
        tokens |= {kw.arg for call in A.walk_calls(scope) for kw in call.keywords if kw.arg}
        if scope is not self.tree:
            tokens |= self.module_level_vocabulary
        return tokens

    @cached_property
    def module_level_vocabulary(self) -> set[str]:
        tokens: set[str] = set()
        for stmt in self.tree.body:
            if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                tokens.add(stmt.name)
                continue
            tokens |= A.referenced_names(stmt) | A.string_constants(stmt)
        return tokens

    def scope_matches(self, node: ast.AST, pattern: re.Pattern[str]) -> bool:
        return any(pattern.search(token) for token in self.scope_vocabulary(node))

    # -------------------------------------------------------------- temporal
    @cached_property
    def is_temporal(self) -> bool:
        """Whether this file handles time-ordered data.

        The gate for the whole UNPURGED-CV category. One strong signal, or two
        weak ones, or an explicitly date-like column name.
        """
        vocab = self.vocabulary
        if vocab & STRONG_TEMPORAL:
            return True
        if any(DATE_COLUMN.match(token) for token in vocab):
            return True
        return len(vocab & WEAK_TEMPORAL) >= 2

    @cached_property
    def temporal_evidence(self) -> list[str]:
        vocab = self.vocabulary
        evidence = sorted(vocab & (STRONG_TEMPORAL | WEAK_TEMPORAL))
        evidence += sorted(t for t in vocab if DATE_COLUMN.match(t))
        return evidence

    # --------------------------------------------------------------- targets
    @cached_property
    def target_columns(self) -> set[str]:
        """Column names identified as the prediction target.

        Resolution order mirrors TAXONOMY.md: the ``y`` argument of a ``.fit``
        call, then assignments to target-named variables, then assignments to
        columns whose names carry a forward-looking prefix.
        """
        targets: set[str] = set()
        aliases: dict[str, str] = {}

        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Assign | ast.AnnAssign):
                continue
            value = node.value
            if value is None:
                continue
            for target in A.assign_targets(node):
                label = A.target_label(target)
                if label is None:
                    continue
                if isinstance(target, ast.Subscript) and TARGET_COLUMN_PREFIXES.match(label):
                    targets.add(label)
                if TARGET_NAMES.match(label):
                    # Prefer the column the variable was read from. Recording the
                    # variable name itself as a target only helps when it cannot be
                    # traced to a column, and hurts otherwise: a bare `y` would
                    # otherwise be matched against every column beginning with y.
                    resolved = A.subscript_key(value) if isinstance(value, ast.Subscript) else None
                    if resolved is None:
                        columns = A.string_constants(value)
                        resolved = next(iter(columns)) if len(columns) == 1 else None
                    if resolved:
                        targets.add(resolved)
                        aliases[label] = resolved
                    else:
                        targets.add(label)

        for call in A.walk_calls(self.tree):
            if A.call_tail(call) not in {"fit", "fit_transform"} or len(call.args) < 2:
                continue
            y_arg = call.args[1]
            if isinstance(y_arg, ast.Subscript):
                key = A.subscript_key(y_arg)
                if key:
                    targets.add(key)
            elif isinstance(y_arg, ast.Name):
                targets.add(aliases.get(y_arg.id, y_arg.id))

        return targets

    @cached_property
    def target_roots(self) -> set[str]:
        """Target names lowercased, for derived-column matching."""
        return {t.lower() for t in self.target_columns}

    def is_target_like(self, name: str | None) -> bool:
        """True for the target itself or any column sharing its root.

        Prefix matching is length-gated. A target literally named ``y`` would
        otherwise claim ``yield_spread`` as a derivative, which is exactly the
        kind of false positive that gets an audit tool switched off.
        """
        if not name:
            return False
        lowered = name.lower()
        for root in self.target_roots:
            if lowered == root:
                return True
            if len(root) >= 3 and lowered.startswith(f"{root}_"):
                return True
            if len(root) >= 5 and lowered.startswith(root):
                return True
        return False

    # ------------------------------------------------------------- constants
    @cached_property
    def int_constants(self) -> dict[str, int]:
        """Module-level ``NAME = <int literal>`` bindings.

        Research scripts put horizons, gaps and window lengths in constants at the
        top of the file and refer to them by name. Without this, every check that
        reasons about a number would give up the moment it was not written inline.
        """
        constants: dict[str, int] = {}
        for stmt in self.tree.body:
            if not isinstance(stmt, ast.Assign | ast.AnnAssign) or stmt.value is None:
                continue
            value = A.const(stmt.value)
            if not isinstance(value, int) or isinstance(value, bool):
                continue
            for target in A.assign_targets(stmt):
                if isinstance(target, ast.Name):
                    constants[target.id] = value
        return constants

    def resolve_int(self, node: ast.AST | None) -> int | None:
        """Fold `node` to an int, following one level of module-constant indirection."""
        if node is None:
            return None
        value = A.const(node)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(node, ast.Name):
            return self.int_constants.get(node.id)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            inner = self.resolve_int(node.operand)
            return None if inner is None else -inner
        return None

    # -------------------------------------------------------------- horizons
    @cached_property
    def forward_horizon(self) -> int | None:
        """Largest forward-looking label horizon in bars, if one is discoverable."""
        horizons: list[int] = []
        for call in A.walk_calls(self.tree):
            if A.call_tail(call) != "shift":
                continue
            value = self.resolve_int(A.arg_or_keyword(call, 0, "periods"))
            if isinstance(value, int) and value < 0:
                horizons.append(-value)
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Assign | ast.AnnAssign):
                continue
            for target in A.assign_targets(node):
                label = A.target_label(target)
                if label and re.search(r"horizon|lookahead|holding_period", label, re.IGNORECASE):
                    value = self.resolve_int(node.value)
                    if isinstance(value, int) and value > 0:
                        horizons.append(value)
        return max(horizons) if horizons else None

    # ---------------------------------------------------------------- splits
    @cached_property
    def split_calls(self) -> list[ast.Call]:
        return [c for c in A.walk_calls(self.tree) if A.call_tail(c) in SPLIT_CALLS]

    @cached_property
    def first_split_line(self) -> int | None:
        lines = [A.span(c)[0] for c in self.split_calls]
        return min(lines) if lines else None

    @cached_property
    def has_purging(self) -> bool:
        return self.matches(PURGE_TERMS)

    @cached_property
    def windowed_names(self) -> set[str]:
        """Variables bound to a rolling, expanding, grouped or resampled view.

        ``history = series.expanding()`` hides the windowing behind a name, and a
        check that only inspects the call chain at the point of use would read
        ``history.mean()`` as a whole-sample statistic. Tracking the binding is
        what stops a correctly-written expanding z-score being reported as a
        full-sample one.
        """
        names: set[str] = set()
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            chain = A.dotted(node.value.func).split(".")
            if not set(chain) & WINDOWING_CALLS:
                continue
            for target in A.assign_targets(node):
                label = A.target_label(target)
                if label:
                    names.add(label)
        return names

    @cached_property
    def pipeline_transformers(self) -> set[int]:
        """ids of Call nodes constructed inside a sklearn Pipeline.

        Transformers wrapped in a Pipeline are refit per fold by the CV machinery,
        so fitting one before a split is not the same defect.
        """
        wrapped: set[int] = set()
        for call in A.walk_calls(self.tree):
            if A.call_tail(call) not in {"Pipeline", "make_pipeline"}:
                continue
            for inner in A.walk_calls(call):
                wrapped.add(id(inner))
        return wrapped

    @cached_property
    def pipeline_variables(self) -> set[str]:
        """Variables bound to a Pipeline, whose ``.fit`` is per-fold safe."""
        names: set[str] = set()
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            if A.call_tail(node.value) in {"Pipeline", "make_pipeline", "GridSearchCV"}:
                for target in A.assign_targets(node):
                    label = A.target_label(target)
                    if label:
                        names.add(label)
        return names


def parse(path: Path, source: str, rel_path: str) -> FileContext | None:
    """Parse `source` into a context, or return None if it does not compile."""
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return None
    return FileContext(path=rel_path, abs_path=path, source=source, tree=tree)
