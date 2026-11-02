"""
Semantic file diffs for ``coding_agent``, via tree-sitter-language-pack.

For each changed file this answers two questions:

* **Did the meaning change?**  A file's *fingerprint* hashes its syntax tree
  with comments left out (except directives such as ``# type: ignore`` or
  ``//go:build``) and with no whitespace between tokens, so reformatting or
  editing a comment leaves it unchanged.  Indentation that carries meaning
  (Python blocks, YAML nesting) is part of the tree's shape, so it still counts.
* **What changed?**  Where the language exposes its structure (functions,
  classes, methods, structs, interfaces, …), the change is summarised per
  symbol: ``ADDED_SYMBOL``, ``REMOVED_SYMBOL``, ``SIGNATURE_CHANGED``,
  ``BODY_CHANGED``; anything outside a symbol is ``MODULE_CHANGED``.

Files with no grammar, files that do not parse, and binary files fall back to
comparing bytes.  Grammars are downloaded by tree-sitter-language-pack on first
use and cached; one that cannot be loaded (offline, say) is skipped for the
rest of the process and its files compared as text.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Iterable

log = logging.getLogger("promptmap")

#: Comments that change what tools do, so editing one is a real change.
_DIRECTIVE = re.compile(
    r"^\s*(?:#!|#\s*(?:type:|noqa|pragma|pylint:|mypy:|fmt:|pyright:|-\*-)"
    r"|//\s*(?:go:|\+build|@ts-|eslint|nolint|prettier-ignore|#)"
    r"|/\*\s*(?:eslint|@ts-|prettier-ignore|global)"
    r"|--\s*noqa)",
    re.IGNORECASE,
)

_unloadable: set[str] = set()


def _pack():
    try:
        import tree_sitter_language_pack
    except ImportError:  # optional at runtime: fall back to text everywhere
        return None
    return tree_sitter_language_pack


def language_for(path: str) -> str | None:
    """The tree-sitter language for *path*, or None if there is none usable."""
    pack = _pack()
    if pack is None:
        return None
    try:
        lang = pack.detect_language_from_path(path)
    except Exception:
        return None
    return lang if lang and lang not in _unloadable else None


# ---------------------------------------------------------------------------
# Per-file analysis
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Symbol:
    kind: str
    signature: str
    fingerprint: str


@dataclass(frozen=True)
class Semantics:
    #: Equal fingerprints mean the files mean the same.
    fingerprint: str
    #: Qualified name → symbol, or None when the file was compared as text.
    symbols: dict[str, Symbol] | None = None

    @property
    def parsed(self) -> bool:
        return self.symbols is not None


def analyse(path: str, data: bytes) -> Semantics:
    """The semantics of *data*, read as the language *path* names."""
    lang = language_for(path)
    if lang is not None:
        found = _analyse(lang, data)
        if found is not None:
            return found
    return Semantics(fingerprint="raw:" + hashlib.sha256(data).hexdigest())


@lru_cache(maxsize=4096)
def _analyse(lang: str, data: bytes) -> Semantics | None:
    pack = _pack()
    try:
        parser = pack.get_parser(lang)
    except Exception as exc:
        _unloadable.add(lang)
        log.info("coding_agent: no %s grammar (%s); comparing those files as text", lang, exc)
        return None
    tree = parser.parse(data)
    if tree.root_node.has_error:
        return None  # a broken file: its bytes are all we can trust
    symbols: dict[str, Symbol] = {}
    try:
        text = data.decode("utf-8")
        result = pack.process(text, pack.ProcessConfig(lang, structure=True))
        _collect(result.structure, "", tree.root_node, data, symbols)
    except Exception:
        log.debug("coding_agent: no structure for %s", lang, exc_info=True)
    return Semantics(fingerprint=_fingerprint(tree.root_node, data), symbols=symbols)


def _collect(items, prefix: str, root, data: bytes, out: dict[str, Symbol]) -> None:
    for item in items:
        name = f"{prefix}{item.name or '<anonymous>'}"
        qual, n = name, 2
        while qual in out:  # overloads, repeated impl blocks, …
            qual, n = f"{name}#{n}", n + 1
        node = root.descendant_for_byte_range(item.span.start_byte, item.span.end_byte)
        out[qual] = Symbol(
            kind=str(item.kind),
            signature=" ".join((item.signature or "").split()),
            fingerprint=_fingerprint(node, data) if node is not None else "",
        )
        _collect(item.children or [], f"{qual}.", root, data, out)


def _fingerprint(node, data: bytes) -> str:
    """A hash of *node*'s tree: node types and token text, comments dropped."""
    digest = hashlib.sha256()
    stack = [node]
    while stack:
        current = stack.pop()
        if current is None:
            digest.update(b")")
            continue
        text = data[current.start_byte:current.end_byte]
        if "comment" in current.type and not _DIRECTIVE.match(text.decode("utf-8", "replace")):
            continue
        if current.child_count == 0:
            digest.update(b"\x00" + text)
            continue
        digest.update(b"(" + current.type.encode())
        stack.append(None)
        stack.extend(reversed(current.children))
    return "ast:" + digest.hexdigest()


# ---------------------------------------------------------------------------
# Tree diffs
# ---------------------------------------------------------------------------

@dataclass
class FileChange:
    path: str
    #: added | deleted | modified | format_only | binary
    status: str
    details: list[str] = field(default_factory=list)

    @property
    def semantic(self) -> bool:
        return self.status != "format_only"

    def lines(self) -> list[str]:
        head = {
            "added": "ADDED_FILE", "deleted": "DELETED_FILE", "modified": "MODIFIED_FILE",
            "format_only": "FORMAT_ONLY (comments/whitespace)", "binary": "BINARY_CHANGED",
        }[self.status]
        return [f"{self.path}: {head}"] + [f"{self.path}: {d}" for d in self.details]


def diff_file(path: str, old: bytes | None, new: bytes | None) -> FileChange:
    if old is None and new is None:
        raise ValueError(f"{path}: nothing to diff")
    if not _is_text(old) or not _is_text(new):
        return FileChange(path, "binary")
    if old is None or new is None:
        sem = analyse(path, old if new is None else new)
        names = sorted(sem.symbols or {})
        label = "symbols" if new is not None else "removed symbols"
        return FileChange(
            path, "added" if old is None else "deleted",
            [f"{label}: {', '.join(names)}"] if names else [],
        )
    a, b = analyse(path, old), analyse(path, new)
    if a.fingerprint == b.fingerprint:
        return FileChange(path, "format_only" if old != new else "modified")
    if not (a.parsed and b.parsed):
        return FileChange(path, "modified", ["TEXT_CHANGED"])
    return FileChange(path, "modified", _symbol_changes(a.symbols, b.symbols))


def _symbol_changes(before: dict[str, Symbol], after: dict[str, Symbol]) -> list[str]:
    changes: dict[str, str] = {}
    for name in sorted(before.keys() | after.keys()):
        old, new = before.get(name), after.get(name)
        if old is None:
            changes[name] = f"ADDED_SYMBOL {new.kind} {name}: {new.signature}"
        elif new is None:
            changes[name] = f"REMOVED_SYMBOL {old.kind} {name}: {old.signature}"
        elif old.signature != new.signature:
            changes[name] = f"SIGNATURE_CHANGED {name}: {old.signature} -> {new.signature}"
        elif old.fingerprint != new.fingerprint:
            changes[name] = f"BODY_CHANGED {new.kind} {name}"
    # A class whose only change is inside a reported method is not news.
    reported = list(changes)
    for name in reported:
        if changes[name].startswith("BODY_CHANGED") and any(o.startswith(name + ".") for o in reported):
            del changes[name]
    return list(changes.values()) or ["MODULE_CHANGED (outside any symbol: imports, top-level code, …)"]


def diff_trees(
    paths: Iterable[str],
    old: Callable[[str], bytes | None],
    new: Callable[[str], bytes | None],
) -> list[FileChange]:
    return [diff_file(p, old(p), new(p)) for p in sorted(paths) if old(p) != new(p)]


def summary(changes: list[FileChange]) -> list[str]:
    return [line for change in changes for line in change.lines()]


def _is_text(data: bytes | None) -> bool:
    if data is None:
        return True
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True
