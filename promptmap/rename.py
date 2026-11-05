"""
``promptmap rename OLD NEW`` — rename a model without losing its history.

A model's name is its file name, and the run database keys its history on
that name: the previous result (``{{ previous }}``, ``coding_agent`` reuse and
``warm_start``) is looked up by model name plus template hash.  Renaming the
file by hand therefore costs a cold run, and leaves every ``ref('OLD')`` broken.

:func:`plan_rename` works out every change without touching anything, so the
CLI can show it (``--dry-run``) before :func:`apply_rename` makes it:

* the model's ``.prompt`` file, keeping its folder and sequence prefix
  (``3_a_old.prompt`` → ``3_a_new.prompt``), and ``validation/OLD.py``;
* references in every other model and test: ``ref('OLD')``, ``ref("OLD.key")``
  and ``config(promptfiles=..., each=...)`` values naming ``OLD``;
* ``given`` / ``expect`` keys in YAML test cases;
* the run database: ``OLD``'s rows (and its ``each=`` items ``OLD[i]``) move to
  ``NEW``, and every model whose file was rewritten has the stored template
  hash of its old source updated to the new one — so downstream models keep
  their previous result too.

Prose that merely mentions the name is left alone; only the forms above are
rewritten.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from promptmap.executor.graph import _prompt_name, load_models

#: A name that is safe as a file stem and inside ref('...').
_VALID_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]*$")

_TEST_SUFFIXES = (".testprompt",)
_YAML_SUFFIXES = (".yml", ".yaml")


class RenameError(ValueError):
    """The rename cannot be done as asked."""


@dataclass
class RenamePlan:
    old: str
    new: str
    #: (from, to) file moves.
    moves: list[tuple[Path, Path]] = field(default_factory=list)
    #: path -> (current text, new text) for files whose references change.
    edits: dict[Path, tuple[str, str]] = field(default_factory=dict)
    #: model name (after the rename) -> (old source, new source) for models
    #: whose own template text changes, so the database can follow.
    retemplate: dict[str, tuple[str, str]] = field(default_factory=dict)


def plan_rename(map_dir: str | Path, old: str, new: str, validation_dir: str | Path = "validation") -> RenamePlan:
    """Every file change renaming model *old* to *new* needs. Touches nothing."""
    map_dir = Path(map_dir)
    if not _VALID_NAME.match(new):
        raise RenameError(f"'{new}' is not a valid model name: use letters, digits, '_' and '-'.")
    if old == new:
        raise RenameError("The old and new names are the same.")
    models = load_models(map_dir)
    if old not in models:
        raise RenameError(f"No model named '{old}' in {map_dir}/ (see `promptmap ls`).")
    if new in models:
        raise RenameError(f"A model named '{new}' already exists: {models[new].path}.")

    plan = RenamePlan(old=old, new=new)

    src = Path(models[old].path)
    prefix_and_name = src.name[: len(src.name) - len(_suffix(src))]
    prefix = prefix_and_name[: len(prefix_and_name) - len(old)]
    dst = src.with_name(f"{prefix}{new}{_suffix(src)}")
    if _prompt_name(dst) != new:
        raise RenameError(f"'{dst.name}' would not be read as model '{new}'.")
    plan.moves.append((src, dst))

    validator = Path(validation_dir) / f"{old}.py"
    if validator.is_file():
        target = validator.with_name(f"{new}.py")
        if target.exists():
            raise RenameError(f"{target} already exists.")
        plan.moves.append((validator, target))

    for name, spec in models.items():
        path = Path(spec.path)
        text = path.read_text(encoding="utf-8")
        changed = rewrite_source(text, old, new)
        if changed != text:
            plan.edits[path] = (text, changed)
            plan.retemplate[new if name == old else name] = (text, changed)

    for path in sorted(map_dir.rglob("*")):
        if not path.is_file() or path in plan.edits:
            continue
        if path.name.endswith(_TEST_SUFFIXES):
            rewrite = rewrite_source
        elif path.suffix in _YAML_SUFFIXES:
            rewrite = rewrite_yaml
        else:
            continue
        text = path.read_text(encoding="utf-8")
        changed = rewrite(text, old, new)
        if changed != text:
            plan.edits[path] = (text, changed)
    return plan


def apply_rename(plan: RenamePlan) -> None:
    """Write the edits, then move the files (an edited file moves with its edit)."""
    for path, (_, text) in plan.edits.items():
        path.write_text(text, encoding="utf-8")
    for src, dst in plan.moves:
        src.rename(dst)


# ---------------------------------------------------------------------------
# Text rewrites
# ---------------------------------------------------------------------------

def _suffix(path: Path) -> str:
    return ".prompt.jinja" if path.name.endswith(".prompt.jinja") else path.suffix


def _names_old(old: str) -> str:
    """A quoted string naming *old*: 'old', "old.key", 'old[0]' (group ``q`` = quote)."""
    return r"""(?P<q>['"])""" + re.escape(old) + r"""(?=(?P=q)|[.\[])"""


_CONFIG_CALL = re.compile(r"\{\{-?\s*config\((.*?)\)\s*-?\}\}", re.DOTALL)
_MODEL_ARGS = re.compile(r"""\b(promptfiles|each)\s*=\s*(\[[^\]]*\]|'[^']*'|"[^"]*")""")


def rewrite_source(text: str, old: str, new: str) -> str:
    """*text* (a .prompt or .testprompt) with references to model *old* renamed."""
    quoted = re.compile(_names_old(old))

    def requote(m: re.Match) -> str:
        return f"{m.group('q')}{new}"

    text = re.sub(r"(?P<call>\bref\(\s*)" + _names_old(old), lambda m: f"{m.group('call')}{m.group('q')}{new}", text)

    def in_config(m: re.Match) -> str:
        args = _MODEL_ARGS.sub(lambda a: a.group(0)[: a.start(2) - a.start(0)] + quoted.sub(requote, a.group(2)), m.group(1))
        return m.group(0)[: m.start(1) - m.start(0)] + args + m.group(0)[m.end(1) - m.start(0):]

    return _CONFIG_CALL.sub(in_config, text)


_SECTION = re.compile(r"^(\s*)(given|expect):\s*(\{.*)?$")
_FLOW_KEY_AFTER = r"(?<=[{,])(\s*)"


def rewrite_yaml(text: str, old: str, new: str) -> str:
    """YAML test cases with ``given`` / ``expect`` keys naming *old* renamed.

    Line-based, so comments and layout survive: block keys directly under a
    ``given:`` / ``expect:`` line, and keys of an inline ``{...}`` mapping.
    """
    flow = re.compile(_FLOW_KEY_AFTER + re.escape(old) + r"(\s*:)")
    out: list[str] = []
    section_indent: int | None = None
    child_indent: int | None = None
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if section_indent is not None and stripped and not stripped.startswith("#"):
            if indent <= section_indent:
                section_indent = child_indent = None
            else:
                child_indent = indent if child_indent is None else child_indent
                if indent == child_indent:
                    line = re.sub(rf"^(\s*){re.escape(old)}(\s*:)", rf"\g<1>{new}\g<2>", line)
        m = _SECTION.match(line.rstrip("\n"))
        if m:
            if m.group(3):  # given: {old: ..., other: ...}
                line = line[: m.start(3)] + flow.sub(rf"\g<1>{new}\g<2>", line[m.start(3):])
            else:
                section_indent, child_indent = len(m.group(1)), None
        out.append(line)
    return "".join(out)
