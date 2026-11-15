"""
A promptmap project as one Python object, and back.

:func:`file_to_construct` reads a project folder — the ``.prompt`` and
``.testprompt`` files under ``map/`` plus ``promptdata.yml`` and
``promptfiles.yml`` at the root — into a :class:`Construct`.
:func:`construct_to_files` writes a construct back, touching only files whose
content changed.  Nothing is ever deleted.

``secrets.yml`` is deliberately left out: a construct gets printed, diffed and
edited, and credentials should not travel with it.

Editing helpers work on the construct, not the disk::

    c = file_to_construct("my_project")
    c.promptdata["topic"] = "cats"
    c.set_config("article", model="low")
    construct_to_files(c)
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from promptmap.project_values import PROMPTDATA_FILE, PROMPTFILES_FILE, load_values

_PROMPT_SUFFIXES = (".prompt.jinja", ".prompt")
_TEST_SUFFIX = ".testprompt"

#: The first ``{{ config(...) }}`` call in a template, with its argument text.
_CONFIG_CALL = re.compile(r"\{\{-?\s*config\((?P<args>.*?)\)\s*-?\}\}", re.DOTALL)


@dataclass
class SourceFile:
    """One template file: its path relative to the map dir, and its text."""

    path: str
    source: str


@dataclass
class Construct:
    """Everything promptmap reads from a project folder to run it."""

    root: Path
    map_dir: str = "map"
    #: model name → file.  Names follow the loader's rules (sequence prefixes stripped).
    models: dict[str, SourceFile] = field(default_factory=dict)
    #: test name → file.
    tests: dict[str, SourceFile] = field(default_factory=dict)
    #: promptdata.yml, including keys with empty values.
    promptdata: dict[str, str] = field(default_factory=dict)
    #: promptfiles.yml, including keys with empty values.
    promptfiles: dict[str, str] = field(default_factory=dict)

    # -- derived views -----------------------------------------------------

    @property
    def map_path(self) -> Path:
        return self.root / self.map_dir

    def specs(self) -> dict:
        """Parse every model into a ``ModelSpec`` (what ``load_models`` returns)."""
        from promptmap.executor.graph import _link_promptfile_deps, build_spec

        specs = {
            name: build_spec(name, f.source, self.map_path / f.path)
            for name, f in self.models.items()
        }
        return _link_promptfile_deps(specs)

    def expected_promptdata(self) -> list[str]:
        """Every ``promptdata()`` key the models and tests read, first-seen order."""
        from promptmap.executor.parser_initial import detect_used_promptdata

        seen: dict[str, None] = {}
        for f in [*self.models.values(), *self.tests.values()]:
            for key in detect_used_promptdata(f.source):
                seen[key] = None
        return list(seen)

    def expected_promptfiles(self) -> list[str]:
        """Every run-level promptfile name the models declare."""
        from promptmap.executor.graph import get_dag_promptfiles

        return get_dag_promptfiles(self.specs())

    def run_promptdata(self) -> dict[str, str]:
        """promptdata values as a run uses them: blank values left out."""
        return {k: v for k, v in self.promptdata.items() if v not in (None, "")}

    def run_promptfiles(self) -> dict[str, str]:
        """promptfile paths as a run uses them, relative to the cwd like before."""
        return {k: v for k, v in self.promptfiles.items() if v not in (None, "")}

    # -- config() editing ----------------------------------------------------

    def get_config(self, name: str, /) -> dict[str, Any]:
        """The model's ``config()`` keys as Python values (unparseable ones as source text)."""
        return read_config_call(self._model(name).source)

    def set_config(self, name: str, /, **values: Any) -> None:
        """Set ``config()`` keys on model *name*; a value of ``None`` removes the key."""
        f = self._model(name)
        f.source = write_config_call(f.source, values)

    def _model(self, name: str) -> SourceFile:
        try:
            return self.models[name]
        except KeyError:
            raise KeyError(f"No model named '{name}'. Models: {', '.join(sorted(self.models))}") from None


# ---------------------------------------------------------------------------
# folder → construct
# ---------------------------------------------------------------------------

def model_name(path: Path) -> str:
    from promptmap.executor.graph import _prompt_name

    return _prompt_name(path)


def read_map(map_dir: str | Path) -> tuple[dict[str, SourceFile], dict[str, SourceFile]]:
    """Read the models and tests under *map_dir*, enforcing unique names."""
    map_dir = Path(map_dir)
    models: dict[str, SourceFile] = {}
    tests: dict[str, SourceFile] = {}
    if not map_dir.exists():
        return models, tests
    for p in sorted({*map_dir.rglob("*.prompt"), *map_dir.rglob("*.prompt.jinja")}):
        name = model_name(p)
        if name in models:
            raise ValueError(
                f"Duplicate model name '{name}': found in both "
                f"'{(map_dir / models[name].path).resolve()}' and '{p.resolve()}'. "
                "Model names must be unique across all subdirectories."
            )
        models[name] = SourceFile(p.relative_to(map_dir).as_posix(), p.read_text(encoding="utf-8"))
    for p in sorted(map_dir.rglob(f"*{_TEST_SUFFIX}")):
        name = p.name[: -len(_TEST_SUFFIX)]
        if name in tests:
            raise ValueError(
                f"Duplicate test name '{name}': found in both '{map_dir / tests[name].path}' "
                f"and '{p}'. Test names must be unique across all subdirectories."
            )
        tests[name] = SourceFile(p.relative_to(map_dir).as_posix(), p.read_text(encoding="utf-8"))
    return models, tests


def _read_all_values(path: Path) -> dict[str, str]:
    """Like ``load_values`` but keeps blank keys, so editing can show them."""
    load_values(path)  # same validation and error messages
    if not path.is_file():
        return {}
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {str(k): "" if v is None else str(v) for k, v in doc.items()}


def file_to_construct(root: str | Path = ".", map_dir: str = "map") -> Construct:
    """Read the project at *root* into a :class:`Construct`.

    *map_dir* is relative to *root* (or absolute).  A missing map dir or values
    file reads as empty; call :meth:`Construct.specs` to get the loader's errors.
    """
    root = Path(root)
    models, tests = read_map(root / map_dir)
    return Construct(
        root=root,
        map_dir=map_dir,
        models=models,
        tests=tests,
        promptdata=_read_all_values(root / PROMPTDATA_FILE),
        promptfiles=_read_all_values(root / PROMPTFILES_FILE),
    )


# ---------------------------------------------------------------------------
# construct → folder
# ---------------------------------------------------------------------------

def _write_if_changed(path: Path, text: str) -> bool:
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return True


def _values_text(path: Path, values: dict[str, str]) -> str:
    """YAML for *values*, keeping the existing file's leading comment block."""
    header: list[str] = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.startswith("#"):
                break
            header.append(line)
    body = yaml.safe_dump(values, sort_keys=False, allow_unicode=True, width=10_000) if values else ""
    return "\n".join(header) + ("\n" if header else "") + body


def construct_to_files(construct: Construct, root: str | Path | None = None) -> list[Path]:
    """Write *construct* to *root* (default: where it was read from).

    Returns the files that changed.  Files on disk that the construct does not
    mention are left alone.  A values file is only rewritten when its values
    changed, so comments survive a no-op round trip.
    """
    root = Path(root) if root is not None else construct.root
    map_path = root / construct.map_dir
    changed: list[Path] = []
    for f in [*construct.models.values(), *construct.tests.values()]:
        p = map_path / f.path
        if _write_if_changed(p, f.source):
            changed.append(p)
    for name, values in ((PROMPTDATA_FILE, construct.promptdata), (PROMPTFILES_FILE, construct.promptfiles)):
        p = root / name
        if _read_all_values(p) == values:
            continue
        if _write_if_changed(p, _values_text(p, values)):
            changed.append(p)
    return changed


# ---------------------------------------------------------------------------
# config() call read / write
# ---------------------------------------------------------------------------

def _parse_call_args(args: str) -> list[tuple[str, ast.expr, str]]:
    """(key, node, source text) for each keyword in a ``config(...)`` argument list."""
    text = f"config({args})"
    call = ast.parse(text, mode="eval").body
    assert isinstance(call, ast.Call)
    return [(kw.arg, kw.value, ast.get_source_segment(text, kw.value) or "") for kw in call.keywords if kw.arg]


def read_config_call(source: str) -> dict[str, Any]:
    """The keywords of the template's ``config()`` call.

    Literal values come back as Python values; anything else (a Jinja
    expression) comes back as its source text.
    """
    m = _CONFIG_CALL.search(source)
    if not m:
        return {}
    out: dict[str, Any] = {}
    for key, node, text in _parse_call_args(m.group("args")):
        try:
            out[key] = ast.literal_eval(node)
        except ValueError:
            out[key] = text
    return out


def _literal(value: Any) -> str:
    """A Jinja literal for *value*."""
    if isinstance(value, bool):
        return "True" if value else "False"
    return json.dumps(value, ensure_ascii=False)


def write_config_call(source: str, values: dict[str, Any]) -> str:
    """Return *source* with *values* set in its ``config()`` call.

    Existing keys keep their place and their original text unless changed; new
    keys go at the end; ``None`` removes a key.  With no ``config()`` call, one is
    added on the first line.  If every key is removed, the call is removed.
    """
    m = _CONFIG_CALL.search(source)
    items: dict[str, str] = {}
    if m:
        items = {k: text for k, _, text in _parse_call_args(m.group("args"))}
    for key, value in values.items():
        if value is None:
            items.pop(key, None)
        else:
            items[key] = _literal(value)
    call = "{{ config(" + ", ".join(f"{k}={v}" for k, v in items.items()) + ") }}" if items else ""
    if m:
        start, end = m.span()
        if not call and source[end:end + 1] == "\n":
            end += 1
        return source[:start] + call + source[end:]
    return f"{call}\n{source}" if call else source


def parse_cli_value(raw: str) -> Any:
    """``true``/``3``/``["a"]`` → JSON value; anything else stays a string."""
    try:
        value = json.loads(raw)
    except ValueError:
        return raw
    return value if isinstance(value, (bool, int, float, list, dict)) else raw
