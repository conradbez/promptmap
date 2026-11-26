"""
A promptmap project as a set of nodes to edit, run and inspect — the Python
API behind tools (``pm-code``, an LLM with tool calls, a UI) that build a
pipeline without writing files themselves.

Every edit goes through :class:`~promptmap.construct.Construct` (read the map,
change it, write back only what changed) or :mod:`promptmap.rename`, so the
same rules hold as for ``promptmap edit`` / ``promptmap rename``::

    ws = Workspace(".")
    ws.add_chat_node("spec", "Write a spec for a todo app.")
    ws.add_coding_agent_node("build", "Implement the spec.")
    ws.connect("spec", "build")
    ws.run(["build"])
    print(ws.output("build"))

``coding_agent`` nodes made here write their result into *code_dir* (default:
the project root) through their ``project=`` config key.

Paths are relative to *root*; :meth:`Workspace.run` runs promptmap with *root*
as the working directory, which is what ``project=`` and promptfiles are
relative to.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterator

from promptmap.construct import (
    Construct,
    SourceFile,
    construct_to_files,
    file_to_construct,
)

#: A name that is safe as a file stem and inside ref('...').
_VALID_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]*$")
_CONFIG_CALL = re.compile(r"\{\{-?\s*config\((?P<args>.*?)\)\s*-?\}\}\n?", re.DOTALL)

#: What promptmap itself puts in every template, with how to use it.
BUILTIN_JINJA = {
    "ref": "ref('name') — another node's output; makes this node depend on it. "
           "ref('name.key') picks a key of a JSON output (ref('build.output') for a coding_agent's answer).",
    "promptdata": "promptdata('key') — a run-time value from promptdata.yml or --promptdata.",
    "config": "{{ config(key=value, ...) }} — node settings (model_type, model, each, inherit_coding_folder, ...). "
              "Set through the node tools rather than by hand.",
    "model": "model.name — this node's name.",
    "previous": "{{ previous }} — this node's last successful output (empty on the first run). "
                "previous.output / previous.prompt.",
    "previous_of": "previous_of('name') — another node's last successful output from an earlier run "
                   "(no dependency, no re-run).",
    "each": "{{ each }} — the current item when the node sets config(each='upstream') to run once per list item.",
    "was_skipped": "was_skipped('name') — true when that node was skipped in this run.",
    "skip_and_set_to_value": "{{ skip_and_set_to_value('text') }} — inside an {% if %}: skip the LLM call "
                             "and use 'text' as this node's output.",
    "skip_this_and_downstream": "{{ skip_this_and_downstream() }} — inside an {% if %}: skip this node "
                                "and everything that depends on it.",
    "return_list_RAG_results": "return_list_RAG_results(query) — results from the project's rag.py.",
}


class WorkspaceError(ValueError):
    """An edit that cannot be made as asked; the message says why."""


def _check_name(name: str) -> None:
    if not _VALID_NAME.match(name or ""):
        raise WorkspaceError(f"'{name}' is not a valid node name: use letters, digits, '_' and '-'.")


def split_config(source: str) -> tuple[str, str]:
    """(the ``{{ config(...) }}`` call with its newline, or "", the rest)."""
    m = _CONFIG_CALL.search(source)
    if not m:
        return "", source
    return m.group(0), source[: m.start()] + source[m.end():]


@contextlib.contextmanager
def _chdir(path: Path) -> Iterator[None]:
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


class Workspace:
    """Edit, run and read the nodes of the project at *root*."""

    def __init__(self, root: str | Path = ".", map_dir: str = "map", code_dir: str | Path | None = None) -> None:
        self.root = Path(root).resolve()
        self.map_dir = map_dir
        self.code_dir = (self.root / code_dir).resolve() if code_dir is not None else self.root

    # -- reading -----------------------------------------------------------

    def construct(self) -> Construct:
        from promptmap.llm import try_load_client_module

        try_load_client_module(str(self.root / self.map_dir))  # project-local kinds
        return file_to_construct(self.root, self.map_dir)

    def nodes(self, prefix: str | None = None) -> list[dict[str, Any]]:
        """Every node as ``{name, path, model_type, depends_on}``, by path.

        *prefix* keeps nodes whose path under the map dir starts with it, e.g.
        ``"1_overall/3_inside/"``; a trailing ``*`` is allowed and ignored.
        """
        c = self.construct()
        specs = c.specs() if c.models else {}
        want = (prefix or "").rstrip("*")
        out = []
        for name, f in sorted(c.models.items(), key=lambda kv: kv[1].path):
            if want and not f.path.startswith(want):
                continue
            spec = specs.get(name)
            out.append({
                "name": name,
                "path": f.path,
                "model_type": (spec.model_type if spec else "") or "llm",
                "depends_on": sorted(spec.depends_on) if spec else [],
            })
        return out

    def source(self, name: str) -> str:
        """The node's full template text."""
        return self.construct()._model(name).source

    def output(self, name: str, run_id: str | None = None) -> str | None:
        """The node's latest successful output (or from *run_id*); None if it never succeeded.

        A ``coding_agent``'s output is summarised: its answer, the files it
        changed or deleted, and where they were written.
        """
        row = self._storage().get_latest_output(name, run_id)
        if row is None:
            return None
        from promptmap.files import _TEXT_ESCAPE_PREFIX, ENVELOPE_PREFIX

        text = row["llm_output_validated"] or row["llm_output"] or ""
        if text.startswith(_TEXT_ESCAPE_PREFIX):
            return text[len(_TEXT_ESCAPE_PREFIX):]
        if text.startswith(ENVELOPE_PREFIX):  # holds files: summarise without loading their bytes
            text = text[len(ENVELOPE_PREFIX):]
        try:
            value = json.loads(text)
        except ValueError:
            return text
        if isinstance(value, dict) and "changes" in value and "result" in value:
            summary = {
                "output": value.get("output", ""),
                "changed_files": sorted(value.get("changes") or {}),
                "deleted": value.get("deleted") or [],
                "cache": value.get("cache"),
                "project": value.get("project"),
            }
            return json.dumps(summary, indent=2, default=str)
        return text

    def chat_history(self, name: str) -> list[dict]:
        """A chat node's current conversation (see :func:`promptmap.get_chat`)."""
        from promptmap import get_chat

        return get_chat(name, storage_backend=self._storage())

    def jinja_functions(self) -> dict[str, str]:
        """Every function a template can call: the built-ins and registered ones, with help text."""
        from promptmap.llm import try_load_client_module
        from promptmap.template_functions import get_template_function, known_template_functions, load_project_file

        try_load_client_module(str(self.root / self.map_dir))
        with contextlib.suppress(Exception):
            load_project_file(str(self.root / self.map_dir))
        out = dict(BUILTIN_JINJA)
        for name in sorted(known_template_functions()):
            fn = get_template_function(name)
            doc = (getattr(fn, "__doc__", None) or "").strip().split("\n\n")[0]
            out[name] = " ".join(doc.split()) or "(registered template function; no docstring)"
        return out

    # -- editing -------------------------------------------------------------

    def add_node(self, name: str, body: str, config: dict[str, Any] | None = None, folder: str = "") -> Path:
        """Create node *name* (``<folder>/<name>.prompt``) with *config* and *body*."""
        from promptmap.construct import write_config_call

        _check_name(name)
        c = self.construct()
        if name in c.models:
            raise WorkspaceError(f"A node named '{name}' already exists: {c.models[name].path}.")
        folder = folder.strip("/")
        if ".." in Path(folder).parts:
            raise WorkspaceError("folder must stay inside the map directory.")
        path = f"{folder}/{name}.prompt" if folder else f"{name}.prompt"
        source = write_config_call(body.rstrip() + "\n", {k: v for k, v in (config or {}).items() if v is not None})
        c.models[name] = SourceFile(path, source)
        construct_to_files(c)
        return c.map_path / path

    def add_chat_node(self, name: str, request: str, chat_instruction: str | None = None, folder: str = "") -> Path:
        """A ``model_type="chat"`` node whose request is *request* (needs promptmap[extras] to run)."""
        return self.add_node(name, request, {"model_type": "chat", "chat_instruction": chat_instruction}, folder)

    def add_coding_agent_node(
        self, name: str, task: str, inherit_coding_folder: str | None = None, folder: str = "",
    ) -> Path:
        """A ``coding_agent`` node that writes its result into the code dir."""
        if inherit_coding_folder:
            self._require_coding_agent(inherit_coding_folder)
        return self.add_node(name, task, {
            "model_type": "coding_agent",
            "project": self._project_path(),
            "inherit_coding_folder": inherit_coding_folder,
        }, folder)

    def set_body(self, name: str, body: str) -> None:
        """Replace the node's text, keeping its ``config()`` call exactly as it is."""
        c = self.construct()
        f = c._model(name)
        call, _ = split_config(f.source)
        f.source = call + body.rstrip() + "\n"
        construct_to_files(c)

    def set_config(self, name: str, **values: Any) -> None:
        """Set ``config()`` keys on the node; ``None`` removes a key."""
        c = self.construct()
        c.set_config(name, **values)
        construct_to_files(c)

    def insert_jinja(self, name: str, jinja: str, position: str = "end") -> None:
        """Add *jinja* to the node's text at the ``"start"`` or ``"end"`` (after config())."""
        if position not in ("start", "end"):
            raise WorkspaceError("position must be 'start' or 'end'.")
        c = self.construct()
        f = c._model(name)
        call, body = split_config(f.source)
        body = f"{jinja}\n{body}" if position == "start" else f"{body.rstrip()}\n\n{jinja}\n"
        f.source = call + body
        self._check_renders(f.source, name)
        construct_to_files(c)

    def connect(self, upstream: str, downstream: str, as_folder: bool = False, label: str | None = None) -> str:
        """Make *downstream* read *upstream*; returns what was added.

        By default a ``ref()`` to upstream's output is appended to downstream's
        text (``ref('x.output')`` for a coding_agent: its answer).  With
        *as_folder*, a coding_agent downstream works on upstream coding_agent's
        code instead (``inherit_coding_folder``).
        """
        c = self.construct()
        down = c._model(downstream)
        c._model(upstream)
        if upstream == downstream:
            raise WorkspaceError("A node cannot depend on itself.")
        if as_folder:
            if c.get_config(downstream).get("model_type") != "coding_agent":
                raise WorkspaceError(f"as_folder needs a coding_agent downstream; '{downstream}' is not one.")
            self._require_coding_agent(upstream, c)
            c.set_config(downstream, inherit_coding_folder=upstream)
            added = f'inherit_coding_folder="{upstream}"'
        else:
            target = f"{upstream}.output" if c.get_config(upstream).get("model_type") == "coding_agent" else upstream
            added = f"{label or f'Input from {upstream}'}:\n{{{{ ref('{target}') }}}}"
            call, body = split_config(down.source)
            down.source = f"{call}{body.rstrip()}\n\n{added}\n"
        self._check_acyclic(c)
        construct_to_files(c)
        return added

    def rename(self, old: str, new: str) -> None:
        """Rename a node: its file, every reference to it and its run history."""
        from promptmap.rename import RenameError, apply_rename, plan_rename

        with _chdir(self.root):
            try:
                plan = plan_rename(self.map_dir, old, new)
            except RenameError as exc:
                raise WorkspaceError(str(exc)) from None
            storage = self._storage()
            if storage.model_has_results(new):
                raise WorkspaceError(f"The run database already has results for '{new}'.")
            apply_rename(plan)
            storage.rename_model(old, new, plan.retemplate)

    # -- running -------------------------------------------------------------

    def run(self, select: list[str] | None = None, **kwargs: Any) -> dict[str, Any]:
        """Run the nodes in *select* (and their upstream), or all; ``promptmap.run``'s result."""
        import promptmap

        c = self.construct()
        kwargs.setdefault("promptdata", c.run_promptdata() or None)
        kwargs.setdefault("verbose", False)
        with _chdir(self.root):
            return promptmap.run(
                map_dir=self.map_dir, select=select or None, storage_backend=self._storage(), **kwargs,
            )

    def chat(self, name: str, message: str, **kwargs: Any) -> Any:
        """Send *message* as a follow-up to chat node *name*; its reply (or a ModelError)."""
        return self.run([name], chat_input={name: message}, **kwargs).get(name)

    # -- helpers -------------------------------------------------------------

    def _storage(self):
        from promptmap.storage.sqlite import SQLiteStorageBackend

        storage = SQLiteStorageBackend(self.root / ".promptmap" / "promptmap.db")
        storage.init_db()
        return storage

    def _project_path(self) -> str:
        rel = os.path.relpath(self.code_dir, self.root)
        return "." if rel == "." else Path(rel).as_posix()

    def _require_coding_agent(self, name: str, c: Construct | None = None) -> None:
        c = c or self.construct()
        c._model(name)
        if c.get_config(name).get("model_type") != "coding_agent":
            raise WorkspaceError(f"'{name}' is not a coding_agent node, so it has no code folder to inherit.")

    def _check_acyclic(self, c: Construct) -> None:
        import networkx as nx

        from promptmap.executor.graph import CyclicDependencyError, build_dag

        try:
            dag = build_dag(c.specs())
        except CyclicDependencyError as exc:
            raise WorkspaceError(f"That would make a cycle: {exc}") from None
        if not nx.is_directed_acyclic_graph(dag):
            raise WorkspaceError(f"That would make a cycle: {nx.find_cycle(dag)}.")

    def _check_renders(self, source: str, name: str) -> None:
        from jinja2 import Environment, TemplateSyntaxError

        try:
            Environment().parse(source)
        except TemplateSyntaxError as exc:
            raise WorkspaceError(f"'{name}' would not parse as Jinja: {exc}") from None
