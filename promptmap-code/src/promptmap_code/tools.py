"""
The tools pm-code's model can call, and what each does.

Every tool goes through :class:`promptmap.workspace.Workspace`: the model never
writes a file itself.  Map edits land in ``map/``; code lands in the code dir
only when a ``coding_agent`` node runs.  ``read_code_dir`` is read-only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from promptmap.workspace import Workspace

#: Directories read_code_dir never lists.
_SKIP_DIRS = {".git", ".promptmap", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache"}
_MAX_LIST = 400


def _fn(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required or []},
        },
    }


_S = {"type": "string"}
_FOLDER = {"type": "string", "description": "Subfolder of map/ to put the node in, e.g. '1_plan/2_api'. Optional."}

TOOLS: list[dict] = [
    _fn("list_nodes", "List the pipeline's nodes: name, path under map/, model_type, depends_on.",
        {"prefix": {"type": "string", "description": "Only nodes whose path starts with this, e.g. "
                                                     "'1_overall/3_inside/*'. Optional."}}),
    _fn("show_node", "Show a node's full template text (config() call and prompt).", {"name": _S}, ["name"]),
    _fn("create_chat_node",
        "Create a chat node: an LLM answer that can be refined with follow-ups (chat_with_node). "
        "Its request is the prompt text; use connect_nodes to feed it other nodes' outputs.",
        {"name": _S, "request": _S, "folder": _FOLDER,
         "chat_instruction": {"type": "string", "description": "System message. Optional."}},
        ["name", "request"]),
    _fn("create_coding_agent_node",
        "Create a coding_agent node: a coding agent that does `task` and writes the resulting code into the "
        "code directory when the node runs. The agent's working folder IS the code directory's root: name "
        "paths from there ('create hello.py', 'edit src/app.py'), never 'in the code directory'. Without "
        "inherit_coding_folder it starts from an empty folder.",
        {"name": _S, "task": _S, "folder": _FOLDER,
         "inherit_coding_folder": {"type": "string",
                                   "description": "An upstream coding_agent node whose code this one continues. Optional."}},
        ["name", "task"]),
    _fn("set_node_prompt", "Replace a node's prompt text. Its config() is kept. Any ref() you leave out is disconnected.",
        {"name": _S, "prompt": _S}, ["name", "prompt"]),
    _fn("rename_node", "Rename a node, updating every ref() to it and keeping its run history.",
        {"old": _S, "new": _S}, ["old", "new"]),
    _fn("connect_nodes",
        "Feed upstream's output into downstream (adds a ref()). With as_folder=true, a coding_agent downstream "
        "instead continues an upstream coding_agent's code.",
        {"upstream": _S, "downstream": _S, "as_folder": {"type": "boolean"},
         "label": {"type": "string", "description": "Text before the inserted output. Optional."}},
        ["upstream", "downstream"]),
    _fn("list_jinja_functions",
        "List every Jinja function a node's template can use (ref, promptdata, previous, previous_of, "
        "skip_and_set_to_value, registered ones, ...), with how to call each.", {}),
    _fn("add_jinja",
        "Add Jinja to a node's template (at the end, or the start after config()). Checked for syntax first. "
        "Use list_jinja_functions to see what is available.",
        {"name": _S, "jinja": _S, "position": {"type": "string", "enum": ["end", "start"]}},
        ["name", "jinja"]),
    _fn("run_nodes", "Run nodes (and everything upstream of them), or the whole pipeline. Unchanged nodes are cached. "
                     "coding_agent nodes write their code into the code directory.",
        {"names": {"type": "array", "items": _S, "description": "Nodes to run; empty runs all."}}),
    _fn("chat_with_node", "Send a follow-up to a chat node; it reruns and returns the new reply.",
        {"name": _S, "message": _S}, ["name", "message"]),
    _fn("read_node_output", "Read a node's latest successful output (a coding_agent's answer and changed files).",
        {"name": _S, "max_chars": {"type": "integer"}}, ["name"]),
    _fn("read_code_dir",
        "Read the code directory (read-only): a folder lists its files, a file shows its text.",
        {"path": {"type": "string", "description": "Path relative to the code directory; default '.'."},
         "max_chars": {"type": "integer"}}),
]


@dataclass
class ToolResult:
    output: str
    ok: bool = True


class Toolbox:
    """Runs a tool call against a :class:`Workspace`."""

    def __init__(self, workspace: Workspace, max_chars: int = 20_000,
                 progress: Callable[[str, str, str], None] | None = None) -> None:
        self.ws = workspace
        self.max_chars = max_chars
        #: ``progress(event, node, detail)`` while nodes run: event is
        #: "start", "step" (detail "3" or "3/40") or "done" (detail: status).
        self.progress = progress
        self._handlers: dict[str, Callable[..., str]] = {
            "list_nodes": self.list_nodes,
            "show_node": lambda name: self.ws.source(name),
            "create_chat_node": self.create_chat_node,
            "create_coding_agent_node": self.create_coding_agent_node,
            "set_node_prompt": self.set_node_prompt,
            "rename_node": self.rename_node,
            "connect_nodes": self.connect_nodes,
            "list_jinja_functions": self.list_jinja_functions,
            "add_jinja": self.add_jinja,
            "run_nodes": self.run_nodes,
            "chat_with_node": self.chat_with_node,
            "read_node_output": self.read_node_output,
            "read_code_dir": self.read_code_dir,
        }

    @property
    def names(self) -> list[str]:
        return list(self._handlers)

    def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        handler = self._handlers.get(name)
        if handler is None:
            return ToolResult(f"Unknown tool '{name}'. Tools: {', '.join(self._handlers)}", ok=False)
        try:
            return ToolResult(handler(**args))
        except TypeError as exc:
            return ToolResult(f"Bad arguments for {name}: {exc}", ok=False)
        except (ValueError, KeyError, FileNotFoundError, AttributeError, ImportError) as exc:
            msg = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
            return ToolResult(f"Error: {msg}", ok=False)

    # -- handlers ----------------------------------------------------------------

    def list_nodes(self, prefix: str | None = None) -> str:
        nodes = self.ws.nodes(prefix)
        if not nodes:
            return "No nodes" + (f" under '{prefix}'." if prefix else " yet.")
        lines = []
        for n in nodes:
            deps = f"  <- {', '.join(n['depends_on'])}" if n["depends_on"] else ""
            lines.append(f"{n['name']}  [{n['model_type']}]  {n['path']}{deps}")
        return "\n".join(lines)

    def create_chat_node(self, name: str, request: str, folder: str = "", chat_instruction: str | None = None) -> str:
        path = self.ws.add_chat_node(name, request, chat_instruction=chat_instruction, folder=folder)
        return f"Created chat node '{name}' ({path.relative_to(self.ws.root)})."

    def create_coding_agent_node(self, name: str, task: str, folder: str = "",
                                 inherit_coding_folder: str | None = None) -> str:
        path = self.ws.add_coding_agent_node(name, task, inherit_coding_folder=inherit_coding_folder or None,
                                             folder=folder)
        return (f"Created coding_agent node '{name}' ({path.relative_to(self.ws.root)}); "
                f"its code goes to {self.ws.code_dir} when it runs.")

    def set_node_prompt(self, name: str, prompt: str) -> str:
        self.ws.set_body(name, prompt)
        return f"Updated '{name}'.\n{self.ws.source(name)}"

    def rename_node(self, old: str, new: str) -> str:
        self.ws.rename(old, new)
        return f"Renamed '{old}' to '{new}'."

    def connect_nodes(self, upstream: str, downstream: str, as_folder: bool = False, label: str | None = None) -> str:
        added = self.ws.connect(upstream, downstream, as_folder=as_folder, label=label)
        return f"Connected {upstream} -> {downstream}: added {added!r}."

    def list_jinja_functions(self) -> str:
        return "\n".join(f"{name}: {help_}" for name, help_ in self.ws.jinja_functions().items())

    def add_jinja(self, name: str, jinja: str, position: str = "end") -> str:
        self.ws.insert_jinja(name, jinja, position)
        return f"Added to '{name}'.\n{self.ws.source(name)}"

    def run_nodes(self, names: list[str] | None = None) -> str:
        import promptmap

        results = self._run(list(names or []) or None)
        lines = []
        for node, value in results.items():
            if isinstance(value, promptmap.ModelError):
                lines.append(f"{node}: ERROR {value.message}")
            elif value is promptmap.ModelStatus.SKIPPED:
                lines.append(f"{node}: skipped (an upstream node failed)")
            else:
                if not isinstance(value, str) or value.startswith("$promptmap:"):  # files: summary, not the tree
                    value = self.ws.output(node) or value
                lines.append(f"{node}: ok — {self._preview(value)}")
        return "\n".join(lines) or "Nothing to run."

    def chat_with_node(self, name: str, message: str) -> str:
        import promptmap

        value = self._run([name], chat_input={name: message}).get(name)
        if isinstance(value, promptmap.ModelError):
            return f"Error: {value.message}"
        return self._clip(str(value))

    def read_node_output(self, name: str, max_chars: int | None = None) -> str:
        out = self.ws.output(name)
        if out is None:
            return f"'{name}' has no successful output yet; run it with run_nodes."
        return self._clip(out, max_chars)

    def read_code_dir(self, path: str = ".", max_chars: int | None = None) -> str:
        base = self.ws.code_dir
        target = (base / path).resolve()
        if target != base and base not in target.parents:
            return "Error: path is outside the code directory."
        if not target.exists():
            return f"Error: {path} does not exist in the code directory."
        if target.is_file():
            try:
                return self._clip(target.read_text(encoding="utf-8"), max_chars)
            except UnicodeDecodeError:
                return f"{path}: binary file, {target.stat().st_size} bytes."
        files = []
        for p in sorted(_walk(target)):
            files.append(p.relative_to(base).as_posix())
            if len(files) >= _MAX_LIST:
                files.append(f"... (more than {_MAX_LIST} files; list a subfolder)")
                break
        return "\n".join(files) or "(empty)"

    # -- helpers -------------------------------------------------------------

    def _run(self, select: list[str] | None, **kwargs: Any) -> dict[str, Any]:
        """``Workspace.run``, reporting node starts, agent steps and finishes to :attr:`progress`."""
        if self.progress is None:
            return self.ws.run(select, **kwargs)
        from promptmap.model_types import progress_listener

        report = self.progress

        def done(result) -> None:
            detail = result.status + (" (cached)" if getattr(result, "cached", False) else "")
            if result.status == "error" and result.error:
                detail += f": {' '.join(str(result.error).split())[:120]}"
            report("done", result.model_name, detail)

        token = progress_listener.set(
            lambda node, step, total: report("step", node, f"{step}/{total}" if total else str(step)))
        try:
            return self.ws.run(select, on_model_start=lambda node: report("start", node, ""),
                               on_model_done=done, **kwargs)
        finally:
            progress_listener.reset(token)

    def _clip(self, text: str, max_chars: int | None = None) -> str:
        limit = max_chars or self.max_chars
        return text if len(text) <= limit else text[:limit] + f"\n... [{len(text) - limit} more chars]"

    def _preview(self, value: Any) -> str:
        text = value if isinstance(value, str) else json.dumps(value, default=str)
        text = " ".join(text.split())
        return text[:200] + ("…" if len(text) > 200 else "")


def _walk(root: Path):
    for p in root.iterdir():
        if p.is_dir():
            if p.name not in _SKIP_DIRS:
                yield from _walk(p)
        else:
            yield p
