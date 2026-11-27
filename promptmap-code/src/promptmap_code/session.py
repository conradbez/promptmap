"""
pm-code's model of a project: chat nodes you talk to, and the nodes their final answers create.

A chat's **final** reply (its latest one, when you say ``/done``) is read for
headings that name the nodes to make next::

    ### coding_agent: build_cli
    Create todo.py with add/list/done commands.

    ### chat: storage
    Work out how tasks are stored.

:meth:`Session.spawn` creates each one in the chat's own folder
(``<chat's folder>/<chat>/``) with ``{{ ref('<chat>') }}``, so it reads the
chat's whole answer.  At most :data:`MAX_CODING_AGENTS` coding agents and
:data:`MAX_CHATS` chats per chat.

Every node pm-code writes carries ``{{ org_outline(up=0, used_by=False) }}``: a bullet map
of where it sits — itself and the nodes it was built from.  ``up=0`` keeps
consumers out, so making a chat's children does not change the chat's own
prompt (which would regenerate its reply).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Callable

from promptmap.workspace import Workspace, WorkspaceError

MAX_CODING_AGENTS = 5
MAX_CHATS = 5

#: The map line added to every node pm-code writes.
OUTLINE = "Where you are in the pipeline:\n{{ org_outline(up=0, used_by=False) }}"

#: Appended to every chat's request: how to end the conversation.
FINISH_GUIDE = """\
## How to finish
You are one chat in a pipeline that builds software. Work out your part with the
user. Once it is clear, end your answer with the nodes to create next, each under
a heading of its own:

### coding_agent: <name>
<one self-contained coding task: what to build or change, file paths from the project root>

### chat: <name>
<a part that needs its own discussion: what that chat should work out>

Give 1 to 5 coding_agent headings, and chat headings only where a part is still
open. Names use letters, digits, '_' and '-'. Every new node also receives your
whole answer, so do not repeat shared context in each one."""

_HEADING = re.compile(r"^#{2,4}[ \t]*(coding_agent|chat)[ \t]*:[ \t]*`?([A-Za-z0-9_-]+)`?[ \t]*$",
                      re.IGNORECASE | re.MULTILINE)


@dataclass
class Child:
    kind: str  # "coding_agent" or "chat"
    name: str
    text: str


@dataclass
class Plan:
    children: list[Child]
    #: Headings left out: over the limit, or a name used twice.
    dropped: list[str]


def parse_plan(reply: str) -> Plan:
    """The ``### coding_agent: name`` / ``### chat: name`` sections of *reply*."""
    matches = list(_HEADING.finditer(reply or ""))
    children: list[Child] = []
    dropped: list[str] = []
    seen: set[str] = set()
    counts = {"coding_agent": 0, "chat": 0}
    limits = {"coding_agent": MAX_CODING_AGENTS, "chat": MAX_CHATS}
    for i, m in enumerate(matches):
        kind, name = m.group(1).lower(), m.group(2)
        end = matches[i + 1].start() if i + 1 < len(matches) else len(reply)
        text = reply[m.end():end].strip()
        if name in seen:
            dropped.append(f"{kind} {name}: name used twice")
            continue
        if counts[kind] >= limits[kind]:
            dropped.append(f"{kind} {name}: more than {limits[kind]}")
            continue
        if not text:
            dropped.append(f"{kind} {name}: no text under the heading")
            continue
        seen.add(name)
        counts[kind] += 1
        children.append(Child(kind, name, text))
    return Plan(children, dropped)


def chat_body(request: str, parent: str | None = None) -> str:
    """A chat node's text: the request, the map, the parent's answer, how to finish."""
    parts = [request.strip(), OUTLINE]
    if parent:
        parts.append(_ref_block(parent))
    parts.append(FINISH_GUIDE)
    return "\n\n".join(parts)


def coding_agent_body(task: str, parent: str) -> str:
    return "\n\n".join([task.strip(), OUTLINE, _ref_block(parent)])


def _ref_block(parent: str) -> str:
    return f"The chat '{parent}' that created this node concluded:\n{{{{ ref('{parent}') }}}}"


class Session:
    """Chats, their final answers, and the nodes those create, on a :class:`Workspace`."""

    def __init__(self, workspace: Workspace, progress: Callable[[str, str, str], None] | None = None,
                 **run_kwargs: Any) -> None:
        self.ws = workspace
        #: ``progress(event, node, detail)`` while nodes run: "start", "step", "done".
        self.progress = progress
        #: Extra ``promptmap.run`` arguments for every run (tests pass ``llm_call``).
        self.run_kwargs = run_kwargs

    # -- chats -------------------------------------------------------------

    def chats(self) -> list[dict]:
        return [n for n in self.ws.nodes() if n["model_type"] == "chat"]

    def is_chat(self, name: str) -> bool:
        return any(n["name"] == name for n in self.chats())

    def create_chat(self, name: str, request: str, parent: str | None = None) -> None:
        folder = self.child_folder(parent) if parent else ""
        self.ws.add_chat_node(name, chat_body(request, parent), folder=folder)

    def reply(self, name: str, message: str | None = None) -> str:
        """Send *message* to chat *name* (none: its first reply, or redo the last). Returns the reply."""
        import promptmap

        kwargs = {"chat_input": {name: message}} if message else {}
        value = self._run([name], **kwargs).get(name)
        if isinstance(value, promptmap.ModelError):
            raise WorkspaceError(value.message)
        if isinstance(value, promptmap.ModelStatus):
            raise WorkspaceError(f"'{name}' was {value.value}.")
        return self.ws.output(name) or str(value)

    def history(self, name: str) -> list[dict]:
        return self.ws.chat_history(name)

    def final(self, name: str) -> str | None:
        """The chat's latest reply: the one its children are made from."""
        return self.ws.output(name)

    # -- what a final answer creates --------------------------------------------

    def child_folder(self, chat: str) -> str:
        path = PurePosixPath(self._node(chat)["path"])
        parent = "" if str(path.parent) == "." else str(path.parent)
        return f"{parent}/{chat}" if parent else chat

    def children(self, chat: str) -> list[dict]:
        """Nodes this chat created: in its folder and ref()ing it."""
        folder = self.child_folder(chat) + "/"
        return [n for n in self.ws.nodes(folder)
                if chat in n["depends_on"] and n["path"].count("/") == folder.count("/")]

    def spawn(self, chat: str) -> tuple[list[tuple[Child, str]], Plan]:
        """Create (or update) the nodes named in the chat's final reply.

        Returns ``[(child, "created" | "updated" | "skipped: why")]`` and the plan.
        """
        reply = self.final(chat)
        if reply is None:
            raise WorkspaceError(f"'{chat}' has no reply yet.")
        plan = parse_plan(reply)
        mine = {n["name"]: n for n in self.children(chat)}
        existing = {n["name"]: n for n in self.ws.nodes()}
        folder = self.child_folder(chat)
        done: list[tuple[Child, str]] = []
        for child in plan.children:
            body = (coding_agent_body(child.text, chat) if child.kind == "coding_agent"
                    else chat_body(child.text, chat))
            try:
                if child.name in mine:
                    if mine[child.name]["model_type"] != child.kind:
                        done.append((child, f"skipped: it exists as a {mine[child.name]['model_type']}"))
                        continue
                    if self.ws.source(child.name).endswith(body.rstrip() + "\n"):
                        done.append((child, "unchanged"))
                        continue
                    self.ws.set_body(child.name, body)
                    done.append((child, "updated"))
                elif child.name in existing:
                    done.append((child, f"skipped: '{child.name}' already exists at {existing[child.name]['path']}"))
                elif child.kind == "coding_agent":
                    self.ws.add_coding_agent_node(child.name, body, folder=folder)
                    done.append((child, "created"))
                else:
                    self.ws.add_chat_node(child.name, body, folder=folder)
                    done.append((child, "created"))
            except WorkspaceError as exc:
                done.append((child, f"skipped: {exc}"))
        return done, plan

    def run(self, names: list[str]) -> dict[str, Any]:
        """Run *names* (and their upstream), reporting progress."""
        return self._run(names)

    # -- the map ----------------------------------------------------------------

    def outline(self, here: str | None = None) -> str:
        """The bullet map: the whole pipeline, with *here* marked ``[you are here]``."""
        from promptmap.org_outline import org_outline
        from promptmap.template_functions import TemplateContext

        c = self.ws.construct()
        if not c.models:
            return ""
        return org_outline(TemplateContext(model=here or "", specs=c.specs()))

    # -- helpers ------------------------------------------------------------------

    def _node(self, name: str) -> dict:
        for n in self.ws.nodes():
            if n["name"] == name:
                return n
        raise WorkspaceError(f"No node named '{name}'.")

    def _run(self, select: list[str], **kwargs: Any) -> dict[str, Any]:
        kwargs = {**self.run_kwargs, **kwargs}
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
