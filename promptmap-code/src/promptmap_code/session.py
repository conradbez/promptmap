"""
pm-code's model of a project: chats you refine, and the pm tools their final replies call.

A chat, in four parts
---------------------
0. **Instructions** (the system message): what a chat can and cannot do,
   and how to call pm tools.  Every pm-code chat has
   ``chat_instruction_file="PM_CODE.md"``, so they all read the project's
   ``PM_CODE.md`` on every run; pm-code writes :data:`DEFAULT_INSTRUCTIONS`
   there when the file is missing.  Edit it to change what every chat is
   told.  The ``.prompt`` file holds none of it.
1. **Original intent.** The chat's ``.prompt`` file is the first user message
   of the conversation: what the user set out to get, in their own words, as
   typed when the chat was created.  It is rendered fresh on every run (with
   its ``ref()``s), so changing what the chat reads — ``/connect`` — changes
   this message, never the history.
2. **Refinements.** Every follow-up the user types after that is a
   refinement of the intent.  Follow-ups and replies live in the run
   database (``chat_turns``), not in the file; the chat always answers with
   its complete, updated answer.
3. **pm tools.** Only the chat's **last** reply is evaluated, when the user
   says ``/done``.  Headings in it call pm tools (:data:`OUTPUT_TOOLS`)::

       ### coding_agent: build_cli
       Create todo.py with add, list and done commands.

       ### chat: storage
       Work out how tasks are stored.

       ### connect: storage -> build_cli
       ### model: build_cli -> anthropic/claude-sonnet-5-5

   New nodes go in the chat's own folder (``<chat's folder>/<chat>/``) with
   ``{{ ref('<chat>') }}``, so each reads the chat's whole answer.  At most
   :data:`MAX_CODING_AGENTS` coding agents and :data:`MAX_CHATS` chats.
   Earlier replies are never evaluated: a draft that named nodes creates none.

The map
-------
Every node pm-code writes carries ``{{ org_outline(up=0, used_by=False) }}``:
a bullet map of where it sits — itself and the nodes it was built from.
``up=0, used_by=False`` keeps consumers out, so creating a chat's children
does not change the chat's own prompt (which would regenerate its reply).
:meth:`Session.nearby` orders other nodes by distance in that map, for the
``/connect`` picker.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Callable

from promptmap.workspace import Workspace, WorkspaceError

MAX_CODING_AGENTS = 5
MAX_CHATS = 5

#: The tools a chat's last reply can call, as ``### <tool>: <argument>`` headings.
OUTPUT_TOOLS = {
    "coding_agent": "create (or update) a coding agent: '### coding_agent: <name>', the task under it",
    "chat": "create (or update) a chat: '### chat: <name>', what it should work out under it",
    "connect": "feed one node's output into another; between two coding agents the downstream one continues "
               "the upstream's code: '### connect: <upstream> -> <downstream>'",
    "model": "set a node's model: '### model: <node> -> <provider>/<model>'",
}

#: The map line added to every node pm-code writes.
OUTLINE = "Where you are in the pipeline:\n{{ org_outline(up=0, used_by=False) }}"

#: The project file every pm-code chat reads as its system message.
INSTRUCTIONS_FILE = "PM_CODE.md"

#: What PM_CODE.md starts as: what a chat can and cannot do, and the pm tools.
DEFAULT_INSTRUCTIONS = """\
# pm-code chat instructions

You are a chat in pm-code, a tool that builds software as a promptmap pipeline of
chats and coding agents. This file is your system message; the user can edit it.

## The conversation
- The first user message is the user's **original intent**, followed by a map of where
  you sit in the pipeline and the outputs of nodes you read.
- Every later user message **refines** that intent. Where they conflict, the later one wins.
- Only your **latest reply** is used. Always reply with the complete, updated answer,
  never just the changes.

## You can
- Discuss, plan and decide with the user; ask questions when the intent is unclear.
- Use what your first message contains: the intent, the map, connected nodes' outputs.
- Call pm tools, by ending your reply with the headings below. They run when the user
  types /done, and only the ones in your latest reply: repeat every heading you still want.

## You cannot
- Write, run or test code. Only coding agents change code.
- See the project's files, the code folder or coding agents' results, unless a node
  that holds them is connected to you.
- Browse the web or call APIs.
- Change your own inputs or model. The user does that with /connect and /model.

## pm tools

### coding_agent: <name>
<one self-contained coding task: what to build or change, file paths from the project root>

### chat: <name>
<a part that needs its own discussion: what that chat should work out>

### connect: <upstream node> -> <downstream node>

### model: <node> -> <provider>/<model>

- Give 1 to 5 coding_agent headings, chat headings only where a part is still open, and
  connect / model only when needed. Names use letters, digits, '_' and '-'.
- Every node you create receives your whole latest answer, so do not repeat shared
  context in each task.
- A coding agent works on a scratch copy and its files are written into the project when
  it finishes. One connected to another coding agent continues that agent's code; one
  connected to none starts from an empty folder. Unconnected coding agents run at the same
  time: give them separate folders.
- A coding agent continues at most one other, so chain dependent steps (a -> b -> c).
- model sets the LLM that runs a node (for a coding agent: the agent writing the code).
  It is not a model the software you are designing calls; put that in the task text.
- You cannot connect into, or set the model of, yourself.
"""

#: The "## How to finish" guide pm-code 0.2.0-0.2.2 wrote into chats' .prompt files.
_OLD_GUIDE = re.compile(r"\n*## How to finish\nYou are one chat in a pipeline.*?shared context in each one\.\n?",
                        re.DOTALL)


def ensure_instructions(root) -> bool:
    """Write :data:`DEFAULT_INSTRUCTIONS` to ``<root>/PM_CODE.md`` if missing; True when written."""
    from pathlib import Path

    path = Path(root) / INSTRUCTIONS_FILE
    if path.exists():
        return False
    path.write_text(DEFAULT_INSTRUCTIONS, encoding="utf-8")
    return True


_HEADING = re.compile(r"^#{2,4}[ \t]*(coding_agent|chat|connect|model)[ \t]*:[ \t]*(.+?)[ \t]*$",
                      re.IGNORECASE | re.MULTILINE)
_NAME = re.compile(r"^`?([A-Za-z0-9_-]+)`?$")
#: A ref() block ``Workspace.connect`` appends: "Input from x:\n{{ ref('x') }}".
_CONNECTED = re.compile(r"^Input from [^\n]*:\n\{\{ ref\('[^']+'\) \}\}$", re.MULTILINE)
_ARROW = re.compile(r"^`?([A-Za-z0-9_-]+)`?\s*->\s*`?([^`\s]+)`?$")


@dataclass
class Call:
    """One pm tool call from a chat's last reply."""

    tool: str  # a key of OUTPUT_TOOLS
    name: str  # the node created / connected from / given a model
    text: str = ""  # coding_agent task or chat request
    target: str = ""  # connect: the downstream node; model: "provider/model"

    @property
    def label(self) -> str:
        return f"{self.tool} {self.name}" + (f" -> {self.target}" if self.target else "")


@dataclass
class Plan:
    calls: list[Call]
    #: Headings left out, with why: over the limit, a name used twice, malformed.
    dropped: list[str]


def parse_plan(reply: str) -> Plan:
    """The pm tool calls (``### tool: argument`` headings) in *reply*."""
    matches = list(_HEADING.finditer(reply or ""))
    calls: list[Call] = []
    dropped: list[str] = []
    created: set[str] = set()
    counts = {"coding_agent": 0, "chat": 0}
    limits = {"coding_agent": MAX_CODING_AGENTS, "chat": MAX_CHATS}
    for i, m in enumerate(matches):
        tool, arg = m.group(1).lower(), m.group(2)
        end = matches[i + 1].start() if i + 1 < len(matches) else len(reply)
        text = reply[m.end():end].strip()
        if tool in counts:
            name = _NAME.match(arg)
            if not name:
                dropped.append(f"{tool} {arg}: not a node name")
            elif name.group(1) in created:
                dropped.append(f"{tool} {name.group(1)}: name used twice")
            elif counts[tool] >= limits[tool]:
                dropped.append(f"{tool} {name.group(1)}: more than {limits[tool]}")
            elif not text:
                dropped.append(f"{tool} {name.group(1)}: no text under the heading")
            else:
                created.add(name.group(1))
                counts[tool] += 1
                calls.append(Call(tool, name.group(1), text))
        else:
            pair = _ARROW.match(arg)
            if pair:
                calls.append(Call(tool, pair.group(1), target=pair.group(2)))
            else:
                dropped.append(f"{tool} {arg}: expected '<node> -> <{'node' if tool == 'connect' else 'provider/model'}>'")
    return Plan(calls, dropped)


def chat_body(intent: str, parent: str | None = None) -> str:
    """A chat's ``.prompt`` text: the original intent, the map and the parent's answer.

    The instructions are not here: they are the system message, from PM_CODE.md.
    """
    parts = [intent.strip(), OUTLINE]
    if parent:
        parts.append(_ref_block(parent))
    return "\n\n".join(parts)


def coding_agent_body(task: str, parent: str) -> str:
    return "\n\n".join([task.strip(), OUTLINE, _ref_block(parent)])


def _ref_block(parent: str) -> str:
    return f"The chat '{parent}' that created this node concluded:\n{{{{ ref('{parent}') }}}}"


class Session:
    """Chats, their refinements, and the pm tools their last replies call, on a :class:`Workspace`."""

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

    def create_chat(self, name: str, intent: str, parent: str | None = None) -> None:
        """A new chat whose ``.prompt`` file starts with *intent*: the user's original intent."""
        folder = self.child_folder(parent) if parent else ""
        self._add_chat(name, chat_body(intent, parent), folder)

    def _add_chat(self, name: str, body: str, folder: str) -> None:
        ensure_instructions(self.ws.root)
        self.ws.add_node(name, body, {"model_type": "chat", "chat_instruction_file": INSTRUCTIONS_FILE}, folder)

    def instructions_path(self):
        return self.ws.root / INSTRUCTIONS_FILE

    def sync_chat(self, name: str) -> bool:
        """Bring a chat pm-code wrote up to date: PM_CODE.md as its system message, no guide in its text.

        Chats from pm-code 0.2.0-0.2.2 carry a "## How to finish" guide in their
        .prompt file and no ``chat_instruction_file``.  Chats pm-code did not
        write (no map line in their text) are left alone.  True when changed.
        """
        from promptmap.workspace import split_config

        body = split_config(self.ws.source(name))[1]
        if OUTLINE not in body:
            return False
        changed = False
        stripped = _OLD_GUIDE.sub("\n", body).rstrip() + "\n"
        if stripped != body.rstrip() + "\n":
            self.ws.set_body(name, stripped)
            changed = True
        config = self.ws.construct().get_config(name)
        if config.get("chat_instruction_file") != INSTRUCTIONS_FILE:
            ensure_instructions(self.ws.root)
            self.ws.set_config(name, chat_instruction_file=INSTRUCTIONS_FILE, chat_instruction=None)
            changed = True
        return changed

    def sync_all(self) -> list[str]:
        """:meth:`sync_chat` every chat; the names of those changed."""
        return [n["name"] for n in self.chats() if self.sync_chat(n["name"])]

    def reply(self, name: str, refinement: str | None = None) -> str:
        """Send a *refinement* of the intent to chat *name*; returns its new, complete answer.

        With none, the chat answers its original intent (first run) or redoes
        its last reply — e.g. after ``/connect`` or ``/model`` changed what it reads.
        """
        import promptmap

        ensure_instructions(self.ws.root)
        kwargs = {"chat_input": {name: refinement}} if refinement else {}
        value = self._run([name], **kwargs).get(name)
        if isinstance(value, promptmap.ModelError):
            raise WorkspaceError(value.message)
        if isinstance(value, promptmap.ModelStatus):
            raise WorkspaceError(f"'{name}' was {value.value}.")
        return self.ws.output(name) or str(value)

    def intent(self, name: str) -> str:
        """The chat's original intent: its ``.prompt`` text, before the map and the finish guide."""
        from promptmap.workspace import split_config

        body = split_config(self.ws.source(name))[1]
        return body.split("\n\n" + OUTLINE, 1)[0].strip()

    def history(self, name: str) -> list[dict]:
        return self.ws.chat_history(name)

    def final(self, name: str) -> str | None:
        """The chat's last reply: the only one whose pm tool calls are evaluated."""
        return self.ws.output(name)

    # -- slash tools ----------------------------------------------------------

    def connect_into(self, chat: str, upstream: str) -> str:
        """Make *chat* read *upstream*: a ref() added after its intent. Returns what was added."""
        self.sync_chat(chat)
        return self.ws.connect(upstream, chat)

    def set_model(self, name: str, provider: str, model: str) -> None:
        """Give node *name* its own model; empty values go back to the project's.

        Coding agents read ``agent_model="provider/model"``; chats and other
        LLM nodes read ``provider=`` / ``model=``.
        """
        if self._node(name)["model_type"] == "coding_agent":
            self.ws.set_config(name, agent_model=f"{provider}/{model}" if provider and model else None)
        else:
            self.ws.set_config(name, provider=provider or None, model=model or None)

    def model_of(self, name: str) -> str:
        """The node's own ``provider/model`` from its config, or "" for the project default."""
        config = self.ws.construct().get_config(name)
        if config.get("agent_model"):
            return str(config["agent_model"])
        provider, model = config.get("provider") or "", config.get("model") or ""
        return f"{provider}/{model}" if provider and model else model

    def path(self, name: str) -> str:
        """The node's file, relative to the project root."""
        return f"{self.ws.map_dir}/{self._node(name)['path']}"

    def nearby(self, name: str) -> list[str]:
        """Nodes *name* can ``ref()``, nearest first in the bullet map.

        Distance is the number of ref() edges between the two nodes, either
        direction; ties keep the order the full outline lists them in.  Nodes
        not connected to *name* come last.  *name* itself, and nodes built on
        it (a ref to one would make a cycle), are left out.
        """
        import networkx as nx

        from promptmap.template_functions import dag_from_specs

        c = self.ws.construct()
        dag = dag_from_specs(c.specs())
        if name not in dag:
            return sorted(dag.nodes)
        distance = nx.single_source_shortest_path_length(dag.to_undirected(), name)
        outline_order = {n: i for i, n in enumerate(_outline_names(self.outline()))}
        blocked = nx.descendants(dag, name) | {name}
        candidates = [n for n in dag.nodes if n not in blocked]
        return sorted(candidates, key=lambda n: (distance.get(n, len(dag) + 1), outline_order.get(n, len(dag)), n))

    # -- what the last reply calls -----------------------------------------------

    def child_folder(self, chat: str) -> str:
        path = PurePosixPath(self._node(chat)["path"])
        parent = "" if str(path.parent) == "." else str(path.parent)
        return f"{parent}/{chat}" if parent else chat

    def children(self, chat: str) -> list[dict]:
        """Nodes this chat created: in its folder and ref()ing it."""
        folder = self.child_folder(chat) + "/"
        return [n for n in self.ws.nodes(folder)
                if chat in n["depends_on"] and n["path"].count("/") == folder.count("/")]

    def pending(self, chat: str) -> Plan:
        """The pm tool calls in the chat's last reply, not yet applied."""
        return parse_plan(self.final(chat) or "")

    def apply(self, chat: str) -> tuple[list[tuple[Call, str]], Plan]:
        """Call the pm tools named in the chat's last reply.

        Nodes are created first, so connect / model can name them.  Returns
        ``[(call, "created" | "updated" | "unchanged" | "connected" | "set" | "skipped: why")]``.
        """
        reply = self.final(chat)
        if reply is None:
            raise WorkspaceError(f"'{chat}' has no reply yet.")
        plan = parse_plan(reply)
        order = {"coding_agent": 0, "chat": 0, "connect": 1, "model": 2}
        done = []
        for call in sorted(plan.calls, key=lambda c: order[c.tool]):
            try:
                done.append((call, self._apply_one(chat, call)))
            except WorkspaceError as exc:
                done.append((call, f"skipped: {exc}"))
        return done, plan

    def _apply_one(self, chat: str, call: Call) -> str:
        if call.tool == "connect":
            if call.target == chat:
                raise WorkspaceError(f"a chat's reply cannot change the chat itself; use /connect in '{chat}'.")
            up, down = self._node(call.name), self._node(call.target)
            if up["model_type"] == down["model_type"] == "coding_agent":
                # Code builds on code: the downstream agent starts from the upstream's tree.
                inherits = self.ws.construct().get_config(call.target).get("inherit_coding_folder")
                if inherits == call.name:
                    return "unchanged"
                if inherits:
                    raise WorkspaceError(f"'{call.target}' already continues '{inherits}'s code; "
                                         "a coding agent continues one other.")
                self.ws.connect(call.name, call.target, as_folder=True)
                return "continues code"
            if call.name in down["depends_on"]:
                return "unchanged"
            self.ws.connect(call.name, call.target)
            return "connected"
        if call.tool == "model":
            if call.name == chat:
                raise WorkspaceError(f"a chat's reply cannot change the chat itself; use /model in '{chat}'.")
            provider, _, model = call.target.partition("/")
            if not model:
                raise WorkspaceError(f"'{call.target}' is not <provider>/<model>.")
            self._node(call.name)
            if self.model_of(call.name) == call.target:
                return "unchanged"
            self.set_model(call.name, provider, model)
            return "set"

        body = coding_agent_body(call.text, chat) if call.tool == "coding_agent" else chat_body(call.text, chat)
        mine = {n["name"]: n for n in self.children(chat)}
        if call.name in mine:
            if mine[call.name]["model_type"] != call.tool:
                raise WorkspaceError(f"it exists as a {mine[call.name]['model_type']}")
            # Keep the inputs connected since (connect / /connect): they follow the body.
            body = "\n\n".join([body, *_CONNECTED.findall(self.ws.source(call.name))])
            if self.ws.source(call.name).endswith(body.rstrip() + "\n"):
                return "unchanged"
            self.ws.set_body(call.name, body)
            if call.tool == "chat":
                self.sync_chat(call.name)
            return "updated"
        existing = {n["name"]: n for n in self.ws.nodes()}
        if call.name in existing:
            raise WorkspaceError(f"'{call.name}' already exists at {existing[call.name]['path']}")
        folder = self.child_folder(chat)
        if call.tool == "coding_agent":
            self.ws.add_coding_agent_node(call.name, body, folder=folder)
        else:
            self._add_chat(call.name, body, folder)
        return "created"

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


def _outline_names(outline: str) -> list[str]:
    """Node names in the order an outline lists them."""
    names = []
    for line in outline.splitlines():
        token = line.strip().removeprefix("- ").split(" ", 1)[0]
        if token and token not in names:
            names.append(token)
    return names
