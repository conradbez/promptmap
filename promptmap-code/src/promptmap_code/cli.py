"""
``pm-code`` — pick a chat, talk it through, and let its final answer create the next nodes.

::

    pm-code              # pick a chat (or name a new one)
    pm-code spec         # go straight into chat 'spec' (created if new)

A chat's ``.prompt`` file is the user's **original intent**: the first
message of the conversation.  Each line typed in the chat is a
**refinement**.  A line starting with ``/`` is a tool (typing ``/`` drops
them down): ``/connect`` picks another node for the chat to read (nearest in
the map first), ``/model`` changes the chat's model, ``/done`` calls the pm
tools named in the chat's **last** reply (create coding agents and chats,
connect nodes, set models; see :mod:`promptmap_code.session`).

The project is the current directory (``--root``): nodes go in ``map/``, runs in
``.promptmap/``, and coding_agent nodes write code into ``--code-dir``.
"""

from __future__ import annotations

import json  # noqa: F401  (used by missing_api_keys)
import os
import sys
from pathlib import Path
from typing import Callable

import click
from rich.console import Console
from rich.markup import escape

console = Console(highlight=False, soft_wrap=True)


def resolve_model_name(root: Path, map_dir: str, model: str | None) -> str | None:
    """--model, else the project's settings.yml, else ``PM_CODE_MODEL`` / ``MSWEA_MODEL_NAME``."""
    if model:
        return model
    from promptmap.llm import litellm_model
    from promptmap.settings import load_settings, resolve_model

    try:
        target = resolve_model(load_settings(str(root / map_dir)))
    except Exception:  # noqa: BLE001 — a broken settings.yml just means "not from there"
        target = None
    if target:
        return litellm_model(*target)
    return os.getenv("PM_CODE_MODEL") or os.getenv("MSWEA_MODEL_NAME") or None


def load_env(root: Path) -> None:
    """Load the project's .env, as ``promptmap run`` does; variables already set win."""
    from dotenv import load_dotenv

    load_dotenv(root / ".env", override=False)


def missing_api_keys(model_name: str) -> list[str]:
    """Environment variables *model_name* needs that are not set (best effort).

    litellm's own check does not know its OpenAI-compatible providers (Sarvam,
    ...), whose key name is in its ``openai_like/providers.json``; both are read.
    Without this a missing key surfaces as a server error that is retried for
    a minute.
    """
    import litellm

    missing: list[str] = []
    try:
        missing += litellm.validate_environment(model=model_name).get("missing_keys") or []
    except Exception:  # noqa: BLE001 — unknown provider: litellm will say so itself
        pass
    provider = model_name.split("/", 1)[0] if "/" in model_name else ""
    try:
        table = json.loads((Path(litellm.__file__).parent / "llms" / "openai_like" / "providers.json").read_text())
        env = (table.get(provider) or {}).get("api_key_env")
        if env and not os.getenv(env) and env not in missing:
            missing.append(env)
    except (OSError, ValueError, AttributeError):
        pass
    return missing


def ensure_project(root: Path, map_dir: str, model_name: str) -> list[str]:
    """Create map/ and, when there is no LLM backend, a settings.yml for *model_name*."""
    notes = []
    (root / map_dir).mkdir(parents=True, exist_ok=True)
    from promptmap.llm import has_llm_backend

    if not has_llm_backend(str(root / map_dir)) and "/" in model_name:
        provider, _, model = model_name.partition("/")
        (root / "settings.yml").write_text(f"provider: {provider}\nmodel: {model}\n", encoding="utf-8")
        notes.append(f"wrote settings.yml (provider: {provider}, model: {model}) for the pipeline's nodes")
    return notes


class NodeProgress:
    """Prints node starts and finishes; a spinner shows the running nodes' agent steps."""

    def __init__(self) -> None:
        import threading

        self.running: dict[str, str] = {}
        self.status = None
        self.lock = threading.Lock()

    def __call__(self, event: str, node: str, detail: str) -> None:
        with self.lock:
            if event == "start":
                self.running[node] = ""
                console.print(f"  [dim]⎿[/dim] [yellow]▶ {escape(node)}[/yellow]")
            elif event == "step":
                self.running[node] = f"step {detail}"
            elif event == "done":
                self.running.pop(node, None)
                colour = "green" if detail.startswith("success") else "yellow" if detail.startswith("skipped") else "red"
                console.print(f"  [dim]⎿[/dim] [{colour}]■ {escape(node)}[/{colour}] [dim]{escape(detail)}[/dim]")
            self._refresh()

    def _refresh(self) -> None:
        if not self.running:
            if self.status is not None:
                self.status.stop()
                self.status = None
            return
        text = "running " + ", ".join(f"{n} {s}".strip() for n, s in self.running.items())
        if self.status is None:
            self.status = console.status(escape(text))
            self.status.start()
        else:
            self.status.update(escape(text))




def _print_reply(text: str) -> None:
    console.print()
    console.print(escape(text))
    console.print()


def _ask(label: str) -> str | None:
    """A line from the terminal; None on Ctrl-D / Ctrl-C."""
    try:
        return click.prompt(label, prompt_suffix="> ", default="", show_default=False).strip()
    except (EOFError, click.Abort):
        return None


def _live_terminal() -> bool:
    from promptmap.settings import _live_terminal as live

    return live()


def _ask_follow_up(chat: str) -> str | None:
    """A refinement or a /tool; in a real terminal, typing '/' drops down the tools."""
    if not _live_terminal():
        return _ask(chat)
    from prompt_toolkit import prompt
    from prompt_toolkit.completion import Completer, Completion

    class Tools(Completer):
        def get_completions(self, document, complete_event):
            text = document.text_before_cursor
            if not text.startswith("/") or " " in text:
                return
            for name, (help_, _) in SLASH.items():
                if name.startswith(text[1:]):
                    yield Completion("/" + name, start_position=-len(text), display_meta=help_)

    try:
        return prompt(f"{chat}> ", completer=Tools(), complete_while_typing=True).strip()
    except (EOFError, KeyboardInterrupt):
        return None


def _choose(label: str, options: list[str], allow_custom: bool = False) -> str:
    """Searchable pick (promptmap's own picker): "" when skipped."""
    from promptmap.settings import choose

    try:
        return choose(label, options, allow_custom=allow_custom)
    except click.Abort:
        return ""


def _print_chats(session) -> list[str]:
    chats = session.chats()
    if not chats:
        console.print("[dim]No chats yet.[/dim]")
        return []
    console.print("[bold]Chats[/bold]")
    for i, n in enumerate(chats, 1):
        depth = n["path"].count("/")
        kids = session.children(n["name"])
        note = f" [dim]→ {len(kids)} node{'s' if len(kids) != 1 else ''}[/dim]" if kids else ""
        console.print(f"  {'  ' * depth}{i}. [cyan]{escape(n['name'])}[/cyan] [dim]{escape(n['path'])}[/dim]{note}")
    return [n["name"] for n in chats]


def select_chat(session) -> str | None:
    """The chat list; returns the picked or newly created chat, or None to quit."""
    names = _print_chats(session)
    label = "Chat number or name (new name starts a chat; empty quits)" if names else "Name a new chat (empty quits)"
    while True:
        choice = _ask(label)
        if not choice:
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(names):
            return names[int(choice) - 1]
        if choice in names:
            return choice
        if new_chat(session, choice):
            return choice


def new_chat(session, name: str) -> bool:
    """Ask for the chat's original intent: it becomes the chat's .prompt file (its first message)."""
    from promptmap.workspace import WorkspaceError

    intent = _ask(f"What do you want from '{name}'")
    if not intent:
        return False
    try:
        session.create_chat(name, intent)
    except WorkspaceError as exc:
        console.print(f"[red]Error:[/red] {escape(str(exc))}")
        return False
    console.print(f"[green]created chat '{escape(name)}'[/green]")
    return True


def run_nodes(session, names: list[str]) -> None:
    import promptmap

    try:
        results = session.run(names)
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted.[/yellow]")
        return
    except Exception as exc:  # noqa: BLE001 — shown, the session continues
        console.print(f"[red]Error:[/red] {escape(str(exc))}")
        return
    for name in names:
        value = results.get(name)
        if isinstance(value, promptmap.ModelError):
            console.print(f"[red]{escape(name)}: {escape(value.message)}[/red]")
        else:
            out = session.ws.output(name)
            if out:
                console.print(f"[bold]{escape(name)}[/bold]\n{escape(out)}")


# -- slash tools: each takes (session, chat, argument) -------------------------------

def _send(session, chat: str, refinement: str | None) -> None:
    try:
        _print_reply(session.reply(chat, refinement))
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted.[/yellow]")
        return
    except Exception as exc:  # noqa: BLE001 — shown, the session continues
        console.print(f"[red]Error:[/red] {escape(str(exc))}")
        return
    pending = session.pending(chat).calls
    if pending:
        console.print(f"[dim]{len(pending)} pm tool call{'s' if len(pending) > 1 else ''} in this reply "
                      f"({escape(', '.join(c.label for c in pending))}). /done applies them.[/dim]")


def tool_done(session, chat: str, arg: str) -> None:
    from promptmap.workspace import WorkspaceError

    try:
        done, plan = session.apply(chat)
    except WorkspaceError as exc:
        console.print(f"[red]Error:[/red] {escape(str(exc))}")
        return
    if not done:
        console.print("[yellow]The last reply calls no pm tools.[/yellow] Ask the chat to finish with "
                      "'### coding_agent: <name>' headings (and '### chat: <name>' for open parts).")
    for call, status in done:
        colour = "dim" if status == "unchanged" else "yellow" if status.startswith("skipped") else "green"
        console.print(f"  [{colour}]{escape(status)}[/{colour}] {escape(call.label)}")
    for note in plan.dropped:
        console.print(f"  [yellow]left out[/yellow] {escape(note)}")
    agents = [c.name for c, s in done if c.tool == "coding_agent" and s in ("created", "updated")]
    if agents and click.confirm(f"Run {len(agents)} coding agent{'s' if len(agents) > 1 else ''} now?",
                                default=True):
        run_nodes(session, agents)


def tool_run(session, chat: str, arg: str) -> None:
    agents = [n["name"] for n in session.children(chat) if n["model_type"] == "coding_agent"]
    if agents:
        run_nodes(session, agents)
    else:
        console.print("[dim]No coding agents yet: /done creates them from the last reply.[/dim]")


def tool_connect(session, chat: str, arg: str) -> None:
    """Pick a node (nearest in the map first); the chat's intent gets a ref() to it."""
    from promptmap.workspace import WorkspaceError

    options = session.nearby(chat)
    if not options:
        console.print("[dim]No other nodes to connect.[/dim]")
        return
    node = arg if arg in options else _choose("Node", options)
    if not node:
        return
    try:
        added = session.connect_into(chat, node)
    except WorkspaceError as exc:
        console.print(f"[red]Error:[/red] {escape(str(exc))}")
        return
    console.print(f"[green]'{escape(chat)}' now reads '{escape(node)}'[/green] [dim]({escape(added.splitlines()[-1])})[/dim]")
    console.print("[dim]Your next refinement uses it; /redo answers again now.[/dim]")


def tool_model(session, chat: str, arg: str) -> None:
    """Set the chat's own model (config provider= / model=); 'default' goes back to settings.yml's."""
    from promptmap.settings import ask_model, ask_provider

    current = session.model_of(chat) or "project default"
    if arg == "default":
        session.set_model(chat, "", "")
        console.print(f"[green]'{escape(chat)}' uses the project's model[/green]")
        return
    if "/" in arg:
        provider, _, model = arg.partition("/")
    else:
        console.print(f"[dim]Now: {escape(current)}. Empty skips.[/dim]")
        try:
            provider = ask_provider()
            model = ask_model(provider) if provider else ""
        except click.Abort:
            return
    if not provider or not model:
        return
    session.set_model(chat, provider, model)
    console.print(f"[green]'{escape(chat)}' now uses {escape(provider)}/{escape(model)}[/green] "
                  "[dim]Your next refinement uses it; /redo answers again now.[/dim]")


def tool_redo(session, chat: str, arg: str) -> None:
    _send(session, chat, None)


def tool_map(session, chat: str, arg: str) -> None:
    console.print(escape(session.outline(chat)))


def tool_intent(session, chat: str, arg: str) -> None:
    console.print(f"[bold]Original intent[/bold] [dim]({escape(session.path(chat))})[/dim]\n{escape(session.intent(chat))}")


def tool_history(session, chat: str, arg: str) -> None:
    console.print(f"[bold]intent:[/bold] {escape(session.intent(chat))}\n")
    for turn in session.history(chat):
        role = "refinement" if turn["role"] == "user" else "reply"
        console.print(f"[bold]{role}:[/bold] {escape(turn['content'])}\n")


def tool_instructions(session, chat: str, arg: str) -> None:
    path = session.instructions_path()
    text = path.read_text(encoding="utf-8") if path.is_file() else "(missing: pm-code writes it on the next run)"
    console.print(f"[bold]System message for every chat[/bold] [dim]({escape(str(path))}; edit it to change "
                  f"what chats are told)[/dim]\n{escape(text)}")


def tool_help(session, chat: str, arg: str) -> None:
    console.print("[dim]Type a refinement of the chat's intent, or a tool:[/dim]")
    for name, (help_, _) in SLASH.items():
        console.print(f"  [cyan]/{name}[/cyan] [dim]{escape(help_)}[/dim]")
    console.print("  [dim]empty line: back to the chat list · Ctrl-D: quit[/dim]")


#: Slash tools, in the order the dropdown and /help show them.
SLASH: dict[str, tuple[str, Callable]] = {
    "done": ("call the pm tools in the last reply (create nodes, connect, set models)", tool_done),
    "connect": ("make this chat read another node: pick one, nearest first", tool_connect),
    "model": ("change this chat's model ('/model default' resets it)", tool_model),
    "run": ("run this chat's coding agents", tool_run),
    "redo": ("answer again, e.g. after /connect or /model", tool_redo),
    "map": ("where this chat sits in the pipeline", tool_map),
    "intent": ("show the original intent (the .prompt file)", tool_intent),
    "history": ("the intent, refinements and replies", tool_history),
    "instructions": ("show the system message every chat gets (PM_CODE.md)", tool_instructions),
    "back": ("back to the chat list", None),
    "help": ("list the tools", tool_help),
}


def resolve_tool(text: str) -> tuple[str | None, str]:
    """('/con nodeA') -> ('connect', 'nodeA'): a tool name, a unique prefix or a pick from the list."""
    from promptmap.settings import filter_options

    word, _, arg = text[1:].strip().partition(" ")
    if word in SLASH:
        return word, arg.strip()
    matches = [n for n in SLASH if n.startswith(word)] if word else []
    if len(matches) == 1:
        return matches[0], arg.strip()
    picked = _choose("Tool", matches or filter_options(list(SLASH), word) or list(SLASH))
    return (picked or None), arg.strip()


def chat_loop(session, chat: str) -> bool:
    """Talk to *chat*; returns False to quit pm-code, True to go back to the list.

    The chat's .prompt file is the user's original intent (the first
    message); each line typed here is a refinement of it, or a /tool.
    """
    console.print(f"\n[bold]{escape(chat)}[/bold]")
    console.print(escape(session.outline(chat)))
    console.print(f"[dim]Intent:[/dim] {escape(session.intent(chat))}")
    if session.final(chat) is None:
        _send(session, chat, None)
    else:
        _print_reply(session.final(chat))
    console.print("[dim]Type a refinement, or / for tools (/help). Empty line: back to the list.[/dim]")
    while True:
        text = _ask_follow_up(chat)
        if text is None:
            return False
        if not text:
            return True
        if not text.startswith("/"):
            _send(session, chat, text)
            continue
        name, arg = resolve_tool(text)
        if name is None:
            continue
        if name == "back":
            return True
        try:
            SLASH[name][1](session, chat, arg)
        except Exception as exc:  # noqa: BLE001 — shown, the session continues
            console.print(f"[red]Error:[/red] {escape(str(exc))}")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("chat", required=False)
@click.option("--model", "-m", default=None, metavar="PROVIDER/MODEL",
              help="LiteLLM model for the nodes, written to settings.yml when the project has none. "
                   "Default: settings.yml's, then $PM_CODE_MODEL.")
@click.option("--root", default=".", show_default=True, type=click.Path(file_okay=False),
              help="The promptmap project (map/, settings.yml, .promptmap/).")
@click.option("--map-dir", default="map", show_default=True, help="Directory of .prompt files, under --root.")
@click.option("--code-dir", default=".", show_default=True,
              help="Where coding_agent nodes write their code, relative to --root.")
def main(chat, model, root, map_dir, code_dir):
    """Pick a chat, talk it through, and turn its final answer into coding agents and more chats.

    CHAT goes straight into that chat (a new name starts one).
    """
    from promptmap.workspace import Workspace

    from promptmap_code.session import Session

    root_path = Path(root).resolve()
    root_path.mkdir(parents=True, exist_ok=True)
    load_env(root_path)
    model_name = resolve_model_name(root_path, map_dir, model)
    if not model_name:
        console.print("[red]Error:[/red] no model. Pass --model anthropic/claude-sonnet-5-5 "
                      "(any LiteLLM name), set $PM_CODE_MODEL, or run `promptmap init`.")
        sys.exit(2)
    missing = missing_api_keys(model_name)
    if missing:
        console.print(f"[red]Error:[/red] {escape(model_name)} needs {', '.join(missing)}, which is not set. "
                      f"Add it to {escape(str(root_path / '.env'))} or export it.")
        sys.exit(2)
    for note in ensure_project(root_path, map_dir, model_name):
        console.print(f"[green]{escape(note)}[/green]")

    workspace = Workspace(root_path, map_dir=map_dir, code_dir=code_dir)
    from promptmap_code.session import INSTRUCTIONS_FILE, ensure_instructions

    session = Session(workspace, progress=NodeProgress())
    if ensure_instructions(root_path):
        console.print(f"[green]wrote {INSTRUCTIONS_FILE}: what every chat is told (edit it to change that)[/green]")
    updated = session.sync_all()
    if updated:
        console.print(f"[green]moved the finish guide into {INSTRUCTIONS_FILE} for: {escape(', '.join(updated))}[/green]")
    console.print(f"[bold]pm-code[/bold] [dim]{escape(model_name)} · map {escape(str(workspace.root / map_dir))} "
                  f"· code {escape(str(workspace.code_dir))}[/dim]")

    current = chat
    if current and not session.is_chat(current) and not new_chat(session, current):
        current = None
    while True:
        if current is None:
            current = select_chat(session)
            if current is None:
                return
        if not chat_loop(session, current):
            return
        current = None


if __name__ == "__main__":
    main()
