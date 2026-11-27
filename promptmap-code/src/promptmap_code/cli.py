"""
``pm-code`` — pick a chat, talk it through, and let its final answer create the next nodes.

::

    pm-code              # pick a chat (or name a new one)
    pm-code spec         # go straight into chat 'spec' (created if new)

In a chat, each line is a follow-up.  ``/done`` reads the chat's final reply
for ``### coding_agent: name`` and ``### chat: name`` headings and creates those
nodes in the chat's folder, each ``ref()``ing the chat; then it offers to run
the coding agents.  ``/map`` shows where the chat sits, ``/back`` returns to
the list.

The project is the current directory (``--root``): nodes go in ``map/``, runs in
``.promptmap/``, and coding_agent nodes write code into ``--code-dir``.
"""

from __future__ import annotations

import json  # noqa: F401  (used by missing_api_keys)
import os
import sys
from pathlib import Path

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




HELP = ("[dim]Type a follow-up. /done: create the nodes the final reply names · /run: run this chat's coding "
        "agents · /map · /history · /back · empty line: back to the list[/dim]")


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
    from promptmap.workspace import WorkspaceError

    request = _ask(f"What should '{name}' work out")
    if not request:
        return False
    try:
        session.create_chat(name, request)
    except WorkspaceError as exc:
        console.print(f"[red]Error:[/red] {escape(str(exc))}")
        return False
    console.print(f"[green]created chat '{escape(name)}'[/green]")
    return True


def spawn(session, chat: str) -> None:
    from promptmap.workspace import WorkspaceError

    try:
        done, plan = session.spawn(chat)
    except WorkspaceError as exc:
        console.print(f"[red]Error:[/red] {escape(str(exc))}")
        return
    if not done:
        console.print("[yellow]The final reply names no nodes.[/yellow] Ask the chat to finish with "
                      "'### coding_agent: <name>' headings (and '### chat: <name>' for open parts).")
    for child, status in done:
        colour = "green" if status in ("created", "updated") else "dim" if status == "unchanged" else "yellow"
        console.print(f"  [{colour}]{escape(status)}[/{colour}] {child.kind} [cyan]{escape(child.name)}[/cyan]")
    for note in plan.dropped:
        console.print(f"  [yellow]left out[/yellow] {escape(note)}")
    agents = [c.name for c, s in done if c.kind == "coding_agent" and s in ("created", "updated")]
    if agents and click.confirm(f"Run {len(agents)} coding agent{'s' if len(agents) > 1 else ''} now?",
                                default=True):
        run_nodes(session, agents)


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


def chat_loop(session, chat: str) -> bool:
    """Talk to *chat*; returns False to quit pm-code, True to go back to the list."""
    from promptmap.workspace import WorkspaceError

    def send(message: str | None) -> None:
        try:
            _print_reply(session.reply(chat, message))
        except KeyboardInterrupt:
            console.print("[yellow]Interrupted.[/yellow]")
        except Exception as exc:  # noqa: BLE001 — shown, the session continues
            console.print(f"[red]Error:[/red] {escape(str(exc))}")

    console.print(f"\n[bold]{escape(chat)}[/bold]")
    console.print(escape(session.outline(chat)))
    if session.final(chat) is None:
        send(None)
    else:
        _print_reply(session.final(chat))
    console.print(HELP)
    while True:
        text = _ask(chat)
        if text is None:
            return False
        if not text or text == "/back":
            return True
        if text == "/done":
            spawn(session, chat)
        elif text == "/run":
            agents = [n["name"] for n in session.children(chat) if n["model_type"] == "coding_agent"]
            if agents:
                run_nodes(session, agents)
            else:
                console.print("[dim]No coding agents yet: /done creates them from the final reply.[/dim]")
        elif text == "/map":
            console.print(escape(session.outline(chat)))
        elif text == "/history":
            try:
                for turn in session.history(chat):
                    console.print(f"[bold]{turn['role']}:[/bold] {escape(turn['content'])}\n")
            except WorkspaceError as exc:
                console.print(f"[red]Error:[/red] {escape(str(exc))}")
        elif text.startswith("/"):
            console.print(HELP)
        else:
            send(text)


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
    session = Session(workspace, progress=NodeProgress())
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
