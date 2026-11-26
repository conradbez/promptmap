"""
``pm-code`` — build software by building a promptmap pipeline, in conversation.

::

    pm-code                              # chat in the terminal
    pm-code "build a todo CLI in python" # one request, then exit
    pm-code -i "build a todo CLI"        # one request, then keep chatting

The project is the current directory (``--root``): nodes go in ``map/``, runs in
``.promptmap/``, and coding_agent nodes write code into ``--code-dir``
(default: the current directory).

The system prompt is pm-code's rules, then ``PM_CODE.md`` from the project root
(if any), then promptmap's skill guide as reference (``--no-skill`` drops it).
"""

from __future__ import annotations

import json
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


def _describe(action: dict, output: dict) -> None:
    args = ", ".join(f"{k}={_short(v)}" for k, v in action.get("args", {}).items())
    colour = "red" if output.get("returncode") else "cyan"
    console.print(f"[{colour}]⏺ {escape(action.get('tool', '?'))}[/{colour}]([dim]{escape(args)}[/dim])")
    lines = str(output.get("output", "")).splitlines()
    for line in lines[:6]:
        console.print(f"  [dim]⎿ {escape(line[:160])}[/dim]")
    if len(lines) > 6:
        console.print(f"  [dim]⎿ … {len(lines) - 6} more lines[/dim]")


def _short(value) -> str:
    text = value if isinstance(value, str) else json.dumps(value)
    text = " ".join(text.split())
    return repr(text[:60] + "…") if len(text) > 60 else repr(text)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("request", required=False)
@click.option("--model", "-m", default=None, metavar="PROVIDER/MODEL",
              help="LiteLLM model that drives pm-code. Default: settings.yml's, then $PM_CODE_MODEL.")
@click.option("--root", default=".", show_default=True, type=click.Path(file_okay=False),
              help="The promptmap project (map/, settings.yml, .promptmap/).")
@click.option("--map-dir", default="map", show_default=True, help="Directory of .prompt files, under --root.")
@click.option("--code-dir", default=".", show_default=True,
              help="Where coding_agent nodes write their code, relative to --root.")
@click.option("--max-steps", default=40, show_default=True, help="Most tool calls per request.")
@click.option("--cost-limit", default=0.0, show_default=True, help="Stop pm-code itself after this many dollars (0: none).")
@click.option("--no-skill", is_flag=True, help="Leave promptmap's skill guide out of the system prompt.")
@click.option("-i", "--interactive", is_flag=True, help="Keep chatting after REQUEST.")
@click.option("--log", "log_path", default=None, type=click.Path(dir_okay=False),
              help="Write the conversation (with tool calls) to this JSON file on exit.")
def main(request, model, root, map_dir, code_dir, max_steps, cost_limit, no_skill, interactive, log_path):
    """Build a promptmap pipeline of chat and coding_agent nodes by talking to it.

    pm-code never writes files itself: it creates, connects, renames and runs
    nodes through promptmap. Code appears in --code-dir when coding_agent nodes run.
    """
    from promptmap.workspace import Workspace

    from promptmap_code.agent import build_agent

    root_path = Path(root).resolve()
    root_path.mkdir(parents=True, exist_ok=True)
    model_name = resolve_model_name(root_path, map_dir, model)
    if not model_name:
        console.print("[red]Error:[/red] no model. Pass --model anthropic/claude-sonnet-5-5 "
                      "(any LiteLLM name), set $PM_CODE_MODEL, or run `promptmap init`.")
        sys.exit(2)
    for note in ensure_project(root_path, map_dir, model_name):
        console.print(f"[green]{escape(note)}[/green]")

    workspace = Workspace(root_path, map_dir=map_dir, code_dir=code_dir)
    agent = build_agent(workspace, model_name, max_steps=max_steps, cost_limit=cost_limit,
                        skill=not no_skill, on_action=_describe)
    console.print(f"[bold]pm-code[/bold] [dim]{escape(model_name)} · map {escape(str(workspace.root / map_dir))} "
                  f"· code {escape(str(workspace.code_dir))}[/dim]")

    def turn(text: str) -> None:
        try:
            reply = agent.send(text)
        except KeyboardInterrupt:
            console.print("[yellow]Interrupted.[/yellow]")
            return
        except Exception as exc:  # noqa: BLE001 — shown, the session continues
            console.print(f"[red]Error:[/red] {escape(str(exc))}")
            return
        console.print()
        console.print(escape(reply))
        console.print()

    try:
        if request:
            turn(request)
            if not interactive:
                return
        console.print("[dim]Describe what to build. Empty line or Ctrl-D quits.[/dim]")
        while True:
            try:
                text = click.prompt("", prompt_suffix="> ", default="", show_default=False)
            except (EOFError, click.Abort):
                break
            if not text.strip():
                break
            turn(text)
    finally:
        if log_path:
            Path(log_path).write_text(json.dumps(agent.log(), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
