"""
``promptmap chat`` — talk to a ``model_type="chat"`` model.

::

    promptmap chat MODEL "Shorter please"   # add a follow-up; runs MODEL (and upstream); prints the reply
    promptmap chat MODEL                    # no message: redo the last reply (pick up upstream changes)
    promptmap chat MODEL -i                 # a back-and-forth loop in the terminal
    promptmap chat MODEL --new              # start a new thread
    promptmap chat MODEL --history          # print the current thread

It is ``run(select=[MODEL], chat_input={MODEL: message})`` plus printing.  A
plain ``promptmap run`` treats chat models like any other model and never adds
turns.
"""

from __future__ import annotations

import sys

import click
from rich.console import Console
from rich.markup import escape

console = Console(highlight=False, soft_wrap=True)
err_console = Console(stderr=True, highlight=False)


def _print_history(model: str, turns: list[dict]) -> None:
    if not turns:
        console.print(f"[dim]No conversation yet for {model}.[/dim]")
        return
    follow_up = 0
    for i, turn in enumerate(turns):
        if turn["role"] == "assistant":
            label = "reply" if i else "reply to the request"
            console.print(f"[bold cyan]{label}[/bold cyan]")
        else:
            follow_up += 1
            console.print(f"[bold yellow]follow-up {follow_up}[/bold yellow]")
        console.print(escape(turn["content"]))
        console.print()


def register_command(main: click.Group) -> None:
    @main.command("chat", short_help="Send a follow-up to a chat model and print its reply.")
    @click.argument("model")
    @click.argument("message", required=False)
    @click.option("--map-dir", default="map", show_default=True, help="Directory containing *.prompt files.")
    @click.option("-i", "--interactive", is_flag=True, help="Keep chatting: read follow-ups from the terminal.")
    @click.option("--new", "new_thread", is_flag=True, help="Start a new conversation (the old one is kept).")
    @click.option("--history", is_flag=True, help="Print the current conversation and exit.")
    @click.option("--promptdata", multiple=True, metavar="KEY=VALUE",
                  help="Inject a variable via promptdata(); overrides promptdata.yml.")
    @click.option("--secret", "secret_flags", multiple=True, metavar="NAME[=VALUE]",
                  help="Run-scoped credential, as for `promptmap run`.")
    @click.option("--verbose", is_flag=True, help="Show the per-model run log.")
    def chat_cmd(
        model: str, message: str | None, map_dir: str, interactive: bool, new_thread: bool,
        history: bool, promptdata: tuple[str, ...], secret_flags: tuple[str, ...], verbose: bool,
    ) -> None:
        """Send MESSAGE as a follow-up to the chat model MODEL and print its reply.

        MODEL must set {{ config(model_type="chat") }}. Its .prompt file is the
        request; follow-ups and replies are kept in the run database. Only the
        latest reply is passed on to downstream models.

        Without MESSAGE the last reply is regenerated in place (picking up
        upstream changes) and no turn is added.

        \b
        Examples:
          promptmap chat poem "Shorter please"
          promptmap chat poem
          promptmap chat poem -i
          promptmap chat poem --new
          promptmap chat poem --history
        """
        import promptmap
        from promptmap import db
        from promptmap.cli import _secrets_or_exit, init_db_or_exit
        from promptmap.construct import file_to_construct
        from promptmap.llm import try_load_client_module
        from promptmap.model_types import get_model_kind

        init_db_or_exit()
        try:
            try_load_client_module(map_dir)
            project = file_to_construct(".", map_dir)
            specs = project.specs()
        except (FileNotFoundError, ValueError) as exc:
            err_console.print(f"[red]Error:[/red] {escape(str(exc))}")
            sys.exit(1)
        spec = specs.get(model)
        if spec is None:
            err_console.print(f"[red]Unknown model:[/red] '{model}'")
            sys.exit(1)
        kind = get_model_kind(spec.model_type)
        if kind is None or not kind.chat:
            err_console.print(
                f"[red]Error:[/red] '{model}' is not a chat model. "
                f"Add {{{{ config(model_type=\"chat\") }}}} to its .prompt file."
            )
            sys.exit(1)

        if history:
            _print_history(model, db.get_chat_turns(model))
            return

        promptdata_vars = project.run_promptdata()
        for value in promptdata:
            if "=" not in value:
                err_console.print(f"[red]Error:[/red] --promptdata must be KEY=VALUE, got: {value!r}")
                sys.exit(1)
            key, _, val = value.partition("=")
            promptdata_vars[key] = val
        secrets = _secrets_or_exit(secret_flags)

        def turn(text: str | None, reset: bool) -> bool:
            try:
                results = promptmap.run(
                    map_dir=map_dir,
                    select=[model],
                    verbose=verbose,
                    promptdata=promptdata_vars or None,
                    secrets=secrets,
                    storage_backend=db.backend(),
                    chat_input={model: text} if text else None,
                    chat_reset=[model] if reset else None,
                )
            except Exception as exc:  # noqa: BLE001 — shown, not raised
                err_console.print(f"[red]Error:[/red] {escape(str(exc))}")
                return False
            value = results.get(model)
            if isinstance(value, (promptmap.ModelError, promptmap.ModelStatus)):
                err_console.print(f"[red]{escape(str(value))}[/red]")
                return False
            console.print(escape(str(value)))
            return True

        ok = turn(message, new_thread)
        if not interactive:
            sys.exit(0 if ok else 1)
        console.print("[dim]Type a follow-up (empty line or Ctrl-D to stop).[/dim]")
        while True:
            try:
                text = click.prompt("", prompt_suffix="> ", default="", show_default=False)
            except (EOFError, click.Abort):
                break
            if not text.strip():
                break
            console.print()
            turn(text, False)
