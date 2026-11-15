"""
``promptmap edit`` — change a project from the command line.

Loads the project with :func:`~promptmap.construct.file_to_construct`, applies
the edits, then writes it back with :func:`~promptmap.construct.construct_to_files`.

Non-interactive by default: with no edit flags it prints what can be edited
and the exact command to do it, so a script or an LLM can find its way
without a prompt.  ``-i`` asks instead.

What can be edited, and where the choices come from:

* promptdata values — keys are the ``promptdata()`` calls in the map, plus any
  already in promptdata.yml.
* a model's ``config()`` keys — offered keys are the ones promptmap knows for
  that model's ``model_type`` (built-ins, the kind's own, registered extras).
* a model's template text — from a file, stdin, or ``$EDITOR``.
"""

from __future__ import annotations

import shlex
import sys

import click

from promptmap.construct import Construct, construct_to_files, file_to_construct, parse_cli_value


def config_keys_for(construct: Construct, model: str) -> list[str]:
    """The config() keys promptmap acts on for *model*'s model_type, sorted."""
    from promptmap.executor.parser_initial import _BUILTIN_CONFIG_KEYS, _extra_config_keys
    from promptmap.model_types import get_model_kind

    model_type = str(construct.get_config(model).get("model_type", ""))
    kind = get_model_kind(model_type)
    kind_keys = set(kind.config_keys) if kind is not None else set()
    return sorted(set(_BUILTIN_CONFIG_KEYS) | kind_keys | _extra_config_keys)


def _split_pairs(pairs: tuple[str, ...], flag: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise click.UsageError(f"{flag} must be KEY=VALUE, got: {pair!r}")
        k, _, v = pair.partition("=")
        out[k.strip()] = v
    return out


def _promptdata_keys(construct: Construct) -> list[str]:
    return list(dict.fromkeys([*construct.expected_promptdata(), *construct.promptdata]))


# ---------------------------------------------------------------------------
# Guide: what can be edited, and the command for it (no prompts)
# ---------------------------------------------------------------------------

def _guide_project(construct: Construct) -> None:
    keys = _promptdata_keys(construct)
    click.echo("promptdata (promptdata.yml):")
    if keys:
        for key in keys:
            click.echo(f"  {key} = {construct.promptdata.get(key, '')!r}")
    else:
        click.echo("  (no promptdata() keys in this map)")
    click.echo("Models:")
    for name, f in construct.models.items():
        click.echo(f"  {name}  ({construct.map_dir}/{f.path})")
    click.echo("\nuse")
    example = keys[0] if keys else "KEY"
    click.echo(f"    promptmap edit --promptdata {example}=VALUE")
    click.echo("    promptmap edit MODEL                  (show a model's config() and how to change it)")


def _guide_model(construct: Construct, model: str) -> None:
    f = construct.models[model]
    current = construct.get_config(model)
    click.echo(f"config() of '{model}':  ({construct.map_dir}/{f.path})")
    if current:
        for k, v in current.items():
            click.echo(f"  {k} = {v!r}")
    else:
        click.echo("  (no config() call)")
    click.echo(f"Known keys: {', '.join(config_keys_for(construct, model))}")
    from promptmap.executor.parser_initial import detect_used_promptdata

    reads = detect_used_promptdata(f.source)
    if reads:
        shown = ", ".join(f"{k}={construct.promptdata.get(k, '')!r}" for k in reads)
        click.echo(f"Reads promptdata: {shown}")
    m = shlex.quote(model)
    click.echo("\nuse")
    click.echo(f"    promptmap edit {m} --config KEY=VALUE        (VALUE is JSON if it parses: true, 3, [\"a\"])")
    click.echo(f"    promptmap edit {m} --unset KEY")
    click.echo(f"    promptmap edit {m} --template-file PATH      ('-' reads stdin)")
    if reads:
        click.echo(f"    promptmap edit --promptdata {reads[0]}=VALUE")


# ---------------------------------------------------------------------------
# Interactive (-i)
# ---------------------------------------------------------------------------

def _edit_promptdata_interactive(construct: Construct) -> None:
    keys = _promptdata_keys(construct)
    if not keys:
        click.echo("No promptdata() keys in this map.")
        return
    click.echo("promptdata (Enter keeps the current value, '-' clears it):")
    for key in keys:
        current = construct.promptdata.get(key, "")
        value = click.prompt(f"  {key}", default=current, show_default=bool(current))
        construct.promptdata[key] = "" if value == "-" else value


def _edit_model_interactive(construct: Construct, model: str) -> None:
    known = config_keys_for(construct, model)
    current = construct.get_config(model)
    click.echo(f"config() of '{model}':")
    for k, v in current.items():
        click.echo(f"  {k} = {v!r}")
    click.echo(f"Known keys: {', '.join(known)}")
    while True:
        key = click.prompt("Key to set (Enter to finish)", default="", show_default=False).strip()
        if not key:
            break
        if key not in known:
            click.echo(f"  note: '{key}' is not a key promptmap acts on for this model.")
        default = current.get(key)
        raw = click.prompt(
            f"  {key} ('-' removes)",
            default="" if default is None else str(default),
            show_default=default is not None,
        )
        construct.set_config(model, **{key: None if raw == "-" else parse_cli_value(raw)})
        current = construct.get_config(model)
    if click.confirm("Edit the template text?", default=False):
        _edit_template_in_editor(construct, model)


def _edit_template_in_editor(construct: Construct, model: str) -> None:
    f = construct.models[model]
    suffix = ".prompt.jinja" if f.path.endswith(".prompt.jinja") else ".prompt"
    edited = click.edit(f.source, extension=suffix, require_save=True)
    if edited is not None:
        f.source = edited


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------

def register_command(main) -> None:
    """Attach ``promptmap edit`` to the *main* Click group."""

    @main.command("edit", short_help="Edit promptdata, a model's config() or its template.")
    @click.argument("model", required=False)
    @click.option("--map-dir", default="map", show_default=True, help="Directory containing *.prompt files.")
    @click.option("--config", "--set", "config_pairs", multiple=True, metavar="KEY=VALUE",
                  help="Set a config() key on MODEL. VALUE is JSON if it parses (true, 3, [\"a\"]), else text.")
    @click.option("--unset", "unset_keys", multiple=True, metavar="KEY", help="Remove a config() key from MODEL.")
    @click.option("--template-file", "template_file", type=click.File("r", encoding="utf-8"), default=None,
                  metavar="PATH", help="Replace MODEL's template with this file's text ('-' reads stdin).")
    @click.option("--template", "edit_template", is_flag=True, help="Open MODEL's template in $EDITOR.")
    @click.option("--promptdata", "promptdata_pairs", multiple=True, metavar="KEY=VALUE",
                  help="Set a value in promptdata.yml.")
    @click.option("-i", "--interactive", is_flag=True, help="Ask for values instead of printing what to run.")
    def edit(model: str | None, map_dir: str, config_pairs: tuple[str, ...], unset_keys: tuple[str, ...],
             template_file, edit_template: bool, promptdata_pairs: tuple[str, ...], interactive: bool) -> None:
        """Edit the project, then write the changed files back.

        Never prompts unless -i is given.  With no edit options it prints the
        current values, the keys you can set, and the command that sets them.

        \b
        Examples:
          promptmap edit                                 (show promptdata and models)
          promptmap edit --promptdata topic=cats
          promptmap edit article                         (show article's config() and options)
          promptmap edit article --config model=low --unset cache
          promptmap edit article --template-file new.prompt
          promptmap edit article -i                      (ask, step by step)
        """
        from promptmap.llm import try_load_client_module

        try:
            try_load_client_module(map_dir)  # registers project-local model kinds and their keys
            construct = file_to_construct(".", map_dir)
        except ValueError as exc:
            click.echo(f"Error: {exc}", err=True)
            sys.exit(1)

        if model is not None and model not in construct.models:
            click.echo(f"Error: no model named '{model}'. Models: {', '.join(sorted(construct.models))}", err=True)
            sys.exit(1)
        if model is None and (config_pairs or unset_keys or edit_template or template_file):
            raise click.UsageError("--config, --unset, --template and --template-file need a MODEL.")

        flagged = bool(config_pairs or unset_keys or edit_template or template_file or promptdata_pairs)

        expected = set(construct.expected_promptdata())
        for k, v in _split_pairs(promptdata_pairs, "--promptdata").items():
            if k not in expected:
                click.echo(f"note: no model reads promptdata('{k}').", err=True)
            construct.promptdata[k] = v

        if model is not None:
            known = set(config_keys_for(construct, model))
            changes: dict = {k: parse_cli_value(v) for k, v in _split_pairs(config_pairs, "--config").items()}
            changes.update({k: None for k in unset_keys})
            for k in changes:
                if k not in known:
                    click.echo(f"note: '{k}' is not a config() key promptmap acts on for '{model}'. "
                               f"Known keys: {', '.join(sorted(known))}", err=True)
            if changes:
                construct.set_config(model, **changes)
            if template_file is not None:
                construct.models[model].source = template_file.read()
            if edit_template:
                _edit_template_in_editor(construct, model)

        if not flagged:
            if interactive:
                if model is None:
                    _edit_promptdata_interactive(construct)
                else:
                    _edit_model_interactive(construct, model)
            else:
                if model is None:
                    _guide_project(construct)
                else:
                    _guide_model(construct, model)
                return

        changed = construct_to_files(construct)
        for path in changed:
            click.echo(f"  wrote  {path}")
        if not changed:
            click.echo("  no changes")
