"""
``promptmap edit`` — change a project from the command line.

Loads the project with :func:`~promptmap.construct.file_to_construct`, applies
the edits, then writes it back with :func:`~promptmap.construct.construct_to_files`.

What can be edited, and where the choices come from:

* promptdata values — keys are the ``promptdata()`` calls in the map, plus any
  already in promptdata.yml.
* a model's ``config()`` keys — offered keys are the ones promptmap knows for
  that model's ``model_type`` (built-ins, the kind's own, registered extras).
* a model's template text — opened in ``$EDITOR``.
"""

from __future__ import annotations

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


def _edit_promptdata_interactive(construct: Construct) -> None:
    keys = list(dict.fromkeys([*construct.expected_promptdata(), *construct.promptdata]))
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
        value = None if raw == "-" else parse_cli_value(raw)
        construct.set_config(model, **{key: value})
        current = construct.get_config(model)
    if click.confirm("Edit the template text?", default=False):
        _edit_template(construct, model)


def _edit_template(construct: Construct, model: str) -> None:
    f = construct.models[model]
    suffix = ".prompt.jinja" if f.path.endswith(".prompt.jinja") else ".prompt"
    edited = click.edit(f.source, extension=suffix, require_save=True)
    if edited is not None:
        f.source = edited


def register_command(main) -> None:
    """Attach ``promptmap edit`` to the *main* Click group."""

    @main.command("edit", short_help="Edit promptdata, a model's config() or its template.")
    @click.argument("model", required=False)
    @click.option("--map-dir", default="map", show_default=True, help="Directory containing *.prompt files.")
    @click.option("--set", "set_pairs", multiple=True, metavar="KEY=VALUE",
                  help="Set a config() key on MODEL. VALUE is JSON if it parses (true, 3, [\"a\"]), else text.")
    @click.option("--unset", "unset_keys", multiple=True, metavar="KEY", help="Remove a config() key from MODEL.")
    @click.option("--template", "edit_template", is_flag=True, help="Open MODEL's template in $EDITOR.")
    @click.option("--promptdata", "promptdata_pairs", multiple=True, metavar="KEY=VALUE",
                  help="Set a value in promptdata.yml.")
    def edit(model: str | None, map_dir: str, set_pairs: tuple[str, ...], unset_keys: tuple[str, ...],
             edit_template: bool, promptdata_pairs: tuple[str, ...]) -> None:
        """Edit the project, then write the changed files back.

        With no options it asks: for the promptdata values when no MODEL is
        given, or for MODEL's config() keys and template when one is.

        \b
        Examples:
          promptmap edit                                 (prompt for promptdata values)
          promptmap edit --promptdata topic=cats
          promptmap edit article                         (prompt for article's config/template)
          promptmap edit article --set model=low --unset cache
          promptmap edit article --template
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
        if model is None and (set_pairs or unset_keys or edit_template):
            raise click.UsageError("--set, --unset and --template need a MODEL.")

        flagged = bool(set_pairs or unset_keys or edit_template or promptdata_pairs)

        expected = set(construct.expected_promptdata())
        for k, v in _split_pairs(promptdata_pairs, "--promptdata").items():
            if k not in expected:
                click.echo(f"note: no model reads promptdata('{k}').", err=True)
            construct.promptdata[k] = v

        if model is not None:
            known = set(config_keys_for(construct, model))
            changes: dict = {k: parse_cli_value(v) for k, v in _split_pairs(set_pairs, "--set").items()}
            changes.update({k: None for k in unset_keys})
            for k in changes:
                if k not in known:
                    click.echo(f"note: '{k}' is not a config() key promptmap acts on for '{model}'.", err=True)
            if changes:
                construct.set_config(model, **changes)
            if edit_template:
                _edit_template(construct, model)

        if not flagged:
            if model is None:
                _edit_promptdata_interactive(construct)
            else:
                _edit_model_interactive(construct, model)

        changed = construct_to_files(construct)
        for path in changed:
            click.echo(f"  wrote  {path}")
        if not changed:
            click.echo("  no changes")
