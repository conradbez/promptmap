"""
Project settings: which provider and model the default LLM backend calls.

A project without a client.py runs its prompts through aisuite, using the
``provider`` and ``model`` in ``settings.yml`` at the project root (next to
``map/``)::

    provider: anthropic
    model: claude-sonnet-4-5

``promptmap init`` writes it (``--provider`` / ``--model``, or asked
interactively).  The values are not checked when a run reads them, so the file
can be edited by hand to any provider/model aisuite accepts.  The model choices
``init`` offers come from ``provider_models.json``, a snapshot of LiteLLM's model
list taken at release time by ``scripts/update_provider_models.py``.

API keys come from the environment (or ``.env``), named the way each provider's
SDK expects: ``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``, ``GEMINI_API_KEY``, ...
"""

from __future__ import annotations

import json
import os
from functools import cache
from pathlib import Path

SETTINGS_FILE = "settings.yml"
_MODELS_FILE = Path(__file__).with_name("provider_models.json")


def settings_path(map_dir: str) -> str:
    """The settings.yml path for *map_dir* (i.e. in its parent)."""
    return os.path.join(os.path.dirname(os.path.abspath(map_dir)), SETTINGS_FILE)


def load_settings(map_dir: str) -> dict:
    """Return settings.yml as a dict, or {} when there isn't one."""
    import yaml

    path = settings_path(map_dir)
    if not os.path.isfile(path):
        return {}
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a mapping like 'provider: openai'.")
    return data


def save_settings(root: Path, provider: str, model: str) -> Path:
    """Write provider/model into *root*/settings.yml, keeping any other keys."""
    import yaml

    path = Path(root) / SETTINGS_FILE
    data = {}
    if path.is_file():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data.update(provider=provider, model=model)
    header = (
        "# LLM used by `promptmap run` when there is no client.py.\n"
        "# provider: any aisuite provider; model: any model that provider serves.\n"
        "# Edit freely — values are not validated. API keys come from the environment.\n"
    )
    path.write_text(header + yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def supported_providers() -> list[str]:
    """Provider names aisuite can call."""
    from aisuite.provider import ProviderFactory

    return sorted(ProviderFactory.get_supported_providers())


@cache
def _provider_models() -> dict[str, list[str]]:
    try:
        return json.loads(_MODELS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def provider_models(provider: str) -> list[str]:
    """Known chat models for *provider*; empty when the snapshot has none."""
    return list(_provider_models().get(provider, []))


# ---------------------------------------------------------------------------
# Interactive selection
# ---------------------------------------------------------------------------

def is_interactive() -> bool:
    """True when stdin is a terminal someone can answer prompts in."""
    import sys

    return sys.stdin.isatty()


_MAX_SHOWN = 30


def choose(label: str, options: list[str], allow_custom: bool) -> str:
    """Ask for one of *options* by number or name; other text filters the list.

    With *allow_custom*, text matching nothing can be used as-is — for a model
    the snapshot doesn't know about.
    """
    import click

    if not options:
        return click.prompt(label).strip()
    shown = options
    while True:
        for i, option in enumerate(shown[:_MAX_SHOWN], 1):
            click.echo(f"  {i:>3}. {option}")
        if len(shown) > _MAX_SHOWN:
            click.echo(f"  ... {len(shown) - _MAX_SHOWN} more — type text to filter")
        hint = "number, name, or text to filter"
        if allow_custom:
            hint += "; any other name is used as typed"
        answer = click.prompt(f"{label} ({hint})").strip()
        if answer.isdigit() and 1 <= int(answer) <= min(len(shown), _MAX_SHOWN):
            return shown[int(answer) - 1]
        if answer in options:
            return answer
        matches = [o for o in options if answer.lower() in o.lower()]
        if matches:
            shown = matches
            continue
        if allow_custom and click.confirm(f"'{answer}' isn't in the list. Use it anyway?", default=True):
            return answer
        click.echo(f"No {label.lower()} matches '{answer}'.")
        shown = options


def ask_provider() -> str:
    return choose("Provider", supported_providers(), allow_custom=False)


def ask_model(provider: str) -> str:
    models = provider_models(provider)
    if not models:
        import click

        click.echo(f"No known model list for {provider}; type the model name.")
    return choose("Model", models, allow_custom=True)
