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


# ---------------------------------------------------------------------------
# .env — the provider's API key
# ---------------------------------------------------------------------------

#: provider -> (env var written to .env, other names that also count as set)
PROVIDER_ENV_VARS: dict[str, tuple[str, tuple[str, ...]]] = {
    "openai": ("OPENAI_API_KEY", ()),
    "anthropic": ("ANTHROPIC_API_KEY", ()),
    "gemini": ("GEMINI_API_KEY", ("GOOGLE_API_KEY",)),
    "groq": ("GROQ_API_KEY", ()),
    "mistral": ("MISTRAL_API_KEY", ()),
    "deepseek": ("DEEPSEEK_API_KEY", ()),
    "cohere": ("COHERE_API_KEY", ()),
    "huggingface": ("HUGGINGFACE_API_KEY", ("HF_TOKEN",)),
}


def _env_names(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().removeprefix("export ").strip()
        if line and not line.startswith("#") and "=" in line:
            names.add(line.split("=", 1)[0].strip())
    return names


def ensure_env_key(root: Path, provider: str, ask: bool) -> str | None:
    """Append *provider*'s API key variable to *root*/.env unless it's already there.

    With *ask*, the value is asked for (hidden; Enter leaves it blank).  A key
    already in .env or the environment, under any of its names, is left alone.
    Also gitignores .env.  Returns a line describing what happened, or None.
    """
    if provider not in PROVIDER_ENV_VARS:
        return None
    name, aliases = PROVIDER_ENV_VARS[provider]
    path = Path(root) / ".env"
    present = _env_names(path)
    if any(n in present or os.environ.get(n) for n in (name, *aliases)):
        return None
    value = ""
    if ask:
        import click

        value = click.prompt(
            f"{name} (Enter to fill in later)", default="", show_default=False, hide_input=True
        ).strip()
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    sep = "" if not text or text.endswith("\n") else "\n"
    path.write_text(f"{text}{sep}{name}={value}\n", encoding="utf-8")
    _gitignore_env(Path(root))
    action = "added" if text else "created"
    return f"{action} {path} ({name}{'' if value else ' — fill in the value'})"


def _gitignore_env(root: Path) -> None:
    path = root / ".gitignore"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if ".env" in {line.strip().lstrip("/") for line in text.splitlines()}:
        return
    sep = "" if not text or text.endswith("\n") else "\n"
    path.write_text(f"{text}{sep}.env\n", encoding="utf-8")
