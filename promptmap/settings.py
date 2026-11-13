"""
Project settings: which provider and model the default LLM backend calls.

A project without a client.py runs its prompts through LiteLLM, using the
``provider`` and ``model`` in ``settings.yml`` at the project root (next to
``map/``)::

    provider: anthropic
    model: claude-sonnet-4-5

``promptmap init`` writes it (``--provider`` / ``--model``, or asked
interactively).  The values are not checked when a run reads them, so the file
can be edited by hand to any provider/model LiteLLM accepts.  The provider and
model choices ``init`` offers come from the model list bundled inside the
installed litellm package — read as a file, so no network call and no litellm
import just to show a menu.

API keys come from the environment (or ``.env``), named the way LiteLLM
expects: ``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``, ``GEMINI_API_KEY``, ...
"""

from __future__ import annotations

import json
import os
from functools import cache
from pathlib import Path

SETTINGS_FILE = "settings.yml"
_LITELLM_MODELS_FILE = "model_prices_and_context_window_backup.json"
_CHAT_MODES = {"chat", "responses"}


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
        "# LLM used by `promptmap run` when there is no client.py (called via LiteLLM).\n"
        "# provider: any LiteLLM provider; model: any model that provider serves.\n"
        "# Edit freely — values are not validated. API keys come from .env / the environment.\n"
    )
    path.write_text(header + yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _strip_prefix(name: str, provider: str) -> str:
    """'groq/llama-3' -> 'llama-3': settings.yml keeps the bare model name."""
    return name[len(provider) + 1:] if name.startswith(provider + "/") else name


@cache
def _provider_models() -> dict[str, list[str]]:
    """Chat models per LiteLLM provider, from litellm's bundled model list."""
    import importlib.util

    spec = importlib.util.find_spec("litellm")
    if spec is None or not spec.submodule_search_locations:
        return {}
    path = Path(next(iter(spec.submodule_search_locations))) / _LITELLM_MODELS_FILE
    try:
        prices = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    models: dict[str, set[str]] = {}
    for name, info in prices.items():
        if not isinstance(info, dict) or info.get("mode") not in _CHAT_MODES or "*" in name:
            continue
        provider = info.get("litellm_provider")
        if provider:
            models.setdefault(provider, set()).add(_strip_prefix(name, provider))
    return {p: sorted(names) for p, names in sorted(models.items())}


def supported_providers() -> list[str]:
    """Providers offered in the menu: those with chat models in LiteLLM's list."""
    return list(_provider_models())


def is_known_provider(provider: str) -> bool:
    """True for a menu provider or any other provider name LiteLLM knows."""
    if provider in _provider_models():
        return True
    import litellm

    return provider in litellm.provider_list


def provider_models(provider: str) -> list[str]:
    """Known chat models for *provider*; empty when LiteLLM lists none."""
    return list(_provider_models().get(provider, []))


# ---------------------------------------------------------------------------
# Interactive selection
# ---------------------------------------------------------------------------

def is_interactive() -> bool:
    """True when stdin is a terminal someone can answer prompts in."""
    import sys

    return sys.stdin.isatty()


_MAX_SHOWN = 30


def choose(label: str, options: list[str], allow_custom: bool, accept=None) -> str:
    """Ask for one of *options* by number or name; other text filters the list.

    Enter on its own returns "" — skip, leave it blank.  With *allow_custom*,
    text matching nothing can be used as-is (a model the list doesn't know).
    *accept* (name -> bool) lets other names through without a confirmation.
    """
    import click

    if not options:
        return click.prompt(f"{label} (Enter to leave blank)", default="", show_default=False).strip()
    shown = options
    while True:
        for i, option in enumerate(shown[:_MAX_SHOWN], 1):
            click.echo(f"  {i:>3}. {option}")
        if len(shown) > _MAX_SHOWN:
            click.echo(f"  ... {len(shown) - _MAX_SHOWN} more — type text to filter")
        hint = "number, name, or text to filter"
        if allow_custom:
            hint += "; any other name is used as typed"
        answer = click.prompt(f"{label} ({hint}; Enter to leave blank)", default="", show_default=False).strip()
        if not answer:
            return ""
        if answer.isdigit() and 1 <= int(answer) <= min(len(shown), _MAX_SHOWN):
            return shown[int(answer) - 1]
        if answer in options or (accept is not None and accept(answer)):
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
    return choose("Provider", supported_providers(), allow_custom=False, accept=is_known_provider)


def ask_model(provider: str) -> str:
    models = provider_models(provider)
    if not models:
        import click

        click.echo(f"No known model list for {provider}; type the model name.")
    return choose("Model", models, allow_custom=True)


# ---------------------------------------------------------------------------
# One entry point for `init` and `run`
# ---------------------------------------------------------------------------

def setup_llm(
    root: Path,
    *,
    provider: str | None = None,
    model: str | None = None,
    interactive: bool = False,
    api_key: str | None = None,
    env_values: dict[str, str] | None = None,
    current: dict | None = None,
    validate_provider: bool = True,
) -> list[str]:
    """Write settings.yml and .env from flags, asking only for what's missing.

    Every prompt has a flag: --provider, --model, --api-key / --env NAME=VALUE.
    A value not given falls back to *current* settings, then (when
    *interactive*) a prompt, then blank.  Enter at every prompt — or
    ``interactive=False`` — writes blank settings and a blank .env template,
    for the user to fill in by hand.  Returns lines to print.

    Raises ValueError for an unknown --provider (when *validate_provider*) or
    an --api-key whose variable can't be told.
    """
    current = current or {}
    if provider is not None:
        provider = provider.strip().lower()
        if provider and validate_provider and not is_known_provider(provider):
            raise ValueError(
                f"unknown provider '{provider}'. Choose one of: {', '.join(supported_providers())}"
            )
    elif current.get("provider"):
        provider = str(current["provider"])
    else:
        provider = ask_provider() if interactive else ""

    if model is None:
        if current.get("model") and current.get("provider") == provider:
            model = str(current["model"])
        elif interactive and provider:
            model = ask_model(provider)
        else:
            model = ""
    model = model.strip()

    path = save_settings(root, provider, model)
    notes = [f"{path}: " + (f"{provider}/{model}" if provider and model else
                            "blank — set provider and model before running")]

    values = dict(env_values or {})
    if api_key:
        names = provider_env_vars(provider, model) if provider else []
        if not names:
            raise ValueError(
                f"can't tell which variable --api-key is for provider '{provider or '(none)'}'; "
                "use --env NAME=VALUE instead."
            )
        values[next((n for n in names if "KEY" in n), names[0])] = api_key
    env_path = Path(root) / ".env"
    for name, value in values.items():
        _set_env_line(env_path, name, value)
        notes.append(f"{env_path}: {name}")
    if values:
        _gitignore_env(Path(root))
    if provider:
        notes += ensure_env_keys(root, provider, model, ask=interactive)
    return notes


# ---------------------------------------------------------------------------
# .env — the provider's API key
# ---------------------------------------------------------------------------

#: Variables LiteLLM lists as alternatives (either one works) -> the one to write.
_ALTERNATIVES: dict[frozenset[str], str] = {
    frozenset({"GOOGLE_API_KEY", "GEMINI_API_KEY"}): "GEMINI_API_KEY",
}


def provider_env_vars(provider: str, model: str) -> list[str]:
    """The variables LiteLLM needs to call *provider*/*model* (keys, bases, ...).

    Asked of ``litellm.validate_environment`` with the environment hidden, so
    the answer is the full list whatever is already set.  Empty for a provider
    LiteLLM doesn't recognise or needs nothing for.
    """
    from unittest import mock

    from promptmap.llm import litellm_model, quiet_litellm

    litellm = quiet_litellm()
    try:
        with mock.patch.dict(os.environ, clear=True):
            names = litellm.validate_environment(litellm_model(provider, model))["missing_keys"]
    except Exception:
        return []
    for group, keep in _ALTERNATIVES.items():
        if group <= set(names):
            names = [n for n in names if n not in group or n == keep]
    return list(dict.fromkeys(names))


def _read_env(path: Path) -> dict[str, str]:
    """NAME -> value for each assignment in a .env file."""
    if not path.is_file():
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().removeprefix("export ").strip()
        if line and not line.startswith("#") and "=" in line:
            name, value = line.split("=", 1)
            values[name.strip()] = value.strip().strip("'\"")
    return values


def _set_env_line(path: Path, name: str, value: str) -> None:
    """Fill *name*'s blank line in .env, or append ``NAME=value``."""
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        bare = line.strip().removeprefix("export ").strip()
        if bare.split("=", 1)[0].strip() == name and "=" in bare:
            lines[i] = f"{name}={value}"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return
    sep = "" if not text or text.endswith("\n") else "\n"
    path.write_text(f"{text}{sep}{name}={value}\n", encoding="utf-8")


def ensure_env_keys(root: Path, provider: str, model: str, ask: bool) -> list[str]:
    """Make sure *root*/.env lists the variables *provider*/*model* needs.

    Missing names are appended as ``NAME=`` (a template to fill in).  With
    *ask*, the user is offered to paste each value now (hidden) — or leave it
    and edit .env later.  A variable that already has a value in .env or the
    environment is left alone.  Also gitignores .env.  Returns one line per
    change, for the caller to print.
    """
    path = Path(root) / ".env"
    notes: list[str] = []
    for name in provider_env_vars(provider, model):
        existing = _read_env(path)
        if existing.get(name) or os.environ.get(name):
            continue
        value = ""
        if ask:
            import click

            if click.confirm(
                f"Add your {name} to {path} now? (or fill it in there yourself later)",
                default=False,
            ):
                value = click.prompt(name, default="", show_default=False, hide_input=True).strip()
        if value or name not in existing:
            _set_env_line(path, name, value)
            notes.append(f"{path}: {name}" + ("" if value else " (blank — fill it in before running)"))
    if notes:
        _gitignore_env(Path(root))
    return notes


def _gitignore_env(root: Path) -> None:
    path = root / ".gitignore"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if ".env" in {line.strip().lstrip("/") for line in text.splitlines()}:
        return
    sep = "" if not text or text.endswith("\n") else "\n"
    path.write_text(f"{text}{sep}.env\n", encoding="utf-8")
