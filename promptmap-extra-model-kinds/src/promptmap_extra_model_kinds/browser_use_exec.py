"""
``browser_use`` — hand the rendered prompt to a `browser-use`_ agent, which does
the task in a real browser and replies with what it found.  Text in, text out.

.. _browser-use: https://github.com/browser-use/browser-use

Defaults: a **local** Chromium (a fresh temporary profile, deleted afterwards)
in a **visible window**, so you can watch it work.  Both can be changed per
model in ``config()``, or for the whole project in settings.yml::

    {{ config(model_type="browser_use", browser_headless=True) }}
    {{ config(model_type="browser_use", browser_cloud=True) }}   # BROWSER_USE_API_KEY

    # settings.yml
    browser_use:
      headless: true      # no window (CI, servers)
      cloud: false        # true: a Browser Use Cloud browser instead of a local one
      model: low          # an alias, provider/model, or a Pydantic AI string

Config keys (each overrides settings.yml's ``browser_use:`` block):

``browser_model``         the LLM driving the browser: a settings.yml alias
                          (``"low"``), ``provider/model`` as in settings.yml, or a
                          Pydantic AI string (``"openai:gpt-4o"``).  Defaults to
                          settings.yml's own provider/model; with neither,
                          browser-use's default (``bu-fast``, BROWSER_USE_API_KEY).
``browser_cloud``         True: a Browser Use Cloud browser.  Default False (local).
``browser_headless``      True: no window.  Default False (a visible window).  On
                          Linux with no display it runs headless anyway.
``browser_max_steps``     step cap for the agent (default 100).
``browser_allowed_domains``  list of domains the browser may visit
                          (``"example.com"``, ``"*.example.org"``).

``PROMPTMAP_BROWSER_HEADLESS=1`` / ``PROMPTMAP_BROWSER_CLOUD=1`` set the
defaults from the environment (e.g. headless in CI) below config() and
settings.yml.

Anything the browser downloads becomes this model's files.  Results go
through the prompt cache, keyed on the task and the resolved model and
browser, so an unchanged task does not re-run.

browser-use runs in its own environment, not promptmap's (its 1.0 release
needs openai 3.x, which litellm 1.102.1 can't use): ``uv`` builds and caches one with the
pinned browser-use on first use, or set ``PROMPTMAP_BROWSER_USE_PYTHON`` to a
Python that has it.  A local browser needs Chrome, Chromium or Edge installed.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from promptmap.model_types import ModelCall, ModelKind, register_model_kind

NAME = "browser_use"

#: Pinned exactly, as browser-use asks: its releases move fast and break APIs.
BROWSER_USE_VERSION = "1.0.4"

CONFIG_KEYS = frozenset({
    "browser_model",
    "browser_cloud",
    "browser_headless",
    "browser_max_steps",
    "browser_allowed_domains",
})

#: settings.yml block holding project-wide defaults.
SETTINGS_KEY = "browser_use"

#: LiteLLM provider name -> Pydantic AI's (browser-use 1.0 takes its strings).
_PYDANTIC_AI_PROVIDER = {
    "gemini": "google-gla",
    "vertex_ai": "google-vertex",
    "xai": "grok",
    "together_ai": "together",
    "azure": "azure",
}

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}


def _flag(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ValueError(f"{name} must be true or false, not {value!r}.")


def _setting(call_config: dict, project: dict, key: str, env: str | None = None, default: Any = None) -> Any:
    """config(browser_<key>) > settings.yml browser_use.<key> > $env > default."""
    if call_config.get(f"browser_{key}") is not None:
        return call_config[f"browser_{key}"]
    if project.get(key) is not None:
        return project[key]
    if env and os.environ.get(env) is not None:
        return os.environ[env]
    return default


def pydantic_ai_model(name: str, settings: dict) -> str:
    """*name* as a browser-use (Pydantic AI) model string.

    A settings.yml alias or ``provider/model`` (LiteLLM names, as in
    settings.yml) becomes ``provider:model``; a ``provider:model`` string is
    kept; a bare model name takes settings.yml's provider.
    """
    from promptmap.settings import model_aliases, resolve_model

    aliases = model_aliases(settings)
    if name in aliases:
        return _to_pydantic_ai(*aliases[name])
    head = name.split("/", 1)[0]
    if ":" in head:
        return name  # already 'openai:gpt-4o'
    if "/" in name:
        return _to_pydantic_ai(*name.split("/", 1))
    target = resolve_model(settings, model=name)
    return _to_pydantic_ai(*target) if target else name


def _to_pydantic_ai(provider: str, model: str) -> str:
    model = model[len(provider) + 1:] if model.startswith(provider + "/") else model
    return f"{_PYDANTIC_AI_PROVIDER.get(provider, provider)}:{model}"


def resolve_options(call: ModelCall) -> dict:
    """Model, browser and limits for this call, from config(), settings.yml and the environment."""
    from promptmap.settings import resolve_model, settings_for

    settings = settings_for(call.spec.path)
    project = settings.get(SETTINGS_KEY) or {}
    if not isinstance(project, dict):
        raise ValueError(f"settings.yml: '{SETTINGS_KEY}' must be a mapping like 'headless: true'.")
    config = call.spec.config

    model_name = _setting(config, project, "model")
    if model_name:
        llm = pydantic_ai_model(str(model_name), settings)
    else:
        default = resolve_model(settings)
        llm = _to_pydantic_ai(*default) if default else None

    cloud = _flag(_setting(config, project, "cloud", "PROMPTMAP_BROWSER_CLOUD", False), "browser_cloud")
    headless = _flag(_setting(config, project, "headless", "PROMPTMAP_BROWSER_HEADLESS", False), "browser_headless")
    if not cloud and not headless and not _has_display():
        headless = True  # nowhere to open a window: run anyway rather than fail
    domains = _setting(config, project, "allowed_domains")
    if isinstance(domains, str):  # config() values arrive as text; a list as JSON
        import json

        text = domains.strip()
        domains = json.loads(text) if text.startswith("[") else [d.strip() for d in text.split(",") if d.strip()]
    return {
        "llm": llm,
        "cloud": cloud,
        "headless": None if cloud else headless,
        "max_steps": int(_setting(config, project, "max_steps", default=100)),
        "allowed_domains": list(domains) if domains else None,
    }


def _has_display() -> bool:
    """False only on Linux/BSD with neither X11 nor Wayland (Docker, CI, SSH)."""
    if sys.platform in ("darwin", "win32"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


async def run_browser_use(rendered: str, call: ModelCall) -> Any:
    options = resolve_options(call)
    # The resolved choices join the key: repointing an alias, or switching
    # local/cloud, is a miss.  Headless is left out — same task, same answer.
    key = (
        f"{rendered}\x00browser_use:{options['llm']}|cloud={options['cloud']}"
        f"|steps={options['max_steps']}|domains={options['allowed_domains']}"
    )
    return await call.compute(key, compute=lambda: _work(rendered, call, options))


async def _work(rendered: str, call: ModelCall, options: dict) -> Any:
    from promptmap.files import Dir, File, FileOutputError, Output

    with tempfile.TemporaryDirectory(prefix="promptmap-browser-") as tmp:
        result = await _exec_agent(rendered, options, Path(tmp), call.secrets)
        files = []
        downloads = Path(tmp) / "downloads"
        if downloads.is_dir():
            for path in sorted(downloads.iterdir()):
                if path.is_symlink():
                    raise FileOutputError(f"downloads/: refusing symlink '{path.name}'.")
                files.append(Dir(path) if path.is_dir() else File(path))  # bytes read now
    if result.get("text") is None:
        raise RuntimeError(
            f"browser_use '{call.spec.name}' did not finish ({result.get('status')}): "
            f"{result.get('error') or 'no answer'}"
        )
    text = result["text"]
    return Output(text, files=files) if files else text


async def _exec_agent(task: str, options: dict, workspace: Path, secrets=None) -> dict:
    """Run browser-use in its own environment and return its answer — a seam for tests."""
    import asyncio
    import json

    from .browser_use_runner import MARKER

    request = dict(options, task=task, workspace=str(workspace))
    env = dict(os.environ)
    # Same hygiene as the default LiteLLM backend: no telemetry (browser-use's
    # includes the URLs visited) unless the user turned it on themselves.
    env.setdefault("ANONYMIZED_TELEMETRY", "false")
    for name in secrets or ():  # run(secrets=...) API keys reach the provider SDKs
        env[name] = secrets[name]
    proc = await asyncio.create_subprocess_exec(
        *runner_command(),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    out, err = await proc.communicate(json.dumps(request).encode())
    for line in out.decode(errors="replace").splitlines():
        if line.startswith(MARKER):
            return json.loads(line[len(MARKER):])
    tail = "\n".join(err.decode(errors="replace").strip().splitlines()[-20:])
    raise RuntimeError(f"browser-use exited ({proc.returncode}) without an answer:\n{tail}")


def runner_command() -> list[str]:
    """How to start browser_use_runner.py in a Python that has browser-use.

    ``PROMPTMAP_BROWSER_USE_PYTHON`` names one (a venv with browser-use
    installed); otherwise ``uv`` builds a cached, isolated one with the pinned
    browser-use.  It can't be promptmap's own environment: browser-use 1.0
    needs openai 3.x, which litellm 1.102.1 can't use.
    """
    import shutil

    # -P: the runner's folder is not put on sys.path, so this package's
    # modules can't shadow browser-use's imports.
    runner = [str(Path(__file__).with_name("browser_use_runner.py"))]
    python = os.environ.get("PROMPTMAP_BROWSER_USE_PYTHON")
    if python:
        return [python, "-P", *runner]
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError(
            "model_type='browser_use' runs browser-use in its own environment and needs uv "
            "(https://docs.astral.sh/uv/), or PROMPTMAP_BROWSER_USE_PYTHON set to a Python "
            f"with browser-use=={BROWSER_USE_VERSION} installed."
        )
    return [
        uv, "run", "--quiet", "--no-project", "--python", ">=3.11",
        "--with", f"browser-use=={BROWSER_USE_VERSION}", "python", "-P", *runner,
    ]


#: A browser agent, text in, text out.
KIND = register_model_kind(ModelKind(name=NAME, exec_fn=run_browser_use, config_keys=CONFIG_KEYS))
