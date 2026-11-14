"""model_type="browser_use": option resolution, aliases, caching, the runner protocol."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from promptmap.model_spec import ModelSpec
from promptmap.model_types import ModelCall

from promptmap_extra_model_kinds import browser_use_exec
from promptmap_extra_model_kinds.browser_use_exec import pydantic_ai_model, resolve_options

from conftest import run_models


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in ("PROMPTMAP_BROWSER_HEADLESS", "PROMPTMAP_BROWSER_CLOUD"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DISPLAY", ":0")  # a display, so the default window is kept


def _project(tmp_path: Path, settings: dict | None = None) -> Path:
    (tmp_path / "map").mkdir()
    path = tmp_path / "map" / "shop.prompt"
    path.write_text("Find it", encoding="utf-8")
    if settings is not None:
        (tmp_path / "settings.yml").write_text(yaml.safe_dump(settings), encoding="utf-8")
    return path


def _call(path: Path, **config) -> ModelCall:
    spec = ModelSpec(name="shop", source="", path=path, model_type="browser_use",
                     config={k: str(v) for k, v in config.items()})
    return ModelCall(spec=spec, outputs={}, llm=None, compute=None)


SETTINGS = {
    "provider": "anthropic", "model": "claude-sonnet-x",
    "aliases": {"low": "openai/gpt-mini", "fast": {"provider": "gemini", "model": "gemini-flash"}},
}


def test_defaults_are_local_visible_and_settings_model(tmp_path):
    opts = resolve_options(_call(_project(tmp_path, SETTINGS)))
    assert opts == {"llm": "anthropic:claude-sonnet-x", "cloud": False, "headless": False,
                    "max_steps": 100, "allowed_domains": None}


def test_config_switches_to_headless_and_cloud(tmp_path):
    path = _project(tmp_path, SETTINGS)
    assert resolve_options(_call(path, browser_headless=True))["headless"] is True
    cloud = resolve_options(_call(path, browser_cloud=True))
    assert cloud["cloud"] is True and cloud["headless"] is None


def test_settings_block_sets_project_defaults_and_config_wins(tmp_path):
    path = _project(tmp_path, {**SETTINGS, "browser_use": {"headless": True, "model": "low", "max_steps": 7}})
    opts = resolve_options(_call(path))
    assert (opts["headless"], opts["llm"], opts["max_steps"]) == (True, "openai:gpt-mini", 7)
    assert resolve_options(_call(path, browser_headless=False))["headless"] is False


def test_env_var_sets_headless_below_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMPTMAP_BROWSER_HEADLESS", "1")
    assert resolve_options(_call(_project(tmp_path, SETTINGS)))["headless"] is True


def test_no_display_runs_headless(tmp_path, monkeypatch):
    monkeypatch.delenv("DISPLAY")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(browser_use_exec.sys, "platform", "linux")
    assert resolve_options(_call(_project(tmp_path, SETTINGS)))["headless"] is True


def test_bad_flag_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="browser_headless"):
        resolve_options(_call(_project(tmp_path, SETTINGS), browser_headless="maybe"))


def test_allowed_domains_from_config_list(tmp_path):
    path = _project(tmp_path, SETTINGS)
    spec_config = {"browser_allowed_domains": json.dumps(["a.com", "*.b.org"])}
    call = ModelCall(spec=ModelSpec(name="s", source="", path=path, config=spec_config),
                     outputs={}, llm=None, compute=None)
    assert resolve_options(call)["allowed_domains"] == ["a.com", "*.b.org"]


@pytest.mark.parametrize("name, expected", [
    ("low", "openai:gpt-mini"),                  # alias
    ("fast", "google-gla:gemini-flash"),         # alias, litellm provider renamed
    ("groq/llama-3", "groq:llama-3"),            # provider/model as in settings.yml
    ("openai:gpt-4o", "openai:gpt-4o"),          # already a Pydantic AI string
    ("claude-opus-x", "anthropic:claude-opus-x"),  # bare: settings.yml's provider
])
def test_model_names(name, expected):
    assert pydantic_ai_model(name, SETTINGS) == expected


def test_no_model_anywhere_leaves_browser_use_default(tmp_path):
    assert resolve_options(_call(_project(tmp_path)))["llm"] is None


def test_secret_key_is_handed_to_the_chat_model(monkeypatch):
    import types

    made = []
    fake = types.SimpleNamespace(ChatAnthropic=lambda model, api_key: made.append((model, api_key)) or "chat")
    monkeypatch.setitem(sys.modules, "browser_use", fake)
    from promptmap.secrets import Secrets

    assert browser_use_exec._chat_model("anthropic:claude-x", Secrets({"ANTHROPIC_API_KEY": "k"})) == "chat"
    assert made == [("claude-x", "k")]
    # No secret: the model string, and browser-use reads the key from the environment.
    assert browser_use_exec._chat_model("anthropic:claude-x", Secrets({})) == "anthropic:claude-x"
    assert browser_use_exec._chat_model("groq:llama", Secrets({"GROQ_API_KEY": "k"})) == "groq:llama"


def test_runs_and_caches(monkeypatch):
    seen = []

    async def fake(task, options, workspace, secrets=None):
        seen.append((task, options))
        (workspace / "downloads").mkdir()
        (workspace / "downloads" / "receipt.txt").write_text("paid")
        return {"text": "found it", "status": "completed", "error": None}

    monkeypatch.setattr(browser_use_exec, "_exec_agent", fake)
    models = {"shop": '{{ config(model_type="browser_use", browser_headless=True) }}Buy milk'}
    storage, _, results = run_models(models)
    out = results["shop"].value
    assert out.text == "found it"
    assert out.files["receipt.txt"].read_bytes() == b"paid"
    assert seen[0][0] == "Buy milk" and seen[0][1]["headless"] is True
    run_models(models, storage=storage)
    assert len(seen) == 1  # second run served from the cache


def test_unfinished_run_fails_the_model(monkeypatch):
    async def fake(task, options, workspace, secrets=None):
        return {"text": None, "status": "error", "error": "ValueError: no key"}

    monkeypatch.setattr(browser_use_exec, "_exec_agent", fake)
    _, _, results = run_models({"shop": '{{ config(model_type="browser_use") }}Buy milk'})
    assert "no key" in (results["shop"].error or "")
