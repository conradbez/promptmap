"""settings.yml + the default aisuite backend + `promptmap init/run --provider/--model`."""

from __future__ import annotations

import types
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from promptmap.cli import main
from promptmap.llm import resolve_llm_call
from promptmap.settings import provider_models, save_settings, supported_providers


@pytest.fixture
def fake_aisuite(monkeypatch):
    """Replace aisuite.Client with one that records calls and answers 'ok'."""
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        message = types.SimpleNamespace(content="ok")
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=message)],
            usage=types.SimpleNamespace(total_tokens=7),
        )

    import aisuite

    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    monkeypatch.setattr(aisuite, "Client", lambda *a, **k: client)
    return calls


@pytest.fixture(autouse=True)
def _no_provider_keys(monkeypatch):
    from promptmap.settings import PROVIDER_ENV_VARS

    for name, aliases in PROVIDER_ENV_VARS.values():
        for n in (name, *aliases):
            monkeypatch.delenv(n, raising=False)


def _project(tmp_path: Path) -> Path:
    (tmp_path / "map").mkdir()
    (tmp_path / "map" / "hello.prompt").write_text("Say hi", encoding="utf-8")
    return tmp_path


def test_snapshot_has_models_for_common_providers():
    for provider in ("anthropic", "openai", "gemini"):
        assert provider in supported_providers()
        assert provider_models(provider), provider


def test_settings_drive_the_default_backend(tmp_path, fake_aisuite):
    proj = _project(tmp_path)
    save_settings(proj, "anthropic", "claude-whatever")
    result = resolve_llm_call(str(proj / "map"))("hi")
    assert result.output == "ok" and result.spent_tokens == 7
    assert fake_aisuite[0]["model"] == "anthropic:claude-whatever"


def test_hand_edited_settings_are_not_validated(tmp_path, fake_aisuite):
    proj = _project(tmp_path)
    (proj / "settings.yml").write_text("provider: madeup\nmodel: not-a-model\n", encoding="utf-8")
    resolve_llm_call(str(proj / "map"))("hi")
    assert fake_aisuite[0]["model"] == "madeup:not-a-model"


def test_client_py_llm_call_overrides_settings(tmp_path, fake_aisuite):
    proj = _project(tmp_path)
    save_settings(proj, "openai", "gpt-4o-mini")
    (proj / "client.py").write_text("def llm_call(prompt):\n    return 'from client'\n", encoding="utf-8")
    assert resolve_llm_call(str(proj / "map"))("hi") == "from client"
    assert not fake_aisuite


def test_no_backend_error_points_at_init(tmp_path):
    with pytest.raises(FileNotFoundError, match="promptmap init"):
        resolve_llm_call(str(_project(tmp_path) / "map"))


def test_text_files_are_appended_to_the_prompt(tmp_path, fake_aisuite):
    proj = _project(tmp_path)
    save_settings(proj, "openai", "m")
    doc = tmp_path / "notes.txt"
    doc.write_text("the notes", encoding="utf-8")
    with open(doc, "rb") as f:
        resolve_llm_call(str(proj / "map"))("Summarise", files=[f])
    content = fake_aisuite[0]["messages"][0]["content"]
    assert content.startswith("Summarise") and "notes.txt" in content and "the notes" in content


def test_init_writes_settings_without_client_py(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(main, ["init", "proj", "--provider", "anthropic", "--model", "claude-x"])
    assert result.exit_code == 0, result.output
    assert yaml.safe_load((tmp_path / "proj" / "settings.yml").read_text()) == {
        "provider": "anthropic", "model": "claude-x",
    }
    assert not (tmp_path / "proj" / "client.py").exists()


def test_init_rejects_unknown_provider(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(main, ["init", "proj", "--provider", "nope", "--model", "m"])
    assert result.exit_code == 1
    assert "unknown provider" in result.output


def test_init_asks_for_provider_and_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("promptmap.settings.is_interactive", lambda: True)
    # "anthropic" by name, then a model the snapshot doesn't know, confirmed.
    result = CliRunner().invoke(main, ["init", "proj"], input="anthropic\nmy-custom-model\ny\nsk-test\n")
    assert result.exit_code == 0, result.output
    settings = yaml.safe_load((tmp_path / "proj" / "settings.yml").read_text())
    assert settings == {"provider": "anthropic", "model": "my-custom-model"}
    assert (tmp_path / "proj" / ".env").read_text() == "ANTHROPIC_API_KEY=sk-test\n"
    assert ".env" in (tmp_path / "proj" / ".gitignore").read_text().splitlines()


def test_init_model_picked_by_number(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("promptmap.settings.is_interactive", lambda: True)
    result = CliRunner().invoke(main, ["init", "proj", "--provider", "openai"], input="1\n\n")
    assert result.exit_code == 0, result.output
    settings = yaml.safe_load((tmp_path / "proj" / "settings.yml").read_text())
    assert settings["model"] == provider_models("openai")[0]


def test_run_provider_and_model_flags_save_settings(tmp_path, monkeypatch, fake_aisuite):
    proj = _project(tmp_path)
    monkeypatch.chdir(proj)
    result = CliRunner().invoke(main, ["run", "--provider", "anything", "--model", "goes"])
    assert result.exit_code == 0, result.output
    assert yaml.safe_load((proj / "settings.yml").read_text()) == {"provider": "anything", "model": "goes"}
    assert fake_aisuite[0]["model"] == "anything:goes"


def test_run_asks_when_nothing_is_configured(tmp_path, monkeypatch, fake_aisuite):
    proj = _project(tmp_path)
    monkeypatch.chdir(proj)
    monkeypatch.setattr("promptmap.settings.is_interactive", lambda: True)
    result = CliRunner().invoke(main, ["run"], input="gemini\nsome-model\ny\n\n")
    assert result.exit_code == 0, result.output
    assert yaml.safe_load((proj / "settings.yml").read_text()) == {"provider": "gemini", "model": "some-model"}


def test_init_appends_key_placeholder_to_existing_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "proj").mkdir()
    (tmp_path / "proj" / ".env").write_text("OTHER=1", encoding="utf-8")
    result = CliRunner().invoke(main, ["init", "proj", "--provider", "gemini", "--model", "m"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "proj" / ".env").read_text() == "OTHER=1\nGEMINI_API_KEY=\n"


@pytest.mark.parametrize("existing", ["HF_TOKEN=abc\n", "export HUGGINGFACE_API_KEY=abc\n"])
def test_existing_key_or_alias_is_left_alone(tmp_path, existing):
    from promptmap.settings import ensure_env_key

    (tmp_path / ".env").write_text(existing, encoding="utf-8")
    assert ensure_env_key(tmp_path, "huggingface", ask=False) is None
    assert (tmp_path / ".env").read_text() == existing


def test_key_already_in_environment_is_not_written(tmp_path, monkeypatch):
    from promptmap.settings import ensure_env_key

    monkeypatch.setenv("OPENAI_API_KEY", "x")
    assert ensure_env_key(tmp_path, "openai", ask=False) is None
    assert not (tmp_path / ".env").exists()


def test_provider_without_known_key_writes_no_env(tmp_path):
    from promptmap.settings import ensure_env_key

    assert ensure_env_key(tmp_path, "ollama", ask=False) is None
    assert not (tmp_path / ".env").exists()
