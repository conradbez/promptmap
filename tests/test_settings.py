"""settings.yml + the default LiteLLM backend + `promptmap init/run --provider/--model`."""

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
def fake_litellm(monkeypatch):
    """Replace litellm.completion with one that records calls and answers 'ok'."""
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        message = types.SimpleNamespace(content="ok")
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=message)],
            usage=types.SimpleNamespace(total_tokens=7),
        )

    import litellm

    monkeypatch.setattr(litellm, "completion", create)
    return calls


@pytest.fixture(autouse=True)
def _no_provider_keys(monkeypatch):
    for n in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
              "GROQ_API_KEY", "HUGGINGFACE_API_KEY", "HF_TOKEN", "AZURE_API_KEY",
              "AZURE_API_BASE", "AZURE_API_VERSION"):
        monkeypatch.delenv(n, raising=False)


def _project(tmp_path: Path) -> Path:
    (tmp_path / "map").mkdir()
    (tmp_path / "map" / "hello.prompt").write_text("Say hi", encoding="utf-8")
    return tmp_path


def test_snapshot_has_models_for_common_providers():
    for provider in ("anthropic", "openai", "gemini"):
        assert provider in supported_providers()
        assert provider_models(provider), provider


def test_settings_drive_the_default_backend(tmp_path, fake_litellm):
    proj = _project(tmp_path)
    save_settings(proj, "anthropic", "claude-whatever")
    result = resolve_llm_call(str(proj / "map"))("hi")
    assert result.output == "ok" and result.spent_tokens == 7
    assert fake_litellm[0]["model"] == "anthropic/claude-whatever"


def test_hand_edited_settings_are_not_validated(tmp_path, fake_litellm):
    proj = _project(tmp_path)
    (proj / "settings.yml").write_text("provider: madeup\nmodel: not-a-model\n", encoding="utf-8")
    resolve_llm_call(str(proj / "map"))("hi")
    assert fake_litellm[0]["model"] == "madeup/not-a-model"


def test_client_py_llm_call_overrides_settings(tmp_path, fake_litellm):
    proj = _project(tmp_path)
    save_settings(proj, "openai", "gpt-4o-mini")
    (proj / "client.py").write_text("def llm_call(prompt):\n    return 'from client'\n", encoding="utf-8")
    assert resolve_llm_call(str(proj / "map"))("hi") == "from client"
    assert not fake_litellm


def test_no_backend_error_points_at_init(tmp_path):
    with pytest.raises(FileNotFoundError, match="promptmap init"):
        resolve_llm_call(str(_project(tmp_path) / "map"))


def test_text_files_are_appended_to_the_prompt(tmp_path, fake_litellm):
    proj = _project(tmp_path)
    save_settings(proj, "openai", "m")
    doc = tmp_path / "notes.txt"
    doc.write_text("the notes", encoding="utf-8")
    with open(doc, "rb") as f:
        resolve_llm_call(str(proj / "map"))("Summarise", files=[f])
    blocks = fake_litellm[0]["messages"][0]["content"]
    assert blocks[0] == {"type": "text", "text": "Summarise"}
    assert blocks[1] == {"type": "text", "text": "--- notes.txt ---\nthe notes"}


def test_binary_files_become_data_blocks(tmp_path, fake_litellm):
    proj = _project(tmp_path)
    save_settings(proj, "openai", "m")
    (tmp_path / "pic.png").write_bytes(b"\x89PNG\r\n\x1a\n\xff")
    (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4\n\xff\xfe")
    with open(tmp_path / "pic.png", "rb") as png, open(tmp_path / "doc.pdf", "rb") as pdf:
        resolve_llm_call(str(proj / "map"))("Describe", files=[png, pdf])
    _, image, document = fake_litellm[0]["messages"][0]["content"]
    assert image["type"] == "image_url" and image["image_url"]["url"].startswith("data:image/png;base64,")
    assert document["type"] == "file" and document["file"]["file_data"].startswith("data:application/pdf;base64,")


def test_prefixed_model_is_not_double_prefixed(tmp_path, fake_litellm):
    proj = _project(tmp_path)
    save_settings(proj, "openrouter", "openrouter/anthropic/claude-x")
    resolve_llm_call(str(proj / "map"))("hi")
    assert fake_litellm[0]["model"] == "openrouter/anthropic/claude-x"


def test_init_accepts_litellm_provider_outside_the_menu(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(main, ["init", "proj", "--provider", "litellm_proxy", "--model", "m"])
    assert result.exit_code == 0, result.output


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
    result = CliRunner().invoke(main, ["init", "proj"], input="anthropic\nmy-custom-model\ny\ny\nsk-test\n")
    assert result.exit_code == 0, result.output
    settings = yaml.safe_load((tmp_path / "proj" / "settings.yml").read_text())
    assert settings == {"provider": "anthropic", "model": "my-custom-model"}
    assert (tmp_path / "proj" / ".env").read_text() == "ANTHROPIC_API_KEY=sk-test\n"
    assert ".env" in (tmp_path / "proj" / ".gitignore").read_text().splitlines()


def test_init_model_picked_by_number(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("promptmap.settings.is_interactive", lambda: True)
    result = CliRunner().invoke(main, ["init", "proj", "--provider", "openai"], input="1\nn\n")
    assert result.exit_code == 0, result.output
    settings = yaml.safe_load((tmp_path / "proj" / "settings.yml").read_text())
    assert settings["model"] == provider_models("openai")[0]


def test_run_provider_and_model_flags_save_settings(tmp_path, monkeypatch, fake_litellm):
    proj = _project(tmp_path)
    monkeypatch.chdir(proj)
    result = CliRunner().invoke(main, ["run", "--provider", "anything", "--model", "goes"])
    assert result.exit_code == 0, result.output
    assert yaml.safe_load((proj / "settings.yml").read_text()) == {"provider": "anything", "model": "goes"}
    assert fake_litellm[0]["model"] == "anything/goes"


def test_run_asks_when_nothing_is_configured(tmp_path, monkeypatch, fake_litellm):
    proj = _project(tmp_path)
    monkeypatch.chdir(proj)
    monkeypatch.setattr("promptmap.settings.is_interactive", lambda: True)
    result = CliRunner().invoke(main, ["run"], input="gemini\nsome-model\ny\nn\n")
    assert result.exit_code == 0, result.output
    assert yaml.safe_load((proj / "settings.yml").read_text()) == {"provider": "gemini", "model": "some-model"}


def test_init_appends_key_placeholder_to_existing_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "proj").mkdir()
    (tmp_path / "proj" / ".env").write_text("OTHER=1", encoding="utf-8")
    result = CliRunner().invoke(main, ["init", "proj", "--provider", "gemini", "--model", "m"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "proj" / ".env").read_text() == "OTHER=1\nGEMINI_API_KEY=\n"


def test_env_vars_come_from_litellm():
    from promptmap.settings import provider_env_vars

    assert provider_env_vars("groq", "llama-3.3-70b-versatile") == ["GROQ_API_KEY"]
    assert provider_env_vars("gemini", "anything") == ["GEMINI_API_KEY"]
    assert set(provider_env_vars("azure", "gpt")) == {"AZURE_API_KEY", "AZURE_API_BASE", "AZURE_API_VERSION"}
    assert provider_env_vars("madeup", "x") == []


def test_env_vars_listed_even_when_already_exported(monkeypatch):
    from promptmap.settings import provider_env_vars

    monkeypatch.setenv("GROQ_API_KEY", "x")
    assert provider_env_vars("groq", "m") == ["GROQ_API_KEY"]


def test_existing_value_in_env_file_is_left_alone(tmp_path):
    from promptmap.settings import ensure_env_keys

    (tmp_path / ".env").write_text("export ANTHROPIC_API_KEY=abc\n", encoding="utf-8")
    assert ensure_env_keys(tmp_path, "anthropic", "m", ask=False) == []
    assert (tmp_path / ".env").read_text() == "export ANTHROPIC_API_KEY=abc\n"


def test_key_already_in_environment_is_not_written(tmp_path, monkeypatch):
    from promptmap.settings import ensure_env_keys

    monkeypatch.setenv("OPENAI_API_KEY", "x")
    assert ensure_env_keys(tmp_path, "openai", "m", ask=False) == []
    assert not (tmp_path / ".env").exists()


def test_declining_leaves_a_blank_template_line(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("promptmap.settings.is_interactive", lambda: True)
    result = CliRunner().invoke(main, ["init", "proj", "--provider", "groq", "--model", "m"], input="n\n")
    assert result.exit_code == 0, result.output
    assert "fill it in there yourself later" in result.output
    assert (tmp_path / "proj" / ".env").read_text() == "GROQ_API_KEY=\n"


def test_blank_template_line_is_filled_in_later(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "proj").mkdir()
    (tmp_path / "proj" / ".env").write_text("A=1\nGROQ_API_KEY=\nB=2\n", encoding="utf-8")
    monkeypatch.setattr("promptmap.settings.is_interactive", lambda: True)
    result = CliRunner().invoke(main, ["init", "proj", "--provider", "groq", "--model", "m"], input="y\ngsk-1\n")
    assert result.exit_code == 0, result.output
    assert (tmp_path / "proj" / ".env").read_text() == "A=1\nGROQ_API_KEY=gsk-1\nB=2\n"


def test_default_backend_turns_litellm_logging_off(tmp_path, monkeypatch, fake_litellm):
    import litellm

    monkeypatch.setattr(litellm, "success_callback", ["langfuse"])
    monkeypatch.setattr(litellm, "failure_callback", ["langfuse"])
    monkeypatch.setattr(litellm, "turn_off_message_logging", False)
    proj = _project(tmp_path)
    save_settings(proj, "openai", "m")
    resolve_llm_call(str(proj / "map"))("hi")
    assert litellm.success_callback == [] and litellm.failure_callback == []
    assert litellm.turn_off_message_logging is True and litellm.telemetry is False


# ---------------------------------------------------------------------------
# Enter = blank, --no-interactive, and a flag for every prompt
# ---------------------------------------------------------------------------

def _settings_of(proj: Path) -> dict:
    return yaml.safe_load((proj / "settings.yml").read_text())


def test_enter_at_every_prompt_writes_blank_settings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("promptmap.settings.is_interactive", lambda: True)
    result = CliRunner().invoke(main, ["init", "proj"], input="\n\n\n\n")
    assert result.exit_code == 0, result.output
    assert _settings_of(tmp_path / "proj") == {"provider": "", "model": ""}
    assert not (tmp_path / "proj" / ".env").exists()


def test_enter_after_provider_leaves_model_and_key_blank(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("promptmap.settings.is_interactive", lambda: True)
    result = CliRunner().invoke(main, ["init", "proj"], input="groq\n\n\n\n")
    assert result.exit_code == 0, result.output
    assert _settings_of(tmp_path / "proj") == {"provider": "groq", "model": ""}
    assert (tmp_path / "proj" / ".env").read_text() == "GROQ_API_KEY=\n"


def test_no_interactive_asks_nothing_even_in_a_terminal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("promptmap.settings.is_interactive", lambda: True)
    result = CliRunner().invoke(main, ["init", "proj", "--no-interactive", "--provider", "groq"])
    assert result.exit_code == 0, result.output
    assert "?" not in result.output  # no prompt was shown
    assert _settings_of(tmp_path / "proj") == {"provider": "groq", "model": ""}
    assert (tmp_path / "proj" / ".env").read_text() == "GROQ_API_KEY=\n"


def test_nointeractive_spelling_works(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(main, ["init", "proj", "--nointeractive"])
    assert result.exit_code == 0, result.output
    assert _settings_of(tmp_path / "proj") == {"provider": "", "model": ""}


def test_every_prompt_has_a_flag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("promptmap.settings.is_interactive", lambda: True)
    result = CliRunner().invoke(main, [
        "init", "proj", "--provider", "azure", "--model", "my-deployment",
        "--api-key", "az-key", "--env", "AZURE_API_BASE=https://x", "--env", "AZURE_API_VERSION=2025-01-01",
    ])  # no input: any prompt would abort
    assert result.exit_code == 0, result.output
    assert _settings_of(tmp_path / "proj") == {"provider": "azure", "model": "my-deployment"}
    env = (tmp_path / "proj" / ".env").read_text().splitlines()
    assert sorted(env) == ["AZURE_API_BASE=https://x", "AZURE_API_KEY=az-key", "AZURE_API_VERSION=2025-01-01"]


def test_api_key_goes_to_the_providers_variable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(main, ["init", "proj", "--provider", "gemini", "--model", "m", "--api-key", "g-1"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "proj" / ".env").read_text() == "GEMINI_API_KEY=g-1\n"


def test_api_key_without_known_variable_points_at_env_flag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(main, ["init", "proj", "--no-interactive", "--api-key", "k"])
    assert result.exit_code == 1
    assert "--env" in result.output and "NAME=VALUE" in result.output


def test_bad_env_flag_is_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(main, ["init", "proj", "--env", "NOEQUALS"])
    assert result.exit_code == 1


def test_run_no_interactive_never_prompts(tmp_path, monkeypatch):
    proj = _project(tmp_path)
    monkeypatch.chdir(proj)
    monkeypatch.setattr("promptmap.settings.is_interactive", lambda: True)
    result = CliRunner().invoke(main, ["run", "--no-interactive"])
    assert result.exit_code == 1
    assert "No LLM configured" in result.output
    assert not (proj / "settings.yml").exists()


def test_run_keeps_current_model_when_only_api_key_given(tmp_path, monkeypatch, fake_litellm):
    proj = _project(tmp_path)
    monkeypatch.chdir(proj)
    save_settings(proj, "groq", "llama")
    result = CliRunner().invoke(main, ["run", "--api-key", "gsk-1"])
    assert result.exit_code == 0, result.output
    assert _settings_of(proj) == {"provider": "groq", "model": "llama"}
    assert (proj / ".env").read_text() == "GROQ_API_KEY=gsk-1\n"


def test_filter_matches_every_word_in_any_order():
    from promptmap.settings import filter_options

    options = ["anthropic/claude-sonnet-4.5", "anthropic/claude-opus-4", "openai/gpt-4o"]
    assert filter_options(options, "claude sonnet") == ["anthropic/claude-sonnet-4.5"]
    assert filter_options(options, "SONNET claude") == ["anthropic/claude-sonnet-4.5"]
    assert filter_options(options, "claude") == options[:2]
    assert filter_options(options, "claude gpt") == []


def test_multi_word_filter_then_pick_by_number(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("promptmap.settings.is_interactive", lambda: True)
    result = CliRunner().invoke(main, ["init", "proj"], input="openrouter\nclaude sonnet\n1\n\n")
    assert result.exit_code == 0, result.output
    assert "matches for 'claude sonnet'" in result.output
    model = _settings_of(tmp_path / "proj")["model"]
    assert "claude" in model and "sonnet" in model


def test_unmatched_text_is_not_used_on_enter(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("promptmap.settings.is_interactive", lambda: True)
    # "zzzz" matches nothing; Enter declines using it; then Enter leaves model blank.
    result = CliRunner().invoke(main, ["init", "proj"], input="openrouter\nzzzz\n\n\n\n")
    assert result.exit_code == 0, result.output
    assert _settings_of(tmp_path / "proj")["model"] == ""


def test_single_match_is_offered_directly(tmp_path, monkeypatch):
    from promptmap.settings import filter_options

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("promptmap.settings.is_interactive", lambda: True)
    models = provider_models("anthropic")
    unique = next(m for m in models if len(filter_options(models, m)) == 1)
    # Upper-cased so it isn't an exact name: goes through the filter, one match, Enter accepts.
    result = CliRunner().invoke(main, ["init", "proj", "--provider", "anthropic"], input=f"{unique.upper()}\n\n\n")
    assert result.exit_code == 0, result.output
    assert _settings_of(tmp_path / "proj")["model"] == unique


# ---------------------------------------------------------------------------
# Live filter prompt (real terminals) — driven with prompt_toolkit pipe input
# ---------------------------------------------------------------------------

def _live(monkeypatch, keys: str, label="Model", options=None, allow_custom=True, accept=None):
    import questionary
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from promptmap import settings

    real = questionary.autocomplete
    options = options or ["anthropic/claude-sonnet-4.5", "anthropic/claude-opus-4", "openai/gpt-4o"]
    import threading
    import time

    def typist(pipe):
        # One chunk per "|" with a pause, like a person: completions are
        # computed in the background and need a moment to appear.
        for chunk in keys.split("|"):
            time.sleep(0.15)
            pipe.send_text(chunk)

    with create_pipe_input() as pipe:
        threading.Thread(target=typist, args=(pipe,), daemon=True).start()
        monkeypatch.setattr(questionary, "autocomplete",
                            lambda *a, **k: real(*a, input=pipe, output=DummyOutput(), **k))
        return settings._live_choose(label, options, allow_custom, accept)


def test_live_filter_tab_picks_first_match(monkeypatch):
    assert _live(monkeypatch, "sonnet claude|\t|\r") == "anthropic/claude-sonnet-4.5"


def test_live_filter_second_tab_moves_to_next_match(monkeypatch):
    assert _live(monkeypatch, "claude|\t|\t|\r") == "anthropic/claude-opus-4"


def test_live_enter_on_empty_is_blank(monkeypatch):
    assert _live(monkeypatch, "\r") == ""


def test_live_custom_name_is_kept_as_typed(monkeypatch):
    assert _live(monkeypatch, "my-future-model\r") == "my-future-model"


def test_live_provider_rejects_unknown_then_takes_a_listed_one(monkeypatch):
    # "zzz" + Enter is refused; Ctrl-U clears the line; "openai" is accepted.
    got = _live(monkeypatch, "zzz|\r|\x15|openai|\r", label="Provider",
                options=["anthropic", "openai"], allow_custom=False, accept=lambda p: False)
    assert got == "openai"


def test_live_enter_on_one_match_takes_it(monkeypatch):
    assert _live(monkeypatch, "gpt|\r") == "openai/gpt-4o"


def test_live_enter_on_several_matches_is_refused_until_picked(monkeypatch):
    # "claude" matches two: Enter refused; Tab picks the first; Enter accepts.
    assert _live(monkeypatch, "claude|\r|\t|\r") == "anthropic/claude-sonnet-4.5"
