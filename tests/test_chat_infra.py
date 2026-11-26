"""Core support for chat kinds (``ModelKind(chat=True)``).

The ``chat`` kind itself ships in promptmap[extras] and is tested there.
This covers what promptmap provides to any chat kind: the stored thread,
``chat_input`` / ``chat_reset``, commit-on-success, rename, ``get_chat()``,
``llm_messages`` (cached, real or flattened messages) and ``promptmap chat``.
"""

from __future__ import annotations

import asyncio

import pytest
from click.testing import CliRunner

import promptmap
from promptmap.chat_messages import flatten_messages
from promptmap.model_types import ModelKind, register_model_kind
from promptmap.storage import MemoryStorageBackend
from promptmap.storage.sqlite import SQLiteStorageBackend


async def _talk(rendered, call):
    """A minimal chat kind: request + whole thread + follow-up; always appends."""
    messages = [{"role": "user", "content": rendered}]
    messages += [{"role": t["role"], "content": t["content"]} for t in call.chat_turns]
    if call.chat_input:
        messages.append({"role": "user", "content": call.chat_input})
    reply = await call.llm_messages(messages)
    turns = ([{"role": "user", "content": call.chat_input}] if call.chat_input else [])
    call.stage_chat(turns + [{"role": "assistant", "content": reply}], None)
    return reply


@pytest.fixture(autouse=True)
def talk_kind(monkeypatch):
    """Register the test kinds for one test only, so other tests' registry is untouched."""
    from promptmap import model_types

    monkeypatch.setattr(model_types, "_REGISTRY", dict(model_types._REGISTRY))
    register_model_kind(ModelKind("talk_test", _talk, chat=True))

TALK = "{{ config(model_type='talk_test') }}Hello"


class MsgLLM:
    def __init__(self) -> None:
        self.sent: list = []

    def __call__(self, prompt: str, messages: list | None = None) -> str:
        self.sent.append(messages)
        return f"reply {len(self.sent)}"


@pytest.fixture(params=["memory", "sqlite"])
def storage(request, tmp_path):
    if request.param == "memory":
        return MemoryStorageBackend()
    return SQLiteStorageBackend(tmp_path / "promptmap.db")


def _run(storage, llm, models=None, **kwargs):
    return promptmap.run(
        models_from_dict=models or {"m": TALK}, storage_backend=storage, llm_call=llm,
        verbose=False, **kwargs,
    )


def _roles(storage, name="m"):
    return [(t["role"], t["content"]) for t in promptmap.get_chat(name, storage_backend=storage)]


def test_the_thread_and_follow_up_reach_the_kind(storage) -> None:
    llm = MsgLLM()
    _run(storage, llm)
    _run(storage, llm, chat_input={"m": "more"})
    assert _roles(storage) == [("assistant", "reply 1"), ("user", "more"), ("assistant", "reply 2")]
    assert llm.sent[1] == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "reply 1"},
        {"role": "user", "content": "more"},
    ]


def test_llm_messages_is_cached_on_the_conversation(storage) -> None:
    llm = MsgLLM()
    storage.init_db()
    storage.commit_chat_turns("m", "r0", [{"role": "assistant", "content": "x"}])
    _run(storage, llm)
    storage.reset_chat("m")
    storage.commit_chat_turns("m", "r0", [{"role": "assistant", "content": "x"}])
    _run(storage, llm)  # same conversation sent again
    assert len(llm.sent) == 1


def test_a_failed_model_commits_nothing(storage) -> None:
    def failing(prompt, messages=None):
        raise RuntimeError("down")

    result = _run(storage, failing, chat_input={"m": "more"})
    assert isinstance(result["m"], promptmap.ModelError)
    assert _roles(storage) == []


def test_a_failing_validator_commits_nothing(storage) -> None:
    from promptmap.executor.executor import execute_run
    from promptmap.executor.graph import build_models_from_dict

    def reject(prompt, output):
        raise ValueError("nope")

    storage.init_db()
    specs = build_models_from_dict({"m": TALK})
    results = asyncio.run(execute_run(
        storage.create_run(model_count=1), list(specs.values()), storage,
        llm_call=MsgLLM(), validators={"m": reject}, chat_input={"m": "more"},
    ))
    assert results[0].status == "error"
    assert _roles(storage) == []


def test_chat_input_names_are_checked(storage) -> None:
    models = {"m": TALK, "plain": "{{ config(model_type='template') }}x"}
    with pytest.raises(ValueError, match="no model 'nope'"):
        _run(storage, MsgLLM(), models=models, chat_input={"nope": "x"})
    with pytest.raises(ValueError, match="'plain' is not a chat model"):
        _run(storage, MsgLLM(), models=models, chat_reset=["plain"])
    with pytest.raises(ValueError, match="not part of this run"):
        _run(storage, MsgLLM(), models=models, chat_input={"m": "x"}, select=["plain"])


def test_chat_reset_and_rename(storage) -> None:
    llm = MsgLLM()
    _run(storage, llm)
    _run(storage, llm, chat_input={"m": "more"})
    _run(storage, llm, chat_reset=["m"])
    # The new thread sends just the request again: a cache hit on run 1.
    assert _roles(storage) == [("assistant", "reply 1")]
    assert len(promptmap.get_chat("m", thread=0, storage_backend=storage)) == 3

    storage.rename_model("m", "n")
    assert _roles(storage, "n") == [("assistant", "reply 1")]
    assert len(promptmap.get_chat("n", thread=0, storage_backend=storage)) == 3
    assert _roles(storage, "m") == []


def test_rename_moves_each_items_and_threads(storage) -> None:
    storage.init_db()
    storage.commit_chat_turns("m[0]", "r1", [{"role": "assistant", "content": "a"}])
    storage.reset_chat("m")
    storage.commit_chat_turns("m[0]", "r2", [{"role": "assistant", "content": "b"}])
    storage.rename_model("m", "n")
    assert [t["content"] for t in storage.get_chat_turns("n[0]")] == ["b"]
    assert [t["content"] for t in storage.get_chat_turns("n[0]", 0)] == ["a"]


def test_commit_can_replace_the_last_turn(storage) -> None:
    storage.init_db()
    storage.commit_chat_turns("m", "r1", [{"role": "assistant", "content": "a"}])
    storage.commit_chat_turns("m", "r2", [], replace_last="b")
    assert [(t["content"], t["run_id"]) for t in storage.get_chat_turns("m")] == [("b", "r2")]


def test_a_prompt_only_llm_call_gets_the_conversation_flattened(storage) -> None:
    prompts: list[str] = []

    def llm(prompt: str) -> str:
        prompts.append(prompt)
        return "ok"

    _run(storage, llm)
    _run(storage, llm, chat_input={"m": "more"})
    assert prompts[1] == "Hello\n\n[your earlier reply]\nok\n[/your earlier reply]\n\n[follow-up 1]\nmore\n[/follow-up 1]"


def test_flatten_with_a_system_message() -> None:
    text = flatten_messages([
        {"role": "system", "content": "S"}, {"role": "user", "content": "R"},
        {"role": "assistant", "content": "a"}, {"role": "user", "content": "f"},
    ])
    assert text == "S\n\nR\n\n[your earlier reply]\na\n[/your earlier reply]\n\n[follow-up 1]\nf\n[/follow-up 1]"


def test_litellm_sends_real_messages(monkeypatch) -> None:
    from promptmap import llm as llm_module

    captured: dict = {}

    class Fake:
        def completion(self, **kwargs):
            captured.update(kwargs)

            class R:
                choices = [type("C", (), {"message": type("M", (), {"content": "ok"})()})()]
                usage = None
            return R()

    monkeypatch.setattr(llm_module, "quiet_litellm", lambda: Fake())
    call = llm_module.litellm_llm_call("openai", "gpt-x")
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    call("flattened", messages=messages)
    assert captured["messages"] == messages


def test_a_kind_without_chat_gets_no_chat_fields(storage) -> None:
    seen = {}

    async def plain(rendered, call):
        seen["turns"], seen["input"] = call.chat_turns, call.chat_input
        return "x"

    register_model_kind(ModelKind("plain_test", plain))
    _run(storage, MsgLLM(), models={"p": "{{ config(model_type='plain_test') }}x"})
    assert seen == {"turns": (), "input": None}


def test_promptmap_chat_cli(tmp_path, monkeypatch) -> None:
    import promptmap.db as db
    from promptmap.cli import main

    monkeypatch.chdir(tmp_path)
    (tmp_path / "map").mkdir()
    (tmp_path / "client.py").write_text(
        "def llm_call(prompt, messages=None):\n"
        "    return 'reply to: ' + messages[-1]['content']\n"
    )
    (tmp_path / "map" / "m.prompt").write_text(TALK)
    (tmp_path / "map" / "plain.prompt").write_text("Hi")
    monkeypatch.setattr(db, "_DEFAULT_BACKEND", SQLiteStorageBackend(tmp_path / ".promptmap" / "p.db"))
    runner = CliRunner()

    assert "reply to: Hello" in runner.invoke(main, ["chat", "m"]).output
    assert "reply to: more" in runner.invoke(main, ["chat", "m", "more"]).output
    history = runner.invoke(main, ["chat", "m", "--history"]).output
    assert "follow-up 1" in history and "more" in history

    assert runner.invoke(main, ["chat", "m", "--new"]).exit_code == 0
    assert "follow-up" not in runner.invoke(main, ["chat", "m", "--history"]).output

    wrong = runner.invoke(main, ["chat", "plain"])
    assert wrong.exit_code == 1 and "not a chat model" in wrong.output

    assert "new conversation" in runner.invoke(main, ["clear-cache", "m", "--chat"]).output
    assert "No conversation yet" in runner.invoke(main, ["chat", "m", "--history"]).output
