"""``model_type="chat"``: an answer refined over several turns."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

import promptmap
from promptmap.executor.chat import DEFAULT_CHAT_INSTRUCTION, build_messages, flatten_messages
from promptmap.storage import MemoryStorageBackend
from promptmap.storage.sqlite import SQLiteStorageBackend


class ChatLLM:
    """Takes ``messages``; answers with a counter and remembers what it was sent."""

    def __init__(self) -> None:
        self.sent: list[list[dict]] = []

    def __call__(self, prompt: str, messages: list[dict] | None = None) -> str:
        self.sent.append(messages)
        return f"reply {len(self.sent)}"


@pytest.fixture(params=["memory", "sqlite"])
def storage(request, tmp_path):
    if request.param == "memory":
        return MemoryStorageBackend()
    return SQLiteStorageBackend(tmp_path / "promptmap.db")


POEM = "{{ config(model_type='chat') }}Write a poem about {{ ref('topic') }}."


def _models(topic: str = "cats", poem: str = POEM) -> dict[str, str]:
    return {"topic": "{{ config(model_type='template') }}" + topic, "poem": poem}


def _run(storage, llm, models=None, **kwargs) -> dict:
    return promptmap.run(
        models_from_dict=models or _models(), storage_backend=storage, llm_call=llm,
        verbose=False, **kwargs,
    )


def _chat(storage, model: str = "poem") -> list[tuple[str, str]]:
    return [(t["role"], t["content"]) for t in promptmap.get_chat(model, storage_backend=storage)]


def _without_system(messages: list[dict]) -> list[tuple[str, str]]:
    return [(m["role"], m["content"]) for m in messages if m["role"] != "system"]


# -- the conversation ---------------------------------------------------------


def test_turns_build_up_across_runs(storage) -> None:
    llm = ChatLLM()
    assert _run(storage, llm)["poem"] == "reply 1"
    assert _run(storage, llm, chat_input={"poem": "Shorter"})["poem"] == "reply 2"
    assert _run(storage, llm, chat_input={"poem": "Rhyme it"})["poem"] == "reply 3"

    assert _chat(storage) == [
        ("assistant", "reply 1"),
        ("user", "Shorter"), ("assistant", "reply 2"),
        ("user", "Rhyme it"), ("assistant", "reply 3"),
    ]
    assert _without_system(llm.sent[2]) == [
        ("user", "Write a poem about cats."),
        ("assistant", "reply 1"),
        ("user", "Shorter"),
        ("assistant", "reply 2"),
        ("user", "Rhyme it"),
    ]


def test_the_chat_instruction_is_the_system_message(storage) -> None:
    llm = ChatLLM()
    _run(storage, llm)
    assert llm.sent[0][0] == {"role": "system", "content": DEFAULT_CHAT_INSTRUCTION}

    custom = "{{ config(model_type='chat', chat_instruction='Be brief.') }}Hi"
    _run(storage, llm, models={"poem": custom})
    assert llm.sent[1][0] == {"role": "system", "content": "Be brief."}

    off = "{{ config(model_type='chat', chat_instruction=false) }}Hi again"
    _run(storage, llm, models={"poem": off})
    assert llm.sent[2][0]["role"] == "user"


def test_a_rerun_without_a_follow_up_replaces_the_last_reply(storage) -> None:
    llm = ChatLLM()
    _run(storage, llm)
    _run(storage, llm, chat_input={"poem": "Shorter"})
    _run(storage, llm, models=_models("dogs"))  # upstream change, no follow-up

    assert _chat(storage) == [
        ("assistant", "reply 1"), ("user", "Shorter"), ("assistant", "reply 3"),
    ]
    # The last reply is left out of what is sent, since it is being redone.
    assert _without_system(llm.sent[2]) == [
        ("user", "Write a poem about dogs."), ("assistant", "reply 1"), ("user", "Shorter"),
    ]


def test_an_upstream_change_reaches_the_request_and_keeps_earlier_replies(storage) -> None:
    llm = ChatLLM()
    _run(storage, llm)
    _run(storage, llm, models=_models("dogs"), chat_input={"poem": "Shorter"})

    assert _without_system(llm.sent[1])[:2] == [
        ("user", "Write a poem about dogs."), ("assistant", "reply 1"),
    ]
    assert _chat(storage)[0] == ("assistant", "reply 1")


def test_a_rerun_with_nothing_changed_is_a_cache_hit(storage) -> None:
    llm = ChatLLM()
    _run(storage, llm)
    _run(storage, llm, chat_input={"poem": "Shorter"})
    assert _run(storage, llm)["poem"] == "reply 2"

    assert len(llm.sent) == 2
    assert _chat(storage) == [("assistant", "reply 1"), ("user", "Shorter"), ("assistant", "reply 2")]


def test_a_failed_run_records_no_turns(storage) -> None:
    llm = ChatLLM()
    _run(storage, llm)

    def failing(prompt, messages=None):
        raise RuntimeError("down")

    result = _run(storage, failing, chat_input={"poem": "Shorter"})
    assert isinstance(result["poem"], promptmap.ModelError)
    assert _chat(storage) == [("assistant", "reply 1")]


def test_a_failing_validator_records_no_turns(storage, tmp_path) -> None:
    from promptmap.executor.executor import execute_run
    from promptmap.executor.graph import build_models_from_dict
    import asyncio

    llm = ChatLLM()
    _run(storage, llm)

    def reject(prompt, output):
        raise ValueError("nope")

    specs = build_models_from_dict(_models())
    run_id = storage.create_run(model_count=2)
    results = asyncio.run(execute_run(
        run_id, list(specs.values()), storage, llm_call=llm,
        validators={"poem": reject}, chat_input={"poem": "Shorter"},
    ))
    assert {r.model_name: r.status for r in results}["poem"] == "error"
    assert _chat(storage) == [("assistant", "reply 1")]


def test_downstream_models_see_only_the_latest_reply(storage) -> None:
    llm = ChatLLM()
    models = {**_models(), "title": "{{ config(model_type='template') }}Title: {{ ref('poem') }}"}
    _run(storage, llm, models=models)
    result = _run(storage, llm, models=models, chat_input={"poem": "Shorter"})
    assert result["title"] == "Title: reply 2"


def test_a_plain_llm_call_gets_the_conversation_flattened(storage) -> None:
    prompts: list[str] = []

    def llm(prompt: str) -> str:
        prompts.append(prompt)
        return f"reply {len(prompts)}"

    _run(storage, llm)
    _run(storage, llm, chat_input={"poem": "Shorter"})

    assert prompts[0] == DEFAULT_CHAT_INSTRUCTION + "\n\nWrite a poem about cats."
    assert prompts[1] == (
        DEFAULT_CHAT_INSTRUCTION + "\n\nWrite a poem about cats.\n\n"
        "[your earlier reply]\nreply 1\n[/your earlier reply]\n\n"
        "[follow-up 1]\nShorter\n[/follow-up 1]"
    )


def test_json_output_parses_the_latest_reply_and_sends_earlier_ones_as_text(storage) -> None:
    sent: list[list[dict]] = []

    def llm(prompt, messages=None):
        sent.append(messages)
        return json.dumps({"lines": len(sent)})

    source = "{{ config(model_type='chat', output_format='json') }}A poem as JSON."
    _run(storage, llm, models={"poem": source})
    result = _run(storage, llm, models={"poem": source}, chat_input={"poem": "Longer"})

    assert json.loads(result["poem"]) == {"lines": 2}
    assert _without_system(sent[1])[1] == ("assistant", '{"lines": 1}')


def test_max_turns_keeps_only_the_latest_turns(storage) -> None:
    llm = ChatLLM()
    source = "{{ config(model_type='chat', max_turns=2) }}Hi"
    for follow_up in (None, "one", "two"):
        _run(storage, llm, models={"poem": source}, chat_input={"poem": follow_up} if follow_up else None)

    # Thread before the last run: reply1, one, reply2.  Two turns kept would
    # start on "one" (a follow-up), so the window starts at reply 2.
    assert _without_system(llm.sent[2]) == [("user", "Hi"), ("assistant", "reply 2"), ("user", "two")]
    assert len(_chat(storage)) == 5  # storage keeps everything


def test_each_item_keeps_its_own_conversation(storage) -> None:
    llm = ChatLLM()
    models = {
        "topics": "{{ config(model_type='template', output_format='json') }}[\"cats\", \"dogs\"]",
        "poem": "{{ config(model_type='chat', each='topics[*]') }}Poem about {{ ref('topics') }}.",
    }
    _run(storage, llm, models=models)
    _run(storage, llm, models=models, chat_input={"poem[1]": "Shorter"})

    assert [r for r, _ in _chat(storage, "poem[0]")] == ["assistant"]
    assert [r for r, _ in _chat(storage, "poem[1]")] == ["assistant", "user", "assistant"]
    assert _chat(storage, "poem[1]")[1] == ("user", "Shorter")


def test_chat_reset_starts_a_new_thread_and_keeps_the_old_one(storage) -> None:
    llm = ChatLLM()
    _run(storage, llm)
    _run(storage, llm, chat_input={"poem": "Shorter"})
    _run(storage, llm, models=_models("dogs"), chat_reset={"poem"})

    assert _chat(storage) == [("assistant", "reply 3")]
    assert len(promptmap.get_chat("poem", thread=0, storage_backend=storage)) == 3
    assert _without_system(llm.sent[2]) == [("user", "Write a poem about dogs.")]


def test_rename_keeps_the_conversation(storage) -> None:
    llm = ChatLLM()
    _run(storage, llm)
    _run(storage, llm, chat_input={"poem": "Shorter"})
    storage.rename_model("poem", "verse")

    assert _chat(storage, "verse") == [("assistant", "reply 1"), ("user", "Shorter"), ("assistant", "reply 2")]
    assert _chat(storage, "poem") == []


def test_rename_moves_each_items_and_reset_threads(storage) -> None:
    storage.init_db()
    storage.commit_chat_turns("poem[0]", "r1", [{"role": "assistant", "content": "a"}])
    storage.reset_chat("poem")
    storage.commit_chat_turns("poem[0]", "r2", [{"role": "assistant", "content": "b"}])
    storage.rename_model("poem", "verse")

    assert [t["content"] for t in storage.get_chat_turns("verse[0]")] == ["b"]
    assert [t["content"] for t in storage.get_chat_turns("verse[0]", 0)] == ["a"]


def test_chat_input_must_name_a_chat_model_in_the_run(storage) -> None:
    llm = ChatLLM()
    with pytest.raises(ValueError, match="no model 'pome'"):
        _run(storage, llm, chat_input={"pome": "x"})
    with pytest.raises(ValueError, match="'topic' is not a chat model"):
        _run(storage, llm, chat_input={"topic": "x"})
    with pytest.raises(ValueError, match="not part of this run"):
        _run(storage, llm, chat_input={"poem": "x"}, select=["topic"])


def test_a_plain_run_never_adds_turns(storage) -> None:
    llm = ChatLLM()
    for _ in range(3):
        _run(storage, llm, models=_models("cats"))
        _run(storage, llm, models=_models("dogs"))
    assert [r for r, _ in _chat(storage)] == ["assistant"]


# -- the pieces -----------------------------------------------------------------


def test_build_messages_follow_up_on_an_empty_thread() -> None:
    messages, replace = build_messages("Req", [], "First", instruction=None)
    assert messages == [{"role": "user", "content": "Req"}, {"role": "user", "content": "First"}]
    assert replace is False


def test_flatten_numbers_follow_ups() -> None:
    messages, _ = build_messages(
        "Req",
        [{"role": "assistant", "content": "a"}, {"role": "user", "content": "f1"},
         {"role": "assistant", "content": "b"}],
        "f2", instruction=None,
    )
    text = flatten_messages(messages)
    assert "[follow-up 1]\nf1\n[/follow-up 1]" in text
    assert text.endswith("[follow-up 2]\nf2\n[/follow-up 2]")


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


# -- CLI ------------------------------------------------------------------------


def test_promptmap_chat_cli(tmp_path, monkeypatch) -> None:
    from promptmap.cli import main

    monkeypatch.chdir(tmp_path)
    (tmp_path / "map").mkdir()
    (tmp_path / "client.py").write_text(
        "def llm_call(prompt, messages=None):\n"
        "    return 'reply to: ' + messages[-1]['content']\n"
    )
    (tmp_path / "map" / "poem.prompt").write_text("{{ config(model_type='chat') }}Write a poem.")
    (tmp_path / "map" / "plain.prompt").write_text("Hi")
    import promptmap.db as db

    monkeypatch.setattr(db, "_DEFAULT_BACKEND", SQLiteStorageBackend(tmp_path / ".promptmap" / "p.db"))
    runner = CliRunner()

    first = runner.invoke(main, ["chat", "poem"])
    assert first.exit_code == 0, first.output
    assert "reply to: Write a poem." in first.output

    follow = runner.invoke(main, ["chat", "poem", "Shorter please"])
    assert "reply to: Shorter please" in follow.output

    history = runner.invoke(main, ["chat", "poem", "--history"])
    assert "follow-up 1" in history.output and "Shorter please" in history.output

    fresh = runner.invoke(main, ["chat", "poem", "--new"])
    assert fresh.exit_code == 0
    assert "follow-up" not in runner.invoke(main, ["chat", "poem", "--history"]).output

    wrong = runner.invoke(main, ["chat", "plain"])
    assert wrong.exit_code == 1 and "not a chat model" in wrong.output

    cleared = runner.invoke(main, ["clear-cache", "poem", "--chat"])
    assert "new conversation" in cleared.output
    assert "No conversation yet" in runner.invoke(main, ["chat", "poem", "--history"]).output
