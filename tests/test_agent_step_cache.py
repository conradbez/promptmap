"""The agent step cache: a run cut off mid-way replays its steps on restart."""

from __future__ import annotations

import json

import pytest

from promptmap import agent_step_cache
from promptmap.executor import mini_swe

from tests.test_each_fanout import run_models

#: The scripted agent: look around, write a file, read it back, submit.
COMMANDS = [
    "pwd",
    "echo hi > note.txt",
    "cat note.txt",
    "printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\nall done\\n'",
]

MODEL = '{{ config(model_type="coding_agent", agent_model="gpt-4o-mini", agent_cost_limit=2) }}write a note'


@pytest.fixture(autouse=True)
def step_cache(tmp_path, monkeypatch):
    path = tmp_path / "agent_steps.db"
    monkeypatch.setenv("PROMPTMAP_AGENT_STEP_CACHE", str(path))
    return path


def _response(step: int):
    import litellm

    return litellm.ModelResponse(
        model="gpt-4o-mini",
        choices=[{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": f"step {step}",
            "tool_calls": [{"id": f"call_{step}", "type": "function", "function": {
                "name": "bash", "arguments": json.dumps({"command": COMMANDS[step]})}}],
        }}],
        usage={"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100},
    )


def scripted_llm(monkeypatch, *, die_at: int | None = None) -> list[list]:
    """A real litellm-backed agent model whose API call is scripted.

    Returns the list of requests that reached the "API" (cache misses).
    ``die_at`` raises on that live call, as if the run was killed.
    """
    from minisweagent.models import get_model

    live: list[list] = []

    def fake_query(messages, **kwargs):
        step = len([m for m in messages if m.get("role") == "assistant"])
        if die_at is not None and len(live) == die_at:
            raise KeyboardInterrupt("cut off")
        live.append(messages)
        return _response(step)

    def model(name, model_cfg):
        m = get_model(name, model_cfg)
        m._query = fake_query
        return m

    monkeypatch.setattr(mini_swe, "_agent_model", model)
    return live


def test_restart_after_cut_off_replays_steps_word_for_word(monkeypatch):
    first = scripted_llm(monkeypatch, die_at=2)
    with pytest.raises(KeyboardInterrupt):
        run_models({"fix": MODEL})
    assert len(first) == 2  # steps 1 and 2 were paid for, then the run died

    # Restart: a fresh run database, so promptmap's own cache has nothing.
    again = scripted_llm(monkeypatch)
    _, _, results = run_models({"fix": MODEL})

    assert results["fix"].value["output"].strip() == "all done"
    assert results["fix"].value["changes"] == {"note.txt": "hi\n"}
    # Steps 1-2 came from the step cache; only steps 3 and 4 hit the API.
    assert len(again) == 2
    assert [m["content"] for m in again[0] if m.get("role") == "assistant"] == ["step 0", "step 1"]


def test_identical_rerun_makes_no_llm_calls(monkeypatch):
    scripted_llm(monkeypatch)
    run_models({"fix": MODEL})
    again = scripted_llm(monkeypatch)
    _, _, results = run_models({"fix": MODEL})
    assert again == []
    assert results["fix"].value["output"].strip() == "all done"


def test_clear_forgets_steps(monkeypatch):
    scripted_llm(monkeypatch)
    run_models({"fix": MODEL})
    assert agent_step_cache.clear(["other"]) == 0
    assert agent_step_cache.clear(["fix"]) == 4
    again = scripted_llm(monkeypatch)
    run_models({"fix": MODEL})
    assert len(again) == 4


def test_off_switch(monkeypatch):
    monkeypatch.setenv("PROMPTMAP_AGENT_STEP_CACHE", "off")
    scripted_llm(monkeypatch)
    run_models({"fix": MODEL})
    again = scripted_llm(monkeypatch)
    run_models({"fix": MODEL})
    assert len(again) == 4


def test_scratch_dir_is_stable_but_never_shared():
    with agent_step_cache.scratch_dir("pm-test-", "k") as a:
        (a / "left-over").write_text("x")
        with agent_step_cache.scratch_dir("pm-test-", "k") as b:
            assert b != a  # same key while a is in use: a random folder
    with agent_step_cache.scratch_dir("pm-test-", "k") as c:
        assert c == a and not (c / "left-over").exists()
