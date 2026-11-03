"""``agent``: the pi agent, text in, text out, with files in and out."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from promptmap.executor import agent as ag
from promptmap.executor import pi
from promptmap.files import File, Output
from tests.test_model_types import run_models

PNG = b"\x89PNG\r\n\x1a\nfake"


def image_llm(prompt, files=None, config=None):
    return File(PNG, name="logo.png") if "draw" in prompt else "ok"


@pytest.fixture
def fake_agent(monkeypatch):
    calls = []

    def exec_agent(task, call, workdir):
        calls.append({"task": task, "inputs": sorted(
            p.relative_to(workdir).as_posix() for p in workdir.rglob("*") if p.is_file()
        )})
        if "write a note" in task:
            (workdir / "outputs" / "note.md").write_text("# note")
        reply = '{"answer": 42}' if "json" in task else "the answer"
        return {"output": reply, "logs": [], "time_run": 0.0}

    monkeypatch.setattr(ag, "_exec_agent", exec_agent)
    return calls


def test_text_in_text_out(fake_agent):
    _, _, results = run_models({
        "a": '{{ config(model_type="agent") }}what is it?',
        "after": '{{ config(model_type="template") }}got: {{ ref("a") }}',
    })
    assert results["a"].value == "the answer"
    assert results["after"].value == "got: the answer"
    assert "outputs/" in fake_agent[0]["task"]


def test_json_output_is_parsed(fake_agent):
    _, _, results = run_models({"a": '{{ config(model_type="agent", output_format="json") }}json please'})
    assert results["a"].value == {"answer": 42}


def test_upstream_files_land_in_inputs(fake_agent):
    _, _, results = run_models(
        {"logo": "draw", "a": '{{ config(model_type="agent", promptfiles="logo") }}describe it'},
        llm_call=image_llm,
    )
    assert results["a"].error == ""
    assert fake_agent[0]["inputs"] == ["inputs/logo/logo.png"]
    assert "inputs/logo/logo.png" in fake_agent[0]["task"]


def test_files_written_to_outputs_become_the_models_files(fake_agent):
    _, _, results = run_models({"a": '{{ config(model_type="agent") }}write a note'})
    value = results["a"].value
    assert isinstance(value, Output)
    assert value.text == "the answer"
    assert value.files["note.md"].read_bytes() == b"# note"


def test_unchanged_task_is_cached(fake_agent):
    from promptmap.storage.memory import MemoryStorageBackend

    storage = MemoryStorageBackend()
    models = {"a": '{{ config(model_type="agent") }}what is it?'}
    run_models(models, storage=storage)
    _, _, results = run_models(models, storage=storage)
    assert results["a"].cached is True
    assert len(fake_agent) == 1


def test_missing_pi_says_how_to_install(monkeypatch):
    monkeypatch.setattr(pi, "_pi_bin", lambda: None)
    _, _, results = run_models({"a": '{{ config(model_type="agent") }}hi'})
    assert "npm i -g @mariozechner/pi-coding-agent" in results["a"].error


def test_pi_events_become_output_and_limits(tmp_path, monkeypatch):
    """run_pi against a fake pi binary that prints JSON events."""
    events = [
        {"type": "session"},
        {"type": "message_update", "message": {}},
        {"type": "message_end", "message": {"role": "assistant", "content": [{"type": "text", "text": "done"}],
                                            "usage": {"cost": {"total": 0.01}}, "stopReason": "stop"}},
        {"type": "turn_end"},
        {"type": "agent_end", "messages": []},
    ]
    script = tmp_path / "pi"
    script.write_text("#!/bin/sh\ncat <<'EOF'\n" + "\n".join(json.dumps(e) for e in events) + "\nEOF\n")
    script.chmod(0o755)
    monkeypatch.setenv("PROMPTMAP_PI_BIN", str(script))
    _, _, results = run_models({"a": '{{ config(model_type="agent") }}hi'})
    assert results["a"].value == "done"
