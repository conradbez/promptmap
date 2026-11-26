"""``agent``: the mini-swe-agent agent, text in, text out, with files in and out."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from promptmap.executor import agent as ag
from promptmap.executor import mini_swe
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


def test_mini_swe_agent_submission_is_output(monkeypatch):
    """run_mini_swe end to end, with a scripted model instead of an LLM."""
    from minisweagent.models.test_models import DeterministicModel, make_output

    seen = {}

    def fake_model(name, model_cfg):
        seen["name"] = name
        return DeterministicModel(outputs=[
            make_output("write", [{"command": "echo '# hi' > outputs/note.md"}]),
            make_output("done", [{"command": "printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\ndone\\n'"}]),
        ])

    monkeypatch.setattr(mini_swe, "_agent_model", fake_model)
    _, _, results = run_models({"a": '{{ config(model_type="agent", agent_model="m1") }}hi'})
    value = results["a"].value
    assert isinstance(value, Output)
    assert value.text.strip() == "done"
    assert set(value.files) == {"note.md"}
    assert seen["name"] == "m1"


def test_agent_is_told_its_remaining_budget(monkeypatch):
    """Every command result ends with steps/spend left; the last step says submit now."""
    from minisweagent.models.test_models import DeterministicModel, make_output

    agents = []

    def fake_model(name, model_cfg):
        model = DeterministicModel(observation_template=model_cfg["observation_template"], outputs=[
            make_output("look", [{"command": "true"}], cost=0.25),
            make_output("look", [{"command": "true"}], cost=0.25),
            make_output("done", [{"command": "printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\nok\\n'"}]),
        ])
        agents.append(model)
        return model

    monkeypatch.setattr(mini_swe, "_agent_model", fake_model)
    seen = []
    real = mini_swe.run_mini_swe

    def spy(task, call, workdir):
        result = real(task, call, workdir)
        seen.extend(m.get("content") for m in result["logs"] if m.get("role") in ("user", "tool"))
        return result

    monkeypatch.setattr(ag, "_exec_agent", spy)
    _, _, results = run_models({
        "a": '{{ config(model_type="agent", agent_step_limit=3, agent_cost_limit=2) }}hi',
    })
    assert results["a"].value.strip() == "ok"
    budgets = [str(c) for c in seen if "<budget>" in str(c)]
    assert "Step 1 of 3 (2 left). $0.25 of $2.00 spent. Submit your final output before" in budgets[0]
    assert "Step 2 of 3 (1 left). $0.50 of $2.00 spent. Your next response is your last" in budgets[1]


def test_run_level_promptfiles_land_in_inputs(fake_agent, tmp_path):
    import io

    spec = tmp_path / "spec.md"
    spec.write_text("# spec v1")
    handle = io.BytesIO(b"raw")
    handle.name = "data.bin"
    _, _, results = run_models(
        {"a": '{{ config(model_type="agent", promptfiles=["spec", "data"]) }}read it'},
        promptfiles={"spec": str(spec), "data": handle},
    )
    assert results["a"].error == ""
    assert fake_agent[0]["inputs"] == ["inputs/data/data.bin", "inputs/spec/spec.md"]
    assert "inputs/spec/spec.md" in fake_agent[0]["task"]
    assert handle.tell() == 0  # put back for anyone else reading it


def test_run_level_promptfile_change_reruns(fake_agent, tmp_path):
    from promptmap.storage.memory import MemoryStorageBackend

    storage = MemoryStorageBackend()
    spec = tmp_path / "spec.md"
    model = {"a": '{{ config(model_type="agent", promptfiles="spec") }}read it'}
    for text in ("v1", "v1", "v2"):
        spec.write_text(text)
        run_models(model, storage=storage, promptfiles={"spec": str(spec)})
    assert len(fake_agent) == 2  # v1 once (then cached), v2 again


def test_missing_run_level_promptfile_is_an_error(fake_agent, tmp_path):
    other = tmp_path / "x.md"
    other.write_text("x")
    _, _, results = run_models(
        {"a": '{{ config(model_type="agent", promptfiles="spec") }}read it'},
        promptfiles={"other": str(other)},
    )
    assert "--promptfile spec=path" in results["a"].error


def test_mini_swe_templates_name_one_submit_command():
    """mini-swe's bare-echo submit is replaced, so the agent is not told two ways to finish."""
    import yaml
    from minisweagent import package_dir

    from promptmap.executor import mini_swe

    raw = yaml.safe_load((package_dir / "config" / "default.yaml").read_text())
    assert mini_swe._BARE_SUBMIT in raw["agent"]["instance_template"]  # mini-swe still says it
    fixed = mini_swe.submit_with_output(raw)
    for section, key in (("agent", "instance_template"), ("model", "format_error_template")):
        assert mini_swe._BARE_SUBMIT not in fixed[section][key]
        assert mini_swe.SUBMIT_COMMAND in fixed[section][key]
    assert mini_swe._BARE_SUBMIT in raw["agent"]["instance_template"]  # input left alone
    assert mini_swe.SUBMIT_COMMAND in mini_swe._SUBMIT_INSTRUCTION
