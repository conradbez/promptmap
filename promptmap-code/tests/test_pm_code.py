"""pm-code: pick a chat, talk to it, and its final reply creates the next nodes."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from promptmap.workspace import Workspace
from promptmap_code.session import MAX_CODING_AGENTS, Session, parse_plan

FINAL = """Here is the plan for the todo app.

### coding_agent: build_cli
Create todo.py with add, list and done commands.

### coding_agent: `write_tests`
Add tests/test_todo.py.

### chat: storage
Decide how tasks are stored.
"""


class FakeLLM:
    """A scripted chat model: replies in order, recording the messages it was sent."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, prompt, messages=None):
        self.calls.append(messages)
        return self.replies.pop(0) if self.replies else "ok"


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "map").mkdir()
    return Workspace(tmp_path)


def test_parse_plan_reads_headings_and_limits():
    plan = parse_plan(FINAL)
    assert [(c.kind, c.name) for c in plan.children] == [
        ("coding_agent", "build_cli"), ("coding_agent", "write_tests"), ("chat", "storage")]
    assert plan.children[0].text == "Create todo.py with add, list and done commands."
    many = "\n".join(f"## coding_agent: a{i}\ntask {i}" for i in range(MAX_CODING_AGENTS + 1))
    plan = parse_plan(many + "\n## Coding_Agent: a0\nagain")
    assert len(plan.children) == MAX_CODING_AGENTS
    assert any("more than" in d for d in plan.dropped) and any("twice" in d for d in plan.dropped)
    assert parse_plan("no headings here").children == []


def test_a_chat_carries_the_map_and_the_finish_guide(ws):
    s = Session(ws)
    s.create_chat("app", "Build a todo app.")
    src = ws.source("app")
    assert 'model_type="chat"' in src or "model_type='chat'" in src
    assert "{{ org_outline(up=0, used_by=False) }}" in src and "### coding_agent: <name>" in src


def test_final_reply_creates_children_that_ref_the_chat(ws):
    llm = FakeLLM("What storage?", FINAL)
    s = Session(ws, llm_call=llm)
    s.create_chat("app", "Build a todo app.")
    assert s.reply("app") == "What storage?"
    assert s.reply("app", "JSON file") == FINAL
    assert llm.calls[-1][-1] == {"role": "user", "content": "JSON file"}

    done, plan = s.spawn("app")
    assert [(c.name, st) for c, st in done] == [
        ("build_cli", "created"), ("write_tests", "created"), ("storage", "created")]
    nodes = {n["name"]: n for n in ws.nodes()}
    assert nodes["build_cli"]["path"] == "app/build_cli.prompt"
    assert nodes["build_cli"]["model_type"] == "coding_agent"
    assert nodes["storage"]["model_type"] == "chat"
    assert all(nodes[n]["depends_on"] == ["app"] for n in ("build_cli", "write_tests", "storage"))
    assert "Create todo.py" in ws.source("build_cli") and "ref('app')" in ws.source("build_cli")
    assert {n["name"] for n in s.children("app")} == {"build_cli", "write_tests", "storage"}

    # A child chat makes its own children one folder deeper.
    assert s.child_folder("storage") == "app/storage"
    # Spawning again from the same reply changes nothing.
    assert {st for _, st in s.spawn("app")[0]} == {"unchanged"}


def test_children_do_not_change_the_chats_prompt(ws):
    """org_outline(up=0, used_by=False) leaves consumers out: the chat's reply is still a cache hit after spawning."""
    llm = FakeLLM(FINAL, "SHOULD NOT BE ASKED")
    s = Session(ws, llm_call=llm)
    s.create_chat("app", "Build a todo app.")
    s.reply("app")
    s.spawn("app")
    assert s.reply("app") == FINAL and len(llm.calls) == 1


def test_spawn_skips_names_taken_elsewhere(ws):
    s = Session(ws, llm_call=FakeLLM(FINAL))
    ws.add_chat_node("storage", "elsewhere")
    s.create_chat("app", "Build a todo app.")
    s.reply("app")
    status = dict((c.name, st) for c, st in s.spawn("app")[0])
    assert status["storage"].startswith("skipped") and status["build_cli"] == "created"


def test_outline_marks_where_you_are(ws):
    s = Session(ws, llm_call=FakeLLM(FINAL))
    s.create_chat("app", "Build a todo app.")
    s.reply("app")
    s.spawn("app")
    text = s.outline("app")
    assert "app [you are here]" in text and "build_cli" in text and "- " in text


def test_cli_new_chat_done_and_quit(tmp_path, monkeypatch):
    from promptmap_code import cli

    llm = FakeLLM(FINAL)
    real_run = Workspace.run
    monkeypatch.setattr(Workspace, "run", lambda self, select=None, **kw: real_run(self, select, llm_call=llm, **kw))
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    keys = "app\nBuild a todo app.\n/done\nn\n\n\n"
    result = CliRunner().invoke(cli.main, ["--root", str(tmp_path), "-m", "openai/gpt-test"], input=keys)
    assert result.exit_code == 0, result.output
    assert "created chat 'app'" in result.output and "Here is the plan" in result.output
    assert "created coding_agent build_cli" in result.output
    assert (tmp_path / "map" / "app" / "build_cli.prompt").is_file()
    assert (tmp_path / "settings.yml").read_text() == "provider: openai\nmodel: gpt-test\n"


def test_cli_needs_a_model(tmp_path, monkeypatch):
    from promptmap_code import cli

    monkeypatch.delenv("PM_CODE_MODEL", raising=False)
    monkeypatch.delenv("MSWEA_MODEL_NAME", raising=False)
    result = CliRunner().invoke(cli.main, ["--root", str(tmp_path)])
    assert result.exit_code == 2 and "no model" in result.output


def test_run_reports_progress(ws):
    import asyncio

    from promptmap.model_types import ModelKind, get_model_kind, register_model_kind, report_progress

    if get_model_kind("stepper_test") is None:
        async def stepper(rendered, call):  # reports steps from a worker thread, like the agent kinds
            def work():
                report_progress(call.spec.name, 2, 5)
                return "POEM"
            return await asyncio.to_thread(work)

        register_model_kind(ModelKind("stepper_test", stepper))
    events = []
    s = Session(ws, progress=lambda *e: events.append(e), llm_call=lambda p: "unused")
    ws.add_node("topic", "cats", {"model_type": "template"})
    ws.add_node("poem", "Poem about:", {"model_type": "stepper_test"})
    ws.connect("topic", "poem")
    assert s.run(["poem"])["poem"] == "POEM"
    assert ("start", "poem", "") in events and ("step", "poem", "2/5") in events
    assert ("done", "poem", "success") in events


def test_cli_progress_printer():
    from promptmap_code.cli import NodeProgress

    p = NodeProgress()
    p("start", "build", "")
    p("step", "build", "3")
    assert p.running == {"build": "step 3"} and p.status is not None
    p("done", "build", "success")
    assert p.running == {} and p.status is None


# -- API keys ----------------------------------------------------------------------

def test_missing_api_keys_knows_openai_like_providers(monkeypatch):
    from promptmap_code.cli import missing_api_keys

    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    assert "SARVAM_API_KEY" in missing_api_keys("sarvam/sarvam-m")
    monkeypatch.setenv("SARVAM_API_KEY", "x")
    assert "SARVAM_API_KEY" not in missing_api_keys("sarvam/sarvam-m")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert missing_api_keys("anthropic/claude-sonnet-5-5") == ["ANTHROPIC_API_KEY"]


def test_cli_fails_fast_on_a_missing_key(tmp_path, monkeypatch):
    from promptmap_code import cli

    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    result = CliRunner().invoke(cli.main, ["--root", str(tmp_path), "-m", "sarvam/sarvam-m"])
    assert result.exit_code == 2 and "SARVAM_API_KEY" in result.output


def test_cli_loads_the_project_env(tmp_path, monkeypatch):
    import os

    from promptmap_code import cli

    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    (tmp_path / ".env").write_text("SARVAM_API_KEY=from-dotenv\n")
    result = CliRunner().invoke(cli.main, ["--root", str(tmp_path), "-m", "sarvam/sarvam-m"], input="\n")
    assert result.exit_code == 0, result.output
    assert os.environ.pop("SARVAM_API_KEY") == "from-dotenv"
