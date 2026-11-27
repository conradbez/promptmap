"""pm-code: pick a chat, refine its intent, use /tools, and its last reply calls pm tools."""

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
    assert [(c.tool, c.name) for c in plan.calls] == [
        ("coding_agent", "build_cli"), ("coding_agent", "write_tests"), ("chat", "storage")]
    assert plan.calls[0].text == "Create todo.py with add, list and done commands."
    many = "\n".join(f"## coding_agent: a{i}\ntask {i}" for i in range(MAX_CODING_AGENTS + 1))
    plan = parse_plan(many + "\n## Coding_Agent: a0\nagain")
    assert len(plan.calls) == MAX_CODING_AGENTS
    assert any("more than" in d for d in plan.dropped) and any("twice" in d for d in plan.dropped)
    assert parse_plan("no headings here").calls == []


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

    done, plan = s.apply("app")
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
    assert {st for _, st in s.apply("app")[0]} == {"unchanged"}


def test_children_do_not_change_the_chats_prompt(ws):
    """org_outline(up=0, used_by=False) leaves consumers out: the chat's reply is still a cache hit after spawning."""
    llm = FakeLLM(FINAL, "SHOULD NOT BE ASKED")
    s = Session(ws, llm_call=llm)
    s.create_chat("app", "Build a todo app.")
    s.reply("app")
    s.apply("app")
    assert s.reply("app") == FINAL and len(llm.calls) == 1


def test_spawn_skips_names_taken_elsewhere(ws):
    s = Session(ws, llm_call=FakeLLM(FINAL))
    ws.add_chat_node("storage", "elsewhere")
    s.create_chat("app", "Build a todo app.")
    s.reply("app")
    status = dict((c.name, st) for c, st in s.apply("app")[0])
    assert status["storage"].startswith("skipped") and status["build_cli"] == "created"


def test_outline_marks_where_you_are(ws):
    s = Session(ws, llm_call=FakeLLM(FINAL))
    s.create_chat("app", "Build a todo app.")
    s.reply("app")
    s.apply("app")
    text = s.outline("app")
    assert "app [you are here]" in text and "build_cli" in text and "- " in text


def test_cli_new_chat_done_and_quit(tmp_path, monkeypatch):
    from promptmap_code import cli

    llm = FakeLLM(FINAL)
    real_run = Workspace.run
    monkeypatch.setattr(Workspace, "run", lambda self, select=None, **kw: real_run(self, select, llm_call=llm, **kw))
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    keys = "app\nBuild a todo app.\n/do\nn\n\n\n"
    result = CliRunner().invoke(cli.main, ["--root", str(tmp_path), "-m", "openai/gpt-test"], input=keys)
    assert result.exit_code == 0, result.output
    assert "created chat 'app'" in result.output and "Here is the plan" in result.output
    assert "created coding_agent build_cli" in result.output and "/done applies them" in result.output
    assert (tmp_path / "map" / "app" / "build_cli.prompt").is_file()
    assert (tmp_path / "settings.yml").read_text() == "provider: openai\nmodel: gpt-test\n"


def test_parse_connect_and_model_calls():
    plan = parse_plan("### connect: storage -> build_cli\n### model: build_cli -> openai/gpt-5\n### connect: oops\n")
    assert [(c.tool, c.name, c.target) for c in plan.calls] == [
        ("connect", "storage", "build_cli"), ("model", "build_cli", "openai/gpt-5")]
    assert plan.dropped == ["connect oops: expected '<node> -> <node>'"]


def test_last_reply_connects_and_sets_models_after_creating(ws):
    reply = FINAL + "\n### connect: storage -> build_cli\n### model: build_cli -> openai/gpt-5\n"
    s = Session(ws, llm_call=FakeLLM("draft ### coding_agent: early\nnot this", reply))
    s.create_chat("app", "Build a todo app.")
    s.reply("app")
    s.reply("app", "go")
    done = dict((c.label, st) for c, st in s.apply("app")[0])
    assert done["connect storage -> build_cli"] == "connected" and done["model build_cli -> openai/gpt-5"] == "set"
    assert "early" not in {n["name"] for n in ws.nodes()}  # only the last reply is evaluated
    assert sorted(ws.nodes("app/build_cli")[0]["depends_on"]) == ["app", "storage"]
    assert s.model_of("build_cli") == "openai/gpt-5"
    assert 'agent_model="openai/gpt-5"' in ws.source("build_cli")  # coding agents' own key
    assert {st for _, st in s.apply("app")[0]} == {"unchanged"}


def test_a_reply_cannot_change_its_own_chat(ws):
    s = Session(ws, llm_call=FakeLLM("### model: app -> openai/x\n### connect: other -> app"))
    ws.add_node("other", "x", {"model_type": "template"})
    s.create_chat("app", "Build.")
    s.reply("app")
    assert all(st.startswith("skipped") for _, st in s.apply("app")[0])


def test_intent_is_the_prompt_files_first_message(ws):
    s = Session(ws)
    s.create_chat("app", "Build a todo app.\nIn python.")
    assert s.intent("app") == "Build a todo app.\nIn python."
    llm = FakeLLM("A", "B")
    s.run_kwargs = {"llm_call": llm}
    s.reply("app")
    s.reply("app", "smaller")
    first, *rest = [m for m in llm.calls[-1] if m["role"] != "system"]
    assert first["role"] == "user" and first["content"].strip().startswith("Build a todo app.")
    assert rest == [{"role": "assistant", "content": "A"}, {"role": "user", "content": "smaller"}]


def test_nearby_orders_by_map_distance_and_skips_cycles(ws):
    s = Session(ws)
    for name, body in {"far": "x", "near": "{{ ref('far') }}", "lone": "y"}.items():
        ws.add_node(name, body, {"model_type": "template"})
    s.create_chat("app", "Build.")
    ws.connect("near", "app")
    ws.add_node("child", "{{ ref('app') }}", {"model_type": "template"})
    assert s.nearby("app") == ["near", "far", "lone"]


def test_slash_connect_and_model(ws, monkeypatch):
    from promptmap_code import cli

    s = Session(ws)
    ws.add_node("notes", "x", {"model_type": "template"})
    s.create_chat("app", "Build.")
    monkeypatch.setattr(cli, "_choose", lambda label, options, allow_custom=False: options[0])
    cli.tool_connect(s, "app", "")
    src = ws.source("app")
    assert src.index("ref('notes')") < src.index("## How to finish")  # inputs sit above the guide
    assert s.intent("app") == "Build."
    cli.tool_model(s, "app", "openai/gpt-5")
    assert s.model_of("app") == "openai/gpt-5"
    cli.tool_model(s, "app", "default")
    assert s.model_of("app") == ""


def test_resolve_tool_takes_unique_prefixes(monkeypatch):
    from promptmap_code import cli

    assert cli.resolve_tool("/con notes") == ("connect", "notes")
    assert cli.resolve_tool("/model openai/x") == ("model", "openai/x")
    monkeypatch.setattr(cli, "_choose", lambda label, options, allow_custom=False: options[-1])
    assert cli.resolve_tool("/")[0] == "help"  # bare '/' opens the list


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
