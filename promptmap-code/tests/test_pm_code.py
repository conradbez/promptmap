"""pm-code: tools act through the Workspace; the agent loop drives them from tool calls."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from promptmap.workspace import Workspace
from promptmap_code.agent import PmCodeAgent, PromptmapEnvironment, ToolModel
from promptmap_code.tools import TOOLS, Toolbox


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "map").mkdir()
    return Workspace(tmp_path)


@pytest.fixture
def box(ws):
    return Toolbox(ws)


def test_every_tool_has_a_handler(box):
    assert sorted(t["function"]["name"] for t in TOOLS) == sorted(box.names)


def test_build_and_inspect_a_pipeline(box, ws):
    assert box.call("create_chat_node", {"name": "spec", "request": "Spec a todo CLI.", "folder": "1_plan"}).ok
    assert box.call("create_coding_agent_node", {"name": "build", "task": "Build it.", "folder": "2_build"}).ok
    r = box.call("connect_nodes", {"upstream": "spec", "downstream": "build"})
    assert r.ok and "ref('spec')" in r.output
    listing = box.call("list_nodes", {}).output
    assert "build  [coding_agent]  2_build/build.prompt  <- spec" in listing
    assert box.call("list_nodes", {"prefix": "1_plan/*"}).output.startswith("spec  [chat]")
    assert box.call("rename_node", {"old": "spec", "new": "plan"}).ok
    assert "ref('plan')" in box.call("show_node", {"name": "build"}).output


def test_errors_come_back_as_failed_results(box):
    r = box.call("show_node", {"name": "nope"})
    assert not r.ok and "No model named 'nope'" in r.output
    assert not box.call("no_such_tool", {}).ok
    assert not box.call("show_node", {"wrong": 1}).ok


def test_jinja_tools(box):
    box.call("create_chat_node", {"name": "n", "request": "Hi."})
    assert "previous_of" in box.call("list_jinja_functions", {}).output
    assert box.call("add_jinja", {"name": "n", "jinja": "{{ promptdata('tone') }}"}).ok
    assert not box.call("add_jinja", {"name": "n", "jinja": "{% if %}"}).ok


def test_read_code_dir_is_read_only_and_confined(box, ws):
    (ws.code_dir / "app.py").write_text("print('hi')\n")
    (ws.code_dir / ".git").mkdir()
    (ws.code_dir / ".git" / "HEAD").write_text("x")
    listing = box.call("read_code_dir", {}).output
    assert "app.py" in listing and ".git" not in listing
    assert box.call("read_code_dir", {"path": "app.py"}).output == "print('hi')\n"
    assert "outside" in box.call("read_code_dir", {"path": "../"}).output


def test_run_and_read_output(box, ws, monkeypatch):
    box.call("create_chat_node", {"name": "spec", "request": "Spec."})
    monkeypatch.setattr(ws, "run", lambda select=None, **kw: {"spec": "THE SPEC"})
    assert box.call("run_nodes", {}).output == "spec: ok — THE SPEC"
    assert "no successful output" in box.call("read_node_output", {"name": "spec"}).output


# -- the agent loop, with scripted LLM replies ----------------------------------

def _response(content=None, calls=()):
    tool_calls = [SimpleNamespace(id=f"c{i}", type="function",
                                  function=SimpleNamespace(name=name, arguments=json.dumps(args)))
                  for i, (name, args) in enumerate(calls)]
    message = SimpleNamespace(
        tool_calls=tool_calls or None,
        model_dump=lambda: {"role": "assistant", "content": content, "tool_calls": [
            {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
            for c in tool_calls] or None},
    )
    choice = SimpleNamespace(message=message, finish_reason="tool_calls" if tool_calls else "stop")
    return SimpleNamespace(choices=[choice], model_dump=lambda **_: {})


def test_agent_calls_tools_until_it_answers(ws, monkeypatch):
    script = [
        _response(calls=[("create_chat_node", {"name": "spec", "request": "Spec."}),
                         ("create_coding_agent_node", {"name": "build", "task": "Build."})]),
        _response(calls=[("connect_nodes", {"upstream": "spec", "downstream": "build"})]),
        _response(content="Pipeline ready: spec -> build."),
    ]
    sent = []

    def fake_query(self, messages, **kw):
        sent.append(messages)
        return script.pop(0)

    monkeypatch.setattr(ToolModel, "_query", fake_query)
    seen = []
    agent = PmCodeAgent(ToolModel(model_name="test/model"),
                        PromptmapEnvironment(Toolbox(ws), on_action=lambda a, o: seen.append(a["tool"])))
    assert agent.send("Build a todo app") == "Pipeline ready: spec -> build."
    assert seen == ["create_chat_node", "create_coding_agent_node", "connect_nodes"]
    assert [n["name"] for n in ws.nodes()] == ["build", "spec"]
    assert ws.nodes()[0]["depends_on"] == ["spec"]
    # tool results go back as role=tool messages answering each call
    tool_msgs = [m for m in sent[-1] if m.get("role") == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["c0", "c1", "c0"]
    assert sent[0][0]["role"] == "system" and str(ws.code_dir) in sent[0][0]["content"]


def test_agent_stops_at_max_steps(ws, monkeypatch):
    monkeypatch.setattr(ToolModel, "_query", lambda self, m, **kw: _response(calls=[("list_nodes", {})]))
    agent = PmCodeAgent(ToolModel(model_name="test/model"), PromptmapEnvironment(Toolbox(ws)), max_steps=3)
    assert "Stopped after 3 tool steps" in agent.send("loop")


def test_cli_one_shot(tmp_path, monkeypatch):
    from promptmap_code import cli

    monkeypatch.setattr(ToolModel, "_query", lambda self, m, **kw: _response(content="Hello."))
    result = CliRunner().invoke(cli.main, ["--root", str(tmp_path), "--model", "openai/gpt-test", "hi"])
    assert result.exit_code == 0, result.output
    assert "Hello." in result.output
    assert (tmp_path / "settings.yml").read_text() == "provider: openai\nmodel: gpt-test\n"


def test_cli_needs_a_model(tmp_path, monkeypatch):
    from promptmap_code import cli

    monkeypatch.delenv("PM_CODE_MODEL", raising=False)
    monkeypatch.delenv("MSWEA_MODEL_NAME", raising=False)
    result = CliRunner().invoke(cli.main, ["--root", str(tmp_path), "hi"])
    assert result.exit_code == 2 and "no model" in result.output


# -- system prompt layering ------------------------------------------------------

def test_system_prompt_layers_rules_project_then_skill(ws):
    from promptmap_code.agent import build_system_prompt

    (ws.root / "PM_CODE.md").write_text("Always use Python 3.12.")
    prompt = build_system_prompt(ws.code_dir, ws.root)
    rules = prompt.index("You are pm-code")
    project = prompt.index("Always use Python 3.12.")
    skill = prompt.index("# Reference: the promptmap skill")
    assert rules < project < skill
    assert "{{ config(...) }}" in prompt and "name: promptmap" not in prompt  # formatted; frontmatter gone
    assert "# Reference" not in build_system_prompt(ws.code_dir, ws.root, skill=False)


def test_packaged_skill_matches_repo_copy():
    from pathlib import Path

    import promptmap_code

    repo = Path(__file__).resolve().parents[2] / "skills" / "promptmap" / "SKILL.md"
    packaged = Path(promptmap_code.__file__).parent / "promptmap_skill.md"
    if not repo.is_file():
        pytest.skip("not in the promptmap repo")
    assert packaged.read_text() == repo.read_text(), (
        "Copy skills/promptmap/SKILL.md to promptmap-code/src/promptmap_code/promptmap_skill.md")


# -- progress while nodes run ----------------------------------------------------

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
    box = Toolbox(ws, progress=lambda *e: events.append(e))
    ws.add_node("topic", "cats", {"model_type": "template"})
    ws.add_node("poem", "Poem about:", {"model_type": "stepper_test"})
    ws.connect("topic", "poem")

    original = ws.run
    ws.run = lambda select=None, **kw: original(select, llm_call=lambda p: "unused", **kw)
    out = box.call("run_nodes", {"names": ["poem"]}).output
    assert "poem: ok — POEM" in out
    assert ("start", "poem", "") in events and ("step", "poem", "2/5") in events
    assert ("done", "poem", "success") in events
    assert events.index(("start", "topic", "")) < events.index(("done", "topic", "success"))


def test_cli_progress_printer():
    from promptmap_code.cli import NodeProgress

    p = NodeProgress()
    p("start", "build", "")
    p("step", "build", "3")
    assert p.running == {"build": "step 3"} and p.status is not None
    p("done", "build", "success")
    assert p.running == {} and p.status is None
