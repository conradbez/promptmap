"""promptmap.workspace: node edits go through the construct, runs read back."""

from __future__ import annotations

import json

import pytest

from promptmap.workspace import Workspace, WorkspaceError, split_config


@pytest.fixture(autouse=True)
def chat_kind():
    """The real chat kind ships in promptmap[extras]; core tests use a stand-in."""
    from promptmap.model_types import ModelKind, get_model_kind, register_model_kind

    if get_model_kind("chat") is None:
        async def reply(rendered, call):
            return "REPLY"

        register_model_kind(ModelKind("chat", reply, chat=True))


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "map").mkdir()
    return Workspace(tmp_path)


def test_add_coding_agent_node_writes_to_code_dir(tmp_path):
    (tmp_path / "map").mkdir()
    ws = Workspace(tmp_path, code_dir="app")
    path = ws.add_coding_agent_node("build", "Write hello.py", folder="2_code")
    assert path == tmp_path / "map" / "2_code" / "build.prompt"
    config = ws.construct().get_config("build")
    assert config == {"model_type": "coding_agent", "project": "app"}
    assert Workspace(tmp_path).construct().get_config("build")["project"] == "app"


def test_code_dir_defaults_to_root(ws):
    ws.add_coding_agent_node("build", "x")
    assert ws.construct().get_config("build")["project"] == "."


def test_add_node_rejects_duplicates_and_bad_names(ws):
    ws.add_chat_node("spec", "Write a spec.")
    with pytest.raises(WorkspaceError, match="already exists"):
        ws.add_chat_node("spec", "again")
    with pytest.raises(WorkspaceError, match="not a valid node name"):
        ws.add_chat_node("bad name", "x")
    with pytest.raises(WorkspaceError, match="inside the map"):
        ws.add_chat_node("ok", "x", folder="../outside")


def test_nodes_lists_types_deps_and_filters_by_path(ws):
    ws.add_chat_node("spec", "Spec.", folder="1_overall")
    ws.add_coding_agent_node("build", "Build.", folder="1_overall/3_inside")
    ws.connect("spec", "build")
    nodes = {n["name"]: n for n in ws.nodes()}
    assert nodes["spec"]["model_type"] == "chat"
    assert nodes["build"]["depends_on"] == ["spec"]
    assert [n["name"] for n in ws.nodes("1_overall/3_inside/*")] == ["build"]
    assert {n["name"] for n in ws.nodes("1_overall/")} == {"spec", "build"}


def test_connect_text_and_folder(ws):
    ws.add_coding_agent_node("api", "Build an API.")
    ws.add_coding_agent_node("tests", "Write tests.")
    ws.add_chat_node("review", "Review the work.")
    added = ws.connect("api", "review")
    assert "ref('api.output')" in added and added in ws.source("review")
    ws.connect("api", "tests", as_folder=True)
    assert ws.construct().get_config("tests")["inherit_coding_folder"] == "api"
    with pytest.raises(WorkspaceError, match="coding_agent downstream"):
        ws.connect("api", "review", as_folder=True)


def test_connect_refuses_cycles_and_leaves_files_alone(ws):
    ws.add_chat_node("a", "A.")
    ws.add_chat_node("b", "B.")
    ws.connect("a", "b")
    before = ws.source("a")
    with pytest.raises(WorkspaceError, match="cycle"):
        ws.connect("b", "a")
    assert ws.source("a") == before


def test_set_body_keeps_config_call_verbatim(ws):
    ws.add_node("n", "old", {"model_type": "chat"})
    ws.set_config("n", model="low")
    call, _ = split_config(ws.source("n"))
    ws.set_body("n", "new text")
    assert ws.source("n") == call + "new text\n"


def test_insert_jinja_checks_syntax(ws):
    ws.add_node("n", "Body.")
    ws.insert_jinja("n", "{% if previous %}Last time: {{ previous }}{% endif %}")
    assert ws.source("n").endswith("{% if previous %}Last time: {{ previous }}{% endif %}\n")
    with pytest.raises(WorkspaceError, match="Jinja"):
        ws.insert_jinja("n", "{% if x %}unclosed")


def test_jinja_functions_lists_builtins_and_plugins(ws):
    funcs = ws.jinja_functions()
    for name in ("ref", "promptdata", "previous_of", "skip_and_set_to_value", "org_outline"):
        assert name in funcs and funcs[name]


def test_run_output_and_rename_keeps_history(ws):
    ws.add_node("topic", "cats", {"model_type": "template"})
    ws.add_node("poem", "Poem about:")
    ws.connect("topic", "poem")
    calls = []

    def llm(prompt: str) -> str:
        calls.append(prompt)
        return "POEM"

    results = ws.run(["poem"], llm_call=llm)
    assert results["poem"] == "POEM" and results["topic"].strip() == "cats"
    assert ws.output("poem") == "POEM"
    assert "cats" in calls[0]

    ws.rename("poem", "verse")
    assert ws.output("verse") == "POEM"
    assert ws.output("poem") is None
    with pytest.raises(WorkspaceError, match="No model named"):
        ws.rename("poem", "x")


def test_output_summarises_coding_agent(ws, monkeypatch):
    stored = {"output": "done", "changes": {"a.py": "x"}, "deleted": [], "result": {"dir": "..."},
              "cache": "miss", "project": {"written": ["a.py"]}}

    class Row(dict):
        pass

    class Storage:
        def get_latest_output(self, name, run_id):
            return Row(llm_output="$promptmap:v1\n" + json.dumps(stored), llm_output_validated=None)

    monkeypatch.setattr(ws, "_storage", lambda: Storage())
    summary = json.loads(ws.output("build"))
    assert summary["changed_files"] == ["a.py"] and summary["output"] == "done"
    assert "result" not in summary
