"""``coding_agent``: Dir in, patched Dir out, with diff/verify/merge reuse."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from promptmap.executor import builtin_kinds
from promptmap.files import Dir
from promptmap.model_spec import ModelSpec
from promptmap.model_types import ModelCall, get_model_kind

from promptmap_extra_model_kinds import coding_agent_exec as fa

BASE = {
    "shared.py": b"x = 1\n",
    "util.py": b"def u():\n    return 0\n",
    "old.txt": b"remove me\n",
}


def tree(**overrides) -> Dir:
    files = dict(BASE)
    for name, data in overrides.items():
        path = name.replace("__", ".")
        if data is None:
            files.pop(path, None)
        else:
            files[path] = data
    return Dir(files, name="app")


def feature_edit(root: Path) -> None:
    (root / "feature.py").write_text("FEATURE = True\n")
    (root / "util.py").write_text("def u():\n    return 1\n")
    (root / "old.txt").unlink()


@pytest.fixture
def agent(monkeypatch):
    fake = {"tasks": [], "edit": feature_edit}

    def exec_agent(task, call):
        fake["tasks"].append(task)
        fake["edit"](Path(call.spec.config["agent_dir"]))
        return json.dumps({"output": f"done {len(fake['tasks'])}", "logs": [], "time_run": 0.1})

    monkeypatch.setattr(builtin_kinds, "_exec_agent", exec_agent)
    return fake


@pytest.fixture
def jev(monkeypatch):
    fake = {"score": 1.0, "asked": []}

    def classify(state, question):
        fake["asked"].append(state)
        return fake["score"]

    monkeypatch.setattr(fa, "resolve_classifier", lambda call: classify)
    return fake


@pytest.fixture
def llm():
    return {"replies": [], "prompts": []}


def run(base: Dir, tmp_path, llm, prompt="add the feature", **config) -> dict:
    spec = ModelSpec(
        name="node", source="", model_type=fa.NAME,
        config={"state_dir": str(tmp_path / "state"), **config},
        promptfiles_used=["up"],
    )

    async def compute(key, compute):
        value = compute()
        return await value if hasattr(value, "__await__") else value

    async def call_llm(prompt):
        llm["prompts"].append(prompt)
        return llm["replies"].pop(0)

    call = ModelCall(spec=spec, outputs={"up": base}, llm=call_llm, compute=compute)
    return asyncio.run(fa.run_coding_agent(prompt, call))


def files(d: Dir) -> dict[str, bytes]:
    return {p: f.read_bytes() for p, f in d.entries.items()}


EXPECTED = {
    "shared.py": b"x = 1\n",
    "util.py": b"def u():\n    return 1\n",
    "feature.py": b"FEATURE = True\n",
}


# ---------------------------------------------------------------------------


def test_registered_and_declared():
    from importlib.metadata import entry_points

    kind = get_model_kind(fa.NAME)
    assert kind is fa.KIND
    assert "agent_dir" not in kind.config_keys
    assert {"jev_cache_threshold", "merge_max_files"} <= kind.config_keys
    assert "base_dir" not in kind.config_keys
    eps = {ep.name: ep.value for ep in entry_points(group="promptmap.model_kinds")}
    assert eps[fa.NAME] == "promptmap_extra_model_kinds.coding_agent_exec"


def test_first_run_returns_patched_tree_and_sparse_changes(tmp_path, agent, jev, llm):
    base = tree()
    out = run(base, tmp_path, llm)

    assert out["cache"] == "miss"
    assert out["output"] == "done 1"
    assert files(out["result"]) == EXPECTED
    assert files(out["changes"]) == {k: EXPECTED[k] for k in ("feature.py", "util.py")}
    assert out["deleted"] == ["old.txt"]
    assert files(base) == BASE  # the input is never modified
    assert fa.OUTCOME_CONTRACT.strip() in agent["tasks"][0]
    assert "Previous attempt" not in agent["tasks"][0]


def test_same_input_is_a_hit_without_any_call(tmp_path, agent, jev, llm):
    run(tree(), tmp_path, llm)
    out = run(tree(), tmp_path, llm)

    assert out["cache"] == "hit"
    assert out["output"] == "done 1"
    assert files(out["result"]) == EXPECTED
    assert len(agent["tasks"]) == 1 and jev["asked"] == [] and llm["prompts"] == []


def test_unrelated_upstream_change_is_replayed_after_verify(tmp_path, agent, jev, llm):
    run(tree(), tmp_path, llm)
    out = run(tree(shared__py=b"x = 2\n"), tmp_path, llm)

    assert out["cache"] == "replay"
    assert len(agent["tasks"]) == 1
    assert files(out["result"]) == {**EXPECTED, "shared.py": b"x = 2\n"}
    (asked,) = jev["asked"]
    assert "-x = 1" in asked and "+x = 2" in asked        # upstream_diff
    assert "+FEATURE = True" in asked                      # node_change


def test_failed_verify_reruns_with_previous_attempt(tmp_path, agent, jev, llm):
    run(tree(), tmp_path, llm)
    jev["score"] = 0.3
    out = run(tree(shared__py=b"raise SystemExit\n"), tmp_path, llm)

    assert out["cache"] == "miss"
    task = agent["tasks"][1]
    assert "## Previous attempt" in task
    assert "score 0.30" in task
    assert "done 1" in task                                # previous answer
    assert "+FEATURE = True" in task and "-remove me" in task  # previous change
    assert "+raise SystemExit" in task                     # upstream since then


def test_collision_is_merged_with_one_llm_call(tmp_path, agent, jev, llm):
    run(tree(), tmp_path, llm)
    llm["replies"] = ["<<<FILE util.py>>>\ndef u():\n    log()\n    return 1\n<<<END>>>\n"]
    out = run(tree(util__py=b"def u():\n    log()\n    return 0\n"), tmp_path, llm)

    assert out["cache"] == "merged"
    assert len(agent["tasks"]) == 1
    (merge_prompt,) = llm["prompts"]
    assert "--- ORIGINAL ---\ndef u():\n    return 0" in merge_prompt
    assert "--- YOURS ---\ndef u():\n    return 1" in merge_prompt
    assert "--- UPSTREAM ---\ndef u():\n    log()\n    return 0" in merge_prompt
    assert files(out["result"])["util.py"] == b"def u():\n    log()\n    return 1\n"
    assert files(out["result"])["feature.py"] == b"FEATURE = True\n"
    assert len(jev["asked"]) == 1

    # The merged result is the new baseline: same input again is a plain hit.
    again = run(tree(util__py=b"def u():\n    log()\n    return 0\n"), tmp_path, llm)
    assert again["cache"] == "hit"
    assert files(again["result"]) == files(out["result"])


def test_upstream_recreating_a_deleted_file_can_merge_to_delete(tmp_path, agent, jev, llm):
    run(tree(), tmp_path, llm)
    llm["replies"] = ["<<<DELETE old.txt>>>\n"]
    out = run(tree(old__txt=b"updated upstream\n"), tmp_path, llm)

    assert out["cache"] == "merged"
    assert "old.txt" not in out["result"].entries
    assert out["deleted"] == ["old.txt"]


def test_bad_merge_reply_falls_back_to_agent(tmp_path, agent, jev, llm):
    run(tree(), tmp_path, llm)
    llm["replies"] = ["sorry, I cannot"]
    out = run(tree(util__py=b"def u():\n    return 9\n"), tmp_path, llm)

    assert out["cache"] == "miss"
    assert "could not be merged" in agent["tasks"][1]


def test_too_many_collisions_skip_the_merge(tmp_path, agent, jev, llm):
    run(tree(), tmp_path, llm)
    out = run(tree(util__py=b"def u():\n    return 9\n"), tmp_path, llm, merge_max_files="0")

    assert out["cache"] == "miss"
    assert llm["prompts"] == []
    assert "merge limit 0" in agent["tasks"][1]


@pytest.mark.parametrize("reply, verdict", [("YES", "replay"), ("NO, it breaks", "miss")])
def test_llm_verifies_when_no_classifier(tmp_path, agent, llm, monkeypatch, reply, verdict):
    monkeypatch.setattr(fa, "resolve_classifier", lambda call: None)
    run(tree(), tmp_path, llm)
    llm["replies"] = [reply]
    out = run(tree(shared__py=b"x = 2\n"), tmp_path, llm)

    assert out["cache"] == verdict
    assert llm["prompts"][0].endswith("Answer with exactly YES or NO.")


def test_changed_prompt_reruns_and_says_why(tmp_path, agent, jev, llm):
    run(tree(), tmp_path, llm)
    out = run(tree(), tmp_path, llm, prompt="add the feature, but in feature2.py")

    assert out["cache"] == "miss"
    assert "the task itself changed" in agent["tasks"][1]


def test_force_refresh(tmp_path, agent, jev, llm):
    run(tree(), tmp_path, llm)
    assert run(tree(), tmp_path, llm, force_refresh="true")["cache"] == "miss"


def test_chained_dir_input_from_another_coding_agent(tmp_path, agent, jev, llm):
    first = run(tree(), tmp_path, llm)
    spec = ModelSpec(
        name="second", source="", model_type=fa.NAME,
        config={"state_dir": str(tmp_path / "state")}, promptfiles_used=["first"],
    )
    call = ModelCall(spec=spec, outputs={"first": first}, llm=None, compute=None)
    assert fa.resolve_base(call) is first["result"]


def test_scratch_snapshot_drops_agent_made_junk_but_keeps_input_files(tmp_path):
    root = tmp_path / "scratch"
    for rel in ("node_modules/x/i.js", "vendor/node_modules/y.js", ".git/HEAD", "main.py", ".venv/bin/py"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("")

    snap = fa.snapshot_dir(root, keep={"vendor/node_modules/y.js"})

    assert set(snap.entries) == {"main.py", "vendor/node_modules/y.js"}


def test_agent_test_artifacts_do_not_become_changes(tmp_path, agent, jev, llm):
    def edit(root):
        feature_edit(root)
        (root / "__pycache__").mkdir()
        (root / "__pycache__" / "util.cpython-313.pyc").write_bytes(b"\0")

    agent["edit"] = edit
    out = run(tree(), tmp_path, llm)

    assert files(out["result"]) == EXPECTED


def test_needs_an_input_tree(tmp_path):
    spec = ModelSpec(name="n", source="", model_type=fa.NAME, config={})
    call = ModelCall(spec=spec, outputs={}, llm=None, compute=None)
    with pytest.raises(ValueError, match="needs an input tree"):
        fa.resolve_base(call)


# ---------------------------------------------------------------------------
# Through promptmap's executor
# ---------------------------------------------------------------------------

def test_chain_through_the_executor(tmp_path, monkeypatch, jev):
    from promptmap.executor import build_models_from_dict, execute_run
    from promptmap.storage.memory import MemoryStorageBackend

    app = tmp_path / "app"
    app.mkdir()
    (app / "api.py").write_text("ROUTES = []\n")
    (app / "notes.md").write_text("v1\n")
    (app / "build.log").write_text("ignored by the snapshot node\n")

    def exec_agent(task, call):
        ws = Path(call.spec.config["agent_dir"])
        if call.spec.name == "api":
            (ws / "api.py").write_text("ROUTES = ['/health']\n")
        else:
            (ws / "test_api.py").write_text("from api import ROUTES\nassert ROUTES\n")
        return json.dumps({"output": call.spec.name, "logs": [], "time_run": 0.0})

    monkeypatch.setattr(builtin_kinds, "_exec_agent", exec_agent)
    llm_seen = []

    def llm_call(prompt, files=None):
        llm_seen.append(files)
        return "YES"

    state = tmp_path / "state"
    models = {
        # A static node decides what the input tree is; the agents only see Dirs.
        "repo": '{{ config(model_type="execute_python", cache=False) }}'
                'from pathlib import Path\n'
                f'output = Dir({{p.name: p.read_bytes() for p in Path("{app}").iterdir() if p.suffix != ".log"}}, name="app")',
        "api": f'{{{{ config(model_type="{fa.NAME}", promptfiles="repo", state_dir="{state}") }}}}add /health',
        "tests": f'{{{{ config(model_type="{fa.NAME}", promptfiles="api", state_dir="{state}") }}}}test it',
        "apply": '{{ config(model_type="execute_python") }}'
                 f'ref("tests")["result"].save("{tmp_path / "out"}")',
    }
    storage = MemoryStorageBackend()
    storage.init_db()

    def run_once():
        specs = list(build_models_from_dict(models).values())
        run_id = storage.create_run(model_count=len(specs))
        results = asyncio.run(execute_run(
            run_id=run_id, ordered_models=specs, storage_backend=storage, llm_call=llm_call,
        ))
        out = {r.model_name: r for r in results}
        assert not any(r.error for r in out.values()), {k: r.error for k, r in out.items()}
        return out

    first = run_once()
    assert first["tests"].value["cache"] == "miss"
    final = tmp_path / "out" / "app"
    assert (final / "api.py").read_text() == "ROUTES = ['/health']\n"
    assert (final / "test_api.py").exists()
    assert not (final / "build.log").exists()
    assert (app / "api.py").read_text() == "ROUTES = []\n"  # source untouched

    # An upstream edit to a file neither agent owns: both replay, no agent runs.
    (app / "notes.md").write_text("v2\n")
    monkeypatch.setattr(builtin_kinds, "_exec_agent", lambda *a: pytest.fail("agent re-ran"))
    second = run_once()
    assert second["api"].value["cache"] == "replay"
    assert second["tests"].value["cache"] == "replay"
    assert (final / "notes.md").read_text() == "v2\n"


def test_text_llm_does_not_attach_the_input_tree(tmp_path, monkeypatch):
    """Merge/verify calls bypass call.llm, which would attach every promptfile."""
    from promptmap.executor import build_models_from_dict, execute_run
    from promptmap.storage.memory import MemoryStorageBackend

    monkeypatch.setattr(fa, "resolve_classifier", lambda call: None)
    app = tmp_path / "app"
    app.mkdir()
    (app / "a.py").write_text("a = 0\n")
    edits = iter(["a = 1\n"])
    monkeypatch.setattr(builtin_kinds, "_exec_agent", lambda task, call: (
        (Path(call.spec.config["agent_dir"]) / "b.py").write_text(next(edits)),
        json.dumps({"output": "ok", "logs": [], "time_run": 0.0}),
    )[1])
    seen = []

    def llm_call(prompt, files=None):
        seen.append(files)
        return "YES"

    state = tmp_path / "state"
    models = {
        "src": f'{{{{ config(model_type="execute_python", cache=False) }}}}output = Dir("{app}")',
        "b": f'{{{{ config(model_type="{fa.NAME}", promptfiles=["src"], state_dir="{state}") }}}}write b',
    }
    storage = MemoryStorageBackend()
    storage.init_db()
    for _ in range(2):
        specs = list(build_models_from_dict(models).values())
        run_id = storage.create_run(model_count=len(specs))
        results = asyncio.run(execute_run(
            run_id=run_id, ordered_models=specs, storage_backend=storage, llm_call=llm_call,
        ))
        assert not any(r.error for r in results), [r.error for r in results]
        (app / "a.py").write_text("a = 2\n")

    assert seen == [None]  # one verify call, sent without files
