"""``coding_agent`` (pi harness): Dir in, patched Dir out, with diff/verify/merge reuse."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from promptmap.files import Dir
from promptmap.model_spec import ModelSpec
from promptmap.executor.parser_model import PreviousRun
from promptmap.model_types import ModelCall, get_model_kind

from promptmap.executor import coding_agent as fa

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

    monkeypatch.setattr(fa, "_exec_agent", exec_agent)
    return fake


@pytest.fixture
def dir_source(monkeypatch):
    """A Python-running kind for tests that need a node producing a Dir.

    ``execute_python`` lives in promptmap[extras]; this stands in for it.
    """
    from promptmap.model_types import _REGISTRY, ModelKind

    async def run(rendered, call):
        ns = {"ref": lambda name: call.outputs.get(name), "Dir": Dir}
        exec(rendered, ns)  # noqa: S102
        return ns.get("output", "")

    monkeypatch.setitem(_REGISTRY, "dir_source", ModelKind(name="dir_source", exec_fn=run))


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


#: Each test's last result, handed back as ``call.previous`` the way promptmap does.
_HISTORY: dict[str, PreviousRun] = {}


def run(base: Dir, tmp_path, llm, prompt="add the feature", **config) -> dict:
    spec = ModelSpec(
        name="node", source="", model_type=fa.NAME, config=config, promptfiles_used=["up"],
    )

    async def compute(key, compute):
        value = compute()
        return await value if hasattr(value, "__await__") else value

    async def call_llm(prompt):
        llm["prompts"].append(prompt)
        return llm["replies"].pop(0)

    key = str(tmp_path)
    call = ModelCall(
        spec=spec, outputs={"up": base}, llm=call_llm, compute=compute, previous=_HISTORY.get(key),
    )
    out = asyncio.run(fa.run_coding_agent(prompt, call))
    _HISTORY[key] = PreviousRun(prompt=prompt, output=out)
    return out


def files(d: Dir) -> dict[str, bytes]:
    return {p: f.read_bytes() for p, f in d.entries.items()}


EXPECTED = {
    "shared.py": b"x = 1\n",
    "util.py": b"def u():\n    return 1\n",
    "feature.py": b"FEATURE = True\n",
}


# ---------------------------------------------------------------------------


def test_registered_and_declared():
    kind = get_model_kind(fa.NAME)
    assert kind is fa.KIND
    assert "agent_dir" not in kind.config_keys
    assert {"jev_cache_threshold", "merge_max_files"} <= kind.config_keys
    assert "base_dir" not in kind.config_keys


def test_first_run_returns_patched_tree_and_sparse_changes(tmp_path, agent, jev, llm):
    base = tree()
    out = run(base, tmp_path, llm)

    assert out["cache"] == "miss"
    assert out["output"] == "done 1"
    assert files(out["result"]) == EXPECTED
    assert out["changes"] == {k: EXPECTED[k].decode() for k in ("feature.py", "util.py")}
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


def test_chained_dir_input_from_another_agent(tmp_path, agent, jev, llm):
    first = run(tree(), tmp_path, llm)
    spec = ModelSpec(
        name="second", source="", model_type=fa.NAME,
        config={}, promptfiles_used=["first"],
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


def test_no_input_tree_starts_from_an_empty_folder(tmp_path):
    spec = ModelSpec(name="n", source="", model_type=fa.NAME, config={})
    call = ModelCall(spec=spec, outputs={}, llm=None, compute=None)
    base = fa.resolve_base(call)
    assert base.entries == {} and base.name == "n"


def test_without_an_input_tree_the_agent_creates_one(tmp_path, agent, jev, llm):
    agent["edit"] = lambda root: (root / "app.py").write_text("print('hi')\n")
    spec = ModelSpec(name="maker", source="", model_type=fa.NAME, config={})

    async def passthrough(key, compute):
        value = compute()
        return await value if hasattr(value, "__await__") else value

    call = ModelCall(spec=spec, outputs={}, llm=None, compute=passthrough)
    out = asyncio.run(fa.run_coding_agent("make an app", call))

    assert files(out["result"]) == {"app.py": b"print('hi')\n"}
    assert out["changes"] == {"app.py": "print('hi')\n"} and out["deleted"] == []


# ---------------------------------------------------------------------------
# Through promptmap's executor
# ---------------------------------------------------------------------------

def test_chain_through_the_executor(tmp_path, monkeypatch, jev, dir_source):
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

    monkeypatch.setattr(fa, "_exec_agent", exec_agent)
    llm_seen = []

    def llm_call(prompt, files=None):
        llm_seen.append(files)
        return "YES"

    models = {
        # A static node decides what the input tree is; the agents only see Dirs.
        "repo": '{{ config(model_type="dir_source", cache=False) }}'
                'from pathlib import Path\n'
                f'output = Dir({{p.name: p.read_bytes() for p in Path("{app}").iterdir() if p.suffix != ".log"}}, name="app")',
        "api": f'{{{{ config(model_type="{fa.NAME}", promptfiles="repo") }}}}add /health',
        "tests": f'{{{{ config(model_type="{fa.NAME}", promptfiles="api") }}}}test it',
        "apply": '{{ config(model_type="dir_source") }}'
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
    monkeypatch.setattr(fa, "_exec_agent", lambda *a: pytest.fail("agent re-ran"))
    second = run_once()
    assert second["api"].value["cache"] == "replay"
    assert second["tests"].value["cache"] == "replay"
    assert (final / "notes.md").read_text() == "v2\n"


def test_text_llm_does_not_attach_the_input_tree(tmp_path, monkeypatch, dir_source):
    """Merge/verify calls bypass call.llm, which would attach every promptfile."""
    from promptmap.executor import build_models_from_dict, execute_run
    from promptmap.storage.memory import MemoryStorageBackend

    monkeypatch.setattr(fa, "resolve_classifier", lambda call: None)
    app = tmp_path / "app"
    app.mkdir()
    (app / "a.py").write_text("a = 0\n")
    edits = iter(["a = 1\n"])
    monkeypatch.setattr(fa, "_exec_agent", lambda task, call: (
        (Path(call.spec.config["agent_dir"]) / "b.py").write_text(next(edits)),
        json.dumps({"output": "ok", "logs": [], "time_run": 0.0}),
    )[1])
    seen = []

    def llm_call(prompt, files=None):
        seen.append(files)
        return "YES"

    models = {
        "src": f'{{{{ config(model_type="dir_source", cache=False) }}}}output = Dir("{app}")',
        "b": f'{{{{ config(model_type="{fa.NAME}", promptfiles=["src"]) }}}}write b',
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


# ---------------------------------------------------------------------------
# Semantic (AST) gating
# ---------------------------------------------------------------------------

needs_grammar = pytest.mark.skipif(
    fa._ast_diff.language_for("x.py") is None, reason="tree-sitter-language-pack not installed"
)


@needs_grammar
def test_comment_only_upstream_change_elsewhere_is_a_hit(tmp_path, agent, jev, llm):
    run(tree(), tmp_path, llm)
    out = run(tree(shared__py=b"x = 1  # the answer\n"), tmp_path, llm)

    assert out["cache"] == "hit"
    assert jev["asked"] == [] and llm["prompts"] == [] and len(agent["tasks"]) == 1
    assert files(out["result"])["shared.py"] == b"x = 1  # the answer\n"  # upstream's edit kept


@needs_grammar
def test_comment_only_collision_merges_without_a_verdict(tmp_path, agent, jev, llm):
    run(tree(), tmp_path, llm)
    commented = b"def u():\n    # upstream note\n    return 0\n"
    llm["replies"] = ["<<<FILE util.py>>>\ndef u():\n    # upstream note\n    return 1\n<<<END>>>\n"]
    out = run(tree(util__py=commented), tmp_path, llm)

    assert out["cache"] == "merged"
    assert jev["asked"] == []                          # the merge kept our meaning exactly
    assert files(out["result"])["util.py"] == b"def u():\n    # upstream note\n    return 1\n"


@needs_grammar
def test_comment_only_collision_whose_merge_changes_meaning_is_verified(tmp_path, agent, jev, llm):
    run(tree(), tmp_path, llm)
    llm["replies"] = ["<<<FILE util.py>>>\ndef u():\n    # upstream note\n    return 2\n<<<END>>>\n"]
    out = run(tree(util__py=b"def u():\n    # upstream note\n    return 0\n"), tmp_path, llm)

    assert out["cache"] == "merged"
    assert len(jev["asked"]) == 1


@needs_grammar
def test_jev_gets_ast_diff_and_previous_sum_of_changes(tmp_path, agent, jev, llm):
    run(tree(), tmp_path, llm)
    run(tree(shared__py=b"x = 1\ndef helper(a, b):\n    return a\n"), tmp_path, llm)

    (asked,) = jev["asked"]
    upstream = asked.split("upstream_input_ast_diff:")[1].split("previous_run_sum_of_changes:")[0]
    previous = asked.split("previous_run_sum_of_changes:")[1].split("replayed_changes:")[0]
    assert "shared.py: ADDED_SYMBOL Function helper: def helper(a, b):" in upstream
    assert "+def helper(a, b):" in upstream
    assert "feature.py: ADDED_FILE" in previous
    assert "util.py: BODY_CHANGED Function u" in previous
    assert "old.txt: DELETED_FILE" in previous
    assert "+FEATURE = True" in previous


@needs_grammar
def test_output_reports_sum_of_changes(tmp_path, agent, jev, llm):
    out = run(tree(), tmp_path, llm)

    assert out["sum_of_changes"]["ast_summary"] == [
        "feature.py: ADDED_FILE",
        "old.txt: DELETED_FILE",
        "util.py: MODIFIED_FILE",
        "util.py: BODY_CHANGED Function u",
    ]
    assert "+FEATURE = True" in out["sum_of_changes"]["unified_diff"]
    assert run(tree(), tmp_path, llm)["sum_of_changes"] == out["sum_of_changes"]  # a hit reports it too


@needs_grammar
def test_rerun_context_carries_ast_summaries(tmp_path, agent, jev, llm):
    run(tree(), tmp_path, llm)
    jev["score"] = 0.0
    run(tree(shared__py=b"def x(): pass\n"), tmp_path, llm)

    task = agent["tasks"][1]
    assert "util.py: BODY_CHANGED Function u" in task
    assert "shared.py: ADDED_SYMBOL Function x" in task


# ---------------------------------------------------------------------------
# git merge-file before the LLM; state comes back through call.previous
# ---------------------------------------------------------------------------

LONG_UTIL = b"def a():\n    return 'a'\n\n\ndef b():\n    return 'b'\n\n\ndef u():\n    return 0\n"


def test_git_merge_file():
    original = b"one\ntwo\nthree\nfour\nfive\n"
    yours = b"one\ntwo\nthree\nfour\nFIVE\n"
    upstream = b"ONE\ntwo\nthree\nfour\nfive\n"
    assert fa.git_merge(original, yours, upstream) == b"ONE\ntwo\nthree\nfour\nFIVE\n"
    assert fa.git_merge(original, yours, b"one\ntwo\nthree\nfour\nfive!\n") is None  # conflict
    assert fa.git_merge(original, None, upstream) is None                            # we deleted it
    assert fa.git_merge(None, b"a\n", b"a\n") == b"a\n"                              # both created alike


def test_git_merge_without_git(monkeypatch):
    monkeypatch.setattr(fa.shutil, "which", lambda name: None)
    assert fa.git_merge(b"a\n", b"b\n", b"a\n") is None


def test_clean_git_merge_needs_no_llm_merge(tmp_path, agent, jev, llm):
    agent["edit"] = lambda root: (root / "util.py").write_text(LONG_UTIL.decode().replace("return 0", "return 1"))
    run(tree(util__py=LONG_UTIL), tmp_path, llm)
    upstream = LONG_UTIL.replace(b"return 'a'", b"return 'A'")
    out = run(tree(util__py=upstream), tmp_path, llm)

    assert out["cache"] == "merged"
    assert out["merged_by"] == {"util.py": "git"}
    assert llm["prompts"] == []                    # no LLM merge call
    assert len(jev["asked"]) == 1                  # upstream changed meaning: still verified
    merged = files(out["result"])["util.py"]
    assert b"return 'A'" in merged and b"return 1" in merged


@needs_grammar
def test_comment_only_collision_merged_by_git_costs_nothing(tmp_path, agent, jev, llm):
    agent["edit"] = lambda root: (root / "util.py").write_text(LONG_UTIL.decode().replace("return 0", "return 1"))
    run(tree(util__py=LONG_UTIL), tmp_path, llm)
    out = run(tree(util__py=b"# utilities\n" + LONG_UTIL), tmp_path, llm)

    assert out["cache"] == "merged"
    assert out["merged_by"] == {"util.py": "git"}
    assert llm["prompts"] == [] and jev["asked"] == [] and len(agent["tasks"]) == 1
    assert files(out["result"])["util.py"].startswith(b"# utilities\n")


def test_conflicting_merge_falls_back_to_llm(tmp_path, agent, jev, llm):
    run(tree(), tmp_path, llm)
    llm["replies"] = ["<<<FILE util.py>>>\ndef u():\n    log()\n    return 1\n<<<END>>>\n"]
    out = run(tree(util__py=b"def u():\n    log()\n    return 0\n"), tmp_path, llm)

    assert out["merged_by"] == {"util.py": "llm"}


def test_output_carries_what_the_next_run_needs(tmp_path, agent, jev, llm):
    base = tree()
    out = run(base, tmp_path, llm)

    assert out["base"]["hashes"] == {p: f.sha256 for p, f in base.entries.items()}
    assert set(out["base"]["originals"]) == {"util.py", "old.txt"}  # only what it touched
    last = fa.last_run(ModelCall(
        spec=None, outputs={}, llm=None, compute=None, previous=PreviousRun(prompt="p", output=out),
    ))
    assert last.ours("util.py") == EXPECTED["util.py"]
    assert last.ours("old.txt") is None
    assert last.was("old.txt") == BASE["old.txt"]
    assert last.owned == {"feature.py", "util.py", "old.txt"}


def test_previous_output_of_another_shape_is_ignored(tmp_path, agent, jev, llm):
    _HISTORY[str(tmp_path)] = PreviousRun(prompt="add the feature", output="a plain answer")
    out = run(tree(), tmp_path, llm)

    assert out["cache"] == "miss"
    assert "Previous attempt" not in agent["tasks"][0]


def test_executor_keeps_no_state_and_without_warm_start_a_template_edit_starts_afresh(tmp_path, monkeypatch, jev, dir_source):
    from promptmap.executor import build_models_from_dict, execute_run
    from promptmap.storage.sqlite import SQLiteStorageBackend

    tasks = []

    def exec_agent(task, call):
        tasks.append(task)
        (Path(call.spec.config["agent_dir"]) / "out.py").write_text("x = 1\n")
        return json.dumps({"output": "ok", "logs": [], "time_run": 0.0})

    monkeypatch.setattr(fa, "_exec_agent", exec_agent)
    monkeypatch.chdir(tmp_path)
    storage = SQLiteStorageBackend(tmp_path / "db" / "promptmap.db")
    storage.init_db()
    src = '{{ config(model_type="dir_source") }}output = Dir({"a.py": b"a = 1\\n"}, name="app")'

    def run_once(agent_source):
        specs = list(build_models_from_dict({"src": src, "agent": agent_source}).values())
        run_id = storage.create_run(model_count=len(specs))
        results = {r.model_name: r for r in asyncio.run(execute_run(
            run_id=run_id, ordered_models=specs, storage_backend=storage, llm_call=lambda p: "YES",
        ))}
        assert not results["agent"].error, results["agent"].error
        return results["agent"].value

    first = f'{{{{ config(model_type="{fa.NAME}", promptfiles="src", warm_start=false) }}}}write out.py'
    assert run_once(first)["cache"] == "miss"
    assert run_once(first)["cache"] == "hit"       # read back from the SQLite run database
    assert len(tasks) == 1
    assert not (tmp_path / ".promptmap").exists()  # nothing kept outside the database

    edited = first + "\nKeep it short."
    assert run_once(edited)["cache"] == "miss"     # an edited template has no previous
    assert "Previous attempt" not in tasks[1]


def test_result_is_the_only_file_output(tmp_path, agent, jev, llm):
    from promptmap.files import encode_output, iter_files

    out = run(tree(), tmp_path, llm)
    labels = {label for label, _ in iter_files(out)}
    assert labels == {f"result/{p}" for p in EXPECTED}
    json.loads(encode_output({k: v for k, v in out.items() if k != "result"}))  # plain data


PNG = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 64


def test_binaries_are_kept_as_hashes_only():
    import hashlib

    assert fa.encode_bytes(b"text") == "text"
    stored = fa.encode_bytes(PNG)
    assert stored == {"sha256": hashlib.sha256(PNG).hexdigest()}
    # Bytes come back from a tree holding that content, else a stand-in.
    assert fa.decode_bytes(stored, Dir({"a.png": PNG}), "a.png") == PNG
    stand_in = fa.decode_bytes(stored)
    assert stand_in != PNG and fa._text(stand_in) is None
    assert stand_in == fa.decode_bytes(stored, Dir({"a.png": b"other"}), "a.png")
    assert fa.git_merge(stand_in, PNG, PNG + b"x") is None  # binaries are never merged


def test_binary_files_round_trip_through_reuse(tmp_path, agent, jev, llm, monkeypatch):
    base = tree(**{"logo.png": PNG})

    def edit(root: Path) -> None:
        feature_edit(root)
        (root / "logo.png").write_bytes(PNG[::-1])   # agent rewrites a binary
        (root / "new.pdf").write_bytes(b"%PDF\xff" * 1000)

    agent["edit"] = edit
    out = run(base, tmp_path, llm)
    for blob in (out["base"]["originals"]["logo.png"], out["changes"]["logo.png"], out["changes"]["new.pdf"]):
        assert set(blob) == {"sha256"}                # no bytes in the stored state

    assert run(base, tmp_path, llm)["cache"] == "hit"
    again = run(tree(**{"logo.png": PNG, "shared.py": b"x = 2\n"}), tmp_path, llm)
    assert again["cache"] == "replay"
    assert again["result"]["logo.png"].read_bytes() == PNG[::-1]
    assert again["result"]["new.pdf"].read_bytes() == b"%PDF\xff" * 1000
    assert len(agent["tasks"]) == 1                   # never re-ran the agent


# ---------------------------------------------------------------------------
# warm_start: a .prompt edit continues from the old template's last result
# ---------------------------------------------------------------------------

def _from_older_template(tmp_path) -> None:
    """Mark this test's stored previous run as coming from an older template."""
    import dataclasses

    key = str(tmp_path)
    _HISTORY[key] = dataclasses.replace(_HISTORY[key], same_template=False)


def test_warm_start_continues_from_previous_changes(tmp_path, agent, jev, llm):
    run(tree(), tmp_path, llm, warm_start="true")
    _from_older_template(tmp_path)

    seen = {}

    def tweak(root: Path) -> None:
        seen["files"] = {p.name for p in root.iterdir()}
        seen["feature"] = (root / "feature.py").read_text()
        (root / "feature.py").write_text("FEATURE = 'loud'\n")

    agent["edit"] = tweak
    out = run(tree(), tmp_path, llm, prompt="add the feature, loudly", warm_start="true")

    assert out["cache"] == "warm"
    # The agent started from its previous result, not the bare input.
    assert seen["feature"] == "FEATURE = True\n" and "old.txt" not in seen["files"]
    task = agent["tasks"][1]
    assert "ALREADY APPLIED" in task
    assert "+add the feature, loudly" in task and "-add the feature" in task
    # Changes are still measured against the real input tree.
    assert files(out["result"]) == {**EXPECTED, "feature.py": b"FEATURE = 'loud'\n"}
    assert set(out["changes"]) == {"feature.py", "util.py"} and out["deleted"] == ["old.txt"]


def test_warm_start_falls_back_when_upstream_edited_owned_files(tmp_path, agent, jev, llm):
    run(tree(), tmp_path, llm, warm_start="true")
    _from_older_template(tmp_path)

    out = run(tree(util__py=b"def u():\n    return 5\n"), tmp_path, llm, prompt="add it v2", warm_start="true")

    assert out["cache"] == "miss"
    assert "the prompt was edited" in agent["tasks"][1]


def test_warm_start_is_the_default_through_the_executor(tmp_path, monkeypatch, jev, dir_source):
    from promptmap.executor import build_models_from_dict, execute_run
    from promptmap.storage.sqlite import SQLiteStorageBackend

    tasks, starts, labels = [], [], []

    def exec_agent(task, call):
        root = Path(call.spec.config["agent_dir"])
        tasks.append(task)
        starts.append(sorted(p.name for p in root.iterdir()))
        (root / "out.py").write_text(f"x = {len(tasks)}\n")
        return json.dumps({"output": "ok", "logs": [], "time_run": 0.0})

    monkeypatch.setattr(fa, "_exec_agent", exec_agent)
    monkeypatch.chdir(tmp_path)
    storage = SQLiteStorageBackend(tmp_path / "db" / "promptmap.db")
    storage.init_db()
    src = '{{ config(model_type="dir_source") }}output = Dir({"a.py": b"a = 1\\n"}, name="app")'

    def run_once(agent_source):
        specs = list(build_models_from_dict({"src": src, "agent": agent_source}).values())
        run_id = storage.create_run(model_count=len(specs))
        results = {r.model_name: r for r in asyncio.run(execute_run(
            run_id=run_id, ordered_models=specs, storage_backend=storage, llm_call=lambda p: "YES",
        ))}
        assert not results["agent"].error, results["agent"].error
        labels.append(results["agent"].cache)
        return results["agent"].value

    first = f'{{{{ config(model_type="{fa.NAME}", promptfiles="src") }}}}write out.py'
    assert run_once(first)["cache"] == "miss"

    edited = first + "\nKeep it short."
    out = run_once(edited)
    assert out["cache"] == "warm"
    assert starts[1] == ["a.py", "out.py"]          # began from the previous result
    assert "+Keep it short." in tasks[1]
    assert run_once(edited)["cache"] == "hit"        # the edited template now has its own previous
    assert len(tasks) == 2

    # clear-cache for this model: the next run is cold, even after another edit.
    assert storage.clear_cache(["agent"], previous=True) > 0
    assert run_once(edited)["cache"] == "miss"
    assert "Previous attempt" not in tasks[2] and "ALREADY APPLIED" not in tasks[2]
    assert starts[2] == ["a.py"]
    assert run_once(edited + "\nAnd tidy.")["cache"] == "warm"  # warm again from the cold run
    # What `promptmap run` shows next to the model, run by run.
    assert labels == ["miss", "warm", "hit", "miss", "warm"]


def test_warm_start_is_on_by_default_for_coding_agent_only():
    from promptmap.model_types import config_value

    agent = ModelSpec(name="a", source="", model_type=fa.NAME, config={})
    off = ModelSpec(name="a", source="", model_type=fa.NAME, config={"warm_start": False})
    plain = ModelSpec(name="p", source="", model_type="", config={})
    assert config_value(agent, "warm_start") is True
    assert config_value(off, "warm_start") is False
    assert config_value(plain, "warm_start", False) is False


def test_rename_keeps_coding_agent_history_and_downstream_reuse(tmp_path, monkeypatch, jev, dir_source):
    from promptmap.executor import execute_run, load_models
    from promptmap.rename import apply_rename, plan_rename
    from promptmap.storage.sqlite import SQLiteStorageBackend

    tasks = []

    def exec_agent(task, call):
        tasks.append(task)
        (Path(call.spec.config["agent_dir"]) / "out.py").write_text("x = 1\n")
        return json.dumps({"output": "ok", "logs": [], "time_run": 0.0})

    monkeypatch.setattr(fa, "_exec_agent", exec_agent)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "map").mkdir()
    (tmp_path / "map" / "src.prompt").write_text(
        '{{ config(model_type="dir_source") }}output = Dir({"a.py": b"a = 1\\n"}, name="app")')
    (tmp_path / "map" / "build.prompt").write_text(
        f'{{{{ config(model_type="{fa.NAME}", promptfiles="src") }}}}write out.py')
    (tmp_path / "map" / "tests.prompt").write_text(
        f'{{{{ config(model_type="{fa.NAME}", promptfiles="build.result") }}}}add tests')
    storage = SQLiteStorageBackend(tmp_path / "db" / "promptmap.db")
    storage.init_db()

    def run_once():
        specs = list(load_models(tmp_path / "map").values())
        run_id = storage.create_run(model_count=len(specs))
        results = {r.model_name: r for r in asyncio.run(execute_run(
            run_id=run_id, ordered_models=specs,
            storage_backend=storage, llm_call=lambda p: "YES",
        ))}
        assert all(not r.error for r in results.values()), {k: r.error for k, r in results.items()}
        return {k: r.cache for k, r in results.items()}

    assert run_once() == {"src": "", "build": "miss", "tests": "miss"}

    plan = plan_rename(tmp_path / "map", "build", "build_app")
    apply_rename(plan)
    assert storage.rename_model("build", "build_app", plan.retemplate) == 1
    assert 'promptfiles="build_app.result"' in (tmp_path / "map" / "tests.prompt").read_text()

    # Both reuse their last change: the renamed model by its moved history, the
    # downstream one because its stored template hash followed its rewritten file.
    assert run_once() == {"src": "", "build_app": "hit", "tests": "hit"}
    assert len(tasks) == 2


def test_independent_coding_agents_run_in_parallel(tmp_path, monkeypatch, jev, dir_source):
    """Two agents that do not depend on each other work side by side; the
    model that reads both starts only after both finished."""
    import threading
    import time

    from promptmap.executor import build_models_from_dict, execute_run
    from promptmap.storage import MemoryStorageBackend

    both_running = threading.Barrier(2, timeout=5)  # breaks unless both agents run at once
    spans: dict[str, tuple[float, float]] = {}

    def exec_agent(task, call):
        name = call.spec.name
        began = time.monotonic()
        if name in ("api", "ui"):
            both_running.wait()
        (Path(call.spec.config["agent_dir"]) / f"{name}.py").write_text("x = 1\n")
        spans[name] = (began, time.monotonic())
        return json.dumps({"output": "ok", "logs": [], "time_run": 0.0})

    monkeypatch.setattr(fa, "_exec_agent", exec_agent)
    src = '{{ config(model_type="dir_source") }}output = Dir({"a.py": b"a = 1\\n"}, name="app")'
    agent = f'{{{{ config(model_type="{fa.NAME}", promptfiles="src") }}}}'
    models = build_models_from_dict({
        "src": src,
        "api": agent + "build the api",
        "ui": agent + "build the ui",
        "review": f'{{{{ config(model_type="{fa.NAME}", promptfiles="api.result") }}}}'
                  "review {{ ref('ui').output }}",
    })
    storage = MemoryStorageBackend()
    results = asyncio.run(execute_run(
        run_id=storage.create_run(model_count=4), ordered_models=list(models.values()),
        storage_backend=storage, llm_call=lambda p: "YES",
    ))
    errors = {r.model_name: r.error for r in results if r.error}
    assert not errors, errors
    assert spans["review"][0] >= max(spans["api"][1], spans["ui"][1])
    assert [r.model_name for r in results][-1] == "review"


def test_max_parallel_caps_how_many_models_run_at_once():
    from promptmap.executor import build_models_from_dict, execute_run
    from promptmap.storage import MemoryStorageBackend

    live = {"now": 0, "peak": 0}

    async def llm(prompt):
        live["now"] += 1
        live["peak"] = max(live["peak"], live["now"])
        await asyncio.sleep(0.02)
        live["now"] -= 1
        return "ok"

    models = build_models_from_dict({f"m{i}": f"prompt {i}" for i in range(5)})
    for limit, expected in ((2, 2), (1, 1), (8, 5)):
        live.update(now=0, peak=0)
        storage = MemoryStorageBackend()
        results = asyncio.run(execute_run(
            run_id=storage.create_run(model_count=5), ordered_models=list(models.values()),
            storage_backend=storage, llm_call=llm, max_parallel=limit,
        ))
        assert all(r.status == "success" for r in results)
        assert live["peak"] == expected, (limit, live["peak"])
