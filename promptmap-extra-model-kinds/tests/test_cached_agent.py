"""``cached_local_agent``: workspace-diff cache, Jev gate and sparse replay."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from promptmap.executor import builtin_kinds
from promptmap.model_spec import ModelSpec
from promptmap.model_types import ModelCall, get_model_kind

from promptmap_extra_model_kinds import cached_agent_exec as cached


@pytest.fixture
def ws(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    (root / "shared.py").write_text("x = 1\n")
    (root / "util.py").write_text("def u():\n    return 0\n")
    (root / "old.txt").write_text("remove me\n")
    return root


@pytest.fixture
def agent(monkeypatch):
    """A fake local agent: the edit it makes is set per test; calls are counted."""
    fake = {"calls": [], "edit": None}

    def exec_agent(task, call):
        fake["calls"].append(task)
        root = Path(call.spec.config["agent_dir"])
        fake["edit"](root)
        return json.dumps({"output": f"done {len(fake['calls'])}", "logs": [], "time_run": 0.1})

    monkeypatch.setattr(builtin_kinds, "_exec_agent", exec_agent)
    return fake


@pytest.fixture
def jev(monkeypatch):
    """A fake classifier with a settable score; each question asked is recorded."""
    fake = {"score": 1.0, "asked": []}

    def classify(state, question):
        fake["asked"].append((state, question))
        return fake["score"]

    monkeypatch.setattr(cached, "resolve_classifier", lambda call: classify)
    return fake


def feature_edit(root: Path) -> None:
    """The node's work: add a feature file, patch util.py, delete old.txt."""
    (root / "feature.py").write_text("FEATURE = True\n")
    (root / "util.py").write_text("def u():\n    return 1\n")
    (root / "old.txt").unlink()


def run(root: Path, tmp_path: Path, prompt="add the feature", **config):
    spec = ModelSpec(
        name="node", source="", model_type=cached.NAME,
        config={"agent_dir": str(root), "state_dir": str(tmp_path / "state"), **config},
    )

    async def compute(key, compute):
        value = compute()
        return await value if hasattr(value, "__await__") else value

    call = ModelCall(spec=spec, outputs={}, llm=None, compute=compute)
    return asyncio.run(cached.run_cached_agent(prompt, call))


def state(tmp_path) -> dict:
    return json.loads((tmp_path / "state" / "node.json").read_text())


def test_registered_as_agent_derivative():
    kind = get_model_kind(cached.NAME)
    assert kind is cached.KIND
    assert builtin_kinds.AGENT.config_keys < kind.config_keys
    assert {"jev_cache_threshold", "force_refresh", "state_dir"} <= kind.config_keys


def test_entry_point_declared():
    from importlib.metadata import entry_points

    eps = {ep.name: ep.value for ep in entry_points(group="promptmap.model_kinds")}
    assert eps.get(cached.NAME) == "promptmap_extra_model_kinds.cached_agent_exec"


def test_initial_run_records_only_the_sparse_delta(ws, tmp_path, agent, jev):
    agent["edit"] = feature_edit
    result = run(ws, tmp_path)

    assert result["cache"] == "miss"
    assert result["output"] == "done 1"
    assert cached.OUTCOME_CONTRACT.strip() in agent["calls"][0]
    saved = state(tmp_path)
    assert saved["cached_replacements"] == {
        "feature.py": "FEATURE = True\n",
        "util.py": "def u():\n    return 1\n",
    }
    assert saved["deleted_files"] == ["old.txt"]
    assert "shared.py" not in saved["cached_replacements"]
    assert set(saved["last_received_files"]) == {"shared.py", "util.py", "old.txt"}
    assert result["changed_files"] == ["feature.py", "util.py"]


def test_zero_diff_is_a_fast_hit_without_agent_or_jev(ws, tmp_path, agent, jev):
    agent["edit"] = feature_edit
    before = cached.snapshot(ws)
    run(ws, tmp_path)
    # The next pipeline run starts from the same incoming workspace.
    _reset(ws, before)

    result = run(ws, tmp_path)

    assert result["cache"] == "hit"
    assert result["output"] == "done 1"
    assert len(agent["calls"]) == 1
    assert jev["asked"] == []
    assert (ws / "feature.py").read_text() == "FEATURE = True\n"
    assert not (ws / "old.txt").exists()


def test_jev_hit_on_unrelated_upstream_edit_keeps_that_edit(ws, tmp_path, agent, jev):
    agent["edit"] = feature_edit
    before = cached.snapshot(ws)
    run(ws, tmp_path)
    _reset(ws, before)
    (ws / "shared.py").write_text("x = 2\n")  # upstream changes an unrelated file

    result = run(ws, tmp_path)

    assert result["cache"] == "jev_hit"
    assert len(agent["calls"]) == 1
    state_text, question = jev["asked"][0]
    assert question == cached.JEV_QUESTION
    assert "add the feature" in state_text
    assert '"feature.py"' in state_text and '"util.py"' in state_text
    assert "-x = 1" in state_text and "+x = 2" in state_text
    assert (ws / "shared.py").read_text() == "x = 2\n"  # upstream edit survives
    assert (ws / "feature.py").read_text() == "FEATURE = True\n"
    assert (ws / "util.py").read_text() == "def u():\n    return 1\n"
    assert not (ws / "old.txt").exists()
    assert state(tmp_path)["last_received_files"]["shared.py"] == "x = 2\n"


def test_jev_miss_on_breaking_diff_reruns_agent(ws, tmp_path, agent, jev):
    agent["edit"] = feature_edit
    before = cached.snapshot(ws)
    run(ws, tmp_path)
    _reset(ws, before)
    (ws / "shared.py").write_text("raise SystemExit\n")
    jev["score"] = 0.4

    result = run(ws, tmp_path)

    assert result["cache"] == "miss"
    assert len(jev["asked"]) == 1
    assert len(agent["calls"]) == 2
    assert result["output"] == "done 2"


def test_threshold_is_configurable(ws, tmp_path, agent, jev):
    agent["edit"] = feature_edit
    before = cached.snapshot(ws)
    run(ws, tmp_path)
    _reset(ws, before)
    (ws / "shared.py").write_text("x = 2\n")
    jev["score"] = 0.6

    assert run(ws, tmp_path, jev_cache_threshold="0.5")["cache"] == "jev_hit"


@pytest.mark.parametrize("upstream", [
    lambda ws: (ws / "util.py").write_text("def u():\n    return 99\n"),   # file we modified
    lambda ws: (ws / "feature.py").write_text("FEATURE = False\n"),       # file we created
    lambda ws: (ws / "old.txt").write_text("changed upstream\n"),         # file we deleted
])
def test_write_collision_forces_a_miss_without_asking_jev(ws, tmp_path, agent, jev, upstream):
    agent["edit"] = feature_edit
    before = cached.snapshot(ws)
    run(ws, tmp_path)
    _reset(ws, before)
    upstream(ws)
    agent["edit"] = lambda root: (root / "feature.py").write_text("FEATURE = 2\n")

    result = run(ws, tmp_path)

    assert result["cache"] == "miss"
    assert jev["asked"] == []
    assert len(agent["calls"]) == 2


def test_changed_prompt_is_a_miss(ws, tmp_path, agent, jev):
    agent["edit"] = feature_edit
    before = cached.snapshot(ws)
    run(ws, tmp_path)
    _reset(ws, before)
    agent["edit"] = lambda root: None

    assert run(ws, tmp_path, prompt="something else")["cache"] == "miss"


def test_force_refresh_bypasses_the_gate(ws, tmp_path, agent, jev):
    agent["edit"] = feature_edit
    before = cached.snapshot(ws)
    run(ws, tmp_path)
    _reset(ws, before)

    assert run(ws, tmp_path, force_refresh="true")["cache"] == "miss"
    assert len(agent["calls"]) == 2


def test_no_classifier_means_miss_on_nonempty_diff(ws, tmp_path, agent, monkeypatch):
    monkeypatch.setattr(cached, "resolve_classifier", lambda call: None)
    agent["edit"] = feature_edit
    before = cached.snapshot(ws)
    run(ws, tmp_path)
    _reset(ws, before)
    (ws / "shared.py").write_text("x = 2\n")

    assert run(ws, tmp_path)["cache"] == "miss"


def test_rerun_ignores_a_cached_agent_result_for_the_same_prompt(ws, tmp_path, agent, jev):
    """A memoised agent answer must not stand in for a real run: nothing to diff."""
    agent["edit"] = feature_edit
    spec = ModelSpec(
        name="node", source="", model_type=cached.NAME,
        config={"agent_dir": str(ws), "state_dir": str(tmp_path / "state")},
    )
    memo = {}

    async def compute(key, compute):
        if key not in memo:
            value = compute()
            memo[key] = await value if hasattr(value, "__await__") else value
        return memo[key]

    call = ModelCall(spec=spec, outputs={}, llm=None, compute=compute)
    before = cached.snapshot(ws)
    asyncio.run(cached.run_cached_agent("task", call))
    _reset(ws, before)
    (tmp_path / "state" / "node.json").unlink()  # state lost, executor cache kept

    result = asyncio.run(cached.run_cached_agent("task", call))

    assert len(agent["calls"]) == 2
    assert result["changed_files"] == ["feature.py", "util.py"]


def test_binary_files_round_trip(ws, tmp_path, agent, jev):
    blob = bytes(range(256))
    agent["edit"] = lambda root: (root / "img.bin").write_bytes(blob)
    before = cached.snapshot(ws)
    run(ws, tmp_path)
    _reset(ws, before)

    assert run(ws, tmp_path)["cache"] == "hit"
    assert (ws / "img.bin").read_bytes() == blob


def test_state_dir_inside_workspace_is_not_snapshotted(ws, tmp_path, agent, jev):
    agent["edit"] = feature_edit
    spec_state = ws / ".cache-state"
    before = cached.snapshot(ws)
    run(ws, tmp_path, state_dir=str(spec_state))
    _reset(ws, before, keep={".cache-state"})

    assert run(ws, tmp_path, state_dir=str(spec_state))["cache"] == "hit"


def _reset(root: Path, files: dict, keep: set[str] = frozenset()) -> None:
    """Put *root* back to the snapshot *files* (the next run's incoming workspace)."""
    for path in sorted(root.rglob("*"), reverse=True):
        rel = path.relative_to(root).parts
        if rel[0] in keep:
            continue
        if path.is_file():
            path.unlink()
    cached.replay(root, files, [])


def test_chained_nodes_through_the_executor(tmp_path, monkeypatch, jev):
    """Two agents on one workspace, run twice through promptmap with a shared cache."""
    from promptmap.executor import build_models_from_dict, execute_run
    from promptmap.storage.memory import MemoryStorageBackend

    root = tmp_path / "ws"
    calls = []

    def exec_agent(task, call):
        calls.append(call.spec.name)
        ws = Path(call.spec.config["agent_dir"])
        if call.spec.name == "base":
            (ws / "api.py").write_text(f"VERSION = {len(calls)}\n")
        else:
            (ws / "tests.py").write_text("assert True\n")
        return json.dumps({"output": call.spec.name, "logs": [], "time_run": 0.0})

    monkeypatch.setattr(builtin_kinds, "_exec_agent", exec_agent)
    cfg = f'model_type="{cached.NAME}", agent_dir="{root}", state_dir="{tmp_path / "state"}"'
    models = {
        "base": f"{{{{ config({cfg}) }}}}write the api",
        "tests": f"{{{{ config({cfg}) }}}}{{{{ ref('base')['output'] }}}}: write tests",
    }
    storage = MemoryStorageBackend()
    storage.init_db()

    def run_once(force_base=False):
        specs = build_models_from_dict(models)
        if force_base:
            specs["base"] = specs["base"].derive(config={**specs["base"].config, "force_refresh": "true"})
        run_id = storage.create_run(model_count=len(specs))
        results = asyncio.run(execute_run(
            run_id=run_id, ordered_models=list(specs.values()),
            storage_backend=storage, llm_call=lambda p, c=None: "",
        ))
        return {r.model_name: r for r in results}

    first = run_once()
    assert not any(r.error for r in first.values()), {k: r.error for k, r in first.items()}
    assert calls == ["base", "tests"]

    # Upstream re-runs and changes api.py (not a file "tests" owns): Jev replays "tests".
    second = run_once(force_base=True)
    assert calls == ["base", "tests", "base"]
    assert second["tests"].value["cache"] == "jev_hit"
    assert (root / "api.py").read_text() == "VERSION = 3\n"
    assert (root / "tests.py").read_text() == "assert True\n"


def test_persistent_workspace_own_leftovers_are_not_upstream_changes(ws, tmp_path, agent, jev):
    """No reset between runs: the node's own previous edits are still on disk."""
    agent["edit"] = feature_edit
    run(ws, tmp_path)

    assert run(ws, tmp_path)["cache"] == "hit"
    assert jev["asked"] == []

    (ws / "shared.py").write_text("x = 2\n")
    assert run(ws, tmp_path)["cache"] == "jev_hit"
    assert "feature.py" not in jev["asked"][0][0].split("upstream_diff:")[1]


def test_rerun_keeps_untouched_leftovers_in_the_delta(ws, tmp_path, agent, jev):
    agent["edit"] = feature_edit
    run(ws, tmp_path)
    (ws / "util.py").write_text("def u():\n    return 7\n")  # upstream collides on util.py
    agent["edit"] = lambda root: (root / "util.py").write_text("def u():\n    return 8\n")

    assert run(ws, tmp_path)["cache"] == "miss"
    saved = state(tmp_path)
    assert saved["cached_replacements"] == {
        "feature.py": "FEATURE = True\n",           # still ours, though untouched this time
        "util.py": "def u():\n    return 8\n",
    }
    assert saved["deleted_files"] == ["old.txt"]
    assert saved["last_received_files"]["util.py"] == "def u():\n    return 7\n"
    assert "feature.py" not in saved["last_received_files"]
