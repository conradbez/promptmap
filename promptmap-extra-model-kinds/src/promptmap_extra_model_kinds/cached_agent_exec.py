"""
``cached_local_agent`` — the local ``agent`` kind, gated by a workspace-diff cache.

A chain of ``agent`` models sharing one ``agent_dir`` re-runs every downstream
agent whenever anything upstream changes.  This kind wraps the built-in
``agent`` kind (same config keys, same backends, same output shape) and keeps,
per model, the files it last *received* and the sparse set of files it
*changed*.  On the next run:

1. **Fast path** — same prompt, workspace identical to last time: replay.
2. **Write-collision guard** — upstream touched a file this model also wrote
   or deleted: re-run, so the replay can never clobber the newer upstream edit.
3. **Jev gate** — otherwise ask a classifier (``classify_call`` in client.py,
   else TypeSafe's hosted Jev when ``TYPESAFE_API_KEY`` is set) whether the
   previous changes still satisfy the prompt given the upstream diff.
   P(yes) >= ``jev_cache_threshold`` replays; anything else re-runs.

The workspace may persist between runs: a file still exactly as this model
left it last time counts as its own leftover, not an upstream change.

*Replay* writes only this model's cached replacements over the current
workspace and removes the files it deleted; every other file, including
fresh upstream edits, is left alone.  No agent tokens are spent.

A *re-run* hands the task to the ``agent`` kind with an outcome contract
appended, then diffs the workspace before and after to record what the agent
created, modified or deleted.

Config keys, beyond the ``agent`` kind's own:

``jev_cache_threshold``  P(yes) needed to replay over a non-empty upstream
                         diff (default 0.85)
``force_refresh``        always re-run the agent (default false)
``state_dir``            where ``<model>.json`` state lives (default
                         ``.promptmap/state``, relative to the project root —
                         the folder holding client.py — else to ``agent_dir``)

The output is the ``agent`` kind's dict plus ``"cache"`` (``"miss"``,
``"hit"`` or ``"jev_hit"``), ``"changed_files"`` and ``"deleted_files"``.
Files the agent wrote to its output directory are not replayed on a hit.
"""

from __future__ import annotations

import base64
import dataclasses
import difflib
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable

from promptmap.executor import builtin_kinds
from promptmap.model_types import ModelCall, register_model_kind

log = logging.getLogger("promptmap")

NAME = "cached_local_agent"
DEFAULT_THRESHOLD = 0.85
DEFAULT_STATE_DIR = ".promptmap/state"

#: Appended to the task on a re-run so the recorded delta stays sparse.
OUTCOME_CONTRACT = (
    "\n\nInspect the shared workspace and achieve the requested outcome. Modify "
    "or create only the files necessary to satisfy the outcome and leave all "
    "unrelated files untouched."
)

#: The Noul question put to the classifier when upstream changed.
JEV_QUESTION = (
    "Given upstream_diff, does the previous implementation in node_owned_files "
    "still completely satisfy outcome_goal without requiring any code or file changes?"
)

#: Never part of the shared workspace snapshot.
_IGNORED_DIRS = frozenset({".git", ".promptmap", "__pycache__", "node_modules", ".venv"})
_IGNORED_PREFIXES = (".promptmap_out_",)

#: A file's content in state: text as-is, binary as {"base64": ...}.
Content = str | dict


# ---------------------------------------------------------------------------
# Workspace snapshots
# ---------------------------------------------------------------------------

def snapshot(root: Path, exclude: Path | None = None) -> dict[str, Content]:
    """Every regular file under *root*, by POSIX relative path."""
    files: dict[str, Content] = {}
    exclude = exclude.resolve() if exclude is not None else None
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in _IGNORED_DIRS
            and not d.startswith(_IGNORED_PREFIXES)
            and (exclude is None or (here / d).resolve() != exclude)
        )
        for name in sorted(filenames):
            path = here / name
            if path.is_symlink() or not path.is_file():
                continue
            files[path.relative_to(root).as_posix()] = _encode(path.read_bytes())
    return files


def _encode(data: bytes) -> Content:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return {"base64": base64.b64encode(data).decode("ascii")}


def _decode(content: Content) -> bytes:
    if isinstance(content, dict):
        return base64.b64decode(content["base64"])
    return content.encode("utf-8")


def changed_paths(before: dict[str, Content], after: dict[str, Content]) -> set[str]:
    """Paths added, modified or removed between two snapshots."""
    return {p for p in before.keys() | after.keys() if before.get(p) != after.get(p)}


def unified_diff(before: dict[str, Content], after: dict[str, Content]) -> str:
    """A unified diff of two snapshots ("" when identical)."""
    chunks: list[str] = []
    for path in sorted(changed_paths(before, after)):
        old, new = before.get(path), after.get(path)
        if isinstance(old, dict) or isinstance(new, dict):
            chunks.append(f"Binary file {path} changed\n")
            continue
        chunks.extend(difflib.unified_diff(
            (old or "").splitlines(keepends=True),
            (new or "").splitlines(keepends=True),
            fromfile=f"a/{path}" if old is not None else "/dev/null",
            tofile=f"b/{path}" if new is not None else "/dev/null",
        ))
        if chunks and not chunks[-1].endswith("\n"):
            chunks[-1] += "\n"
    return "".join(chunks)


def replay(root: Path, replacements: dict[str, Content], deleted: list[str]) -> None:
    """Write *replacements* over *root* and remove *deleted*; touch nothing else."""
    for rel, content in replacements.items():
        path = _inside(root, rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_decode(content))
    for rel in deleted:
        _inside(root, rel).unlink(missing_ok=True)


def _inside(root: Path, rel: str) -> Path:
    path = (root / rel).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"{NAME}: refusing path outside agent_dir: {rel!r}")
    return path


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def state_path(call: ModelCall, workdir: Path) -> Path:
    configured = Path(str(call.spec.config.get("state_dir") or DEFAULT_STATE_DIR)).expanduser()
    if not configured.is_absolute():
        configured = (_project_root(call) or workdir) / configured
    return configured / f"{call.spec.name}.json"


def _project_root(call: ModelCall) -> Path | None:
    path = call.spec.path
    if not path.is_file():
        return None
    for parent in path.resolve().parents:
        if (parent / "client.py").is_file():
            return parent
    return path.resolve().parent


def load_state(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True))
    os.replace(tmp, path)


def prompt_hash(rendered: str, config: dict) -> str:
    """The rendered prompt plus the agent settings that change what it would do."""
    salt = json.dumps({k: str(config.get(k, "")) for k in ("agent_backend", "agent_model")}, sort_keys=True)
    return hashlib.sha256((rendered + "\x00" + salt).encode()).hexdigest()


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

def resolve_classifier(call: ModelCall) -> Callable[[str, str], float] | None:
    """client.py's ``classify_call``, else hosted Jev if ``TYPESAFE_API_KEY`` is set."""
    from promptmap.llm import client_module_for

    module = client_module_for(call.spec.path)
    found = getattr(module, "classify_call", None) if module is not None else None
    if found is not None:
        return found
    if os.environ.get("TYPESAFE_API_KEY") or "TYPESAFE_API_KEY" in call.secrets:
        from promptmap.classifier import systemone_classifier

        key = call.secrets["TYPESAFE_API_KEY"] if "TYPESAFE_API_KEY" in call.secrets else None
        return systemone_classifier(api_key=key)
    return None


def jev_state(goal: str, owned: list[str], diff: str) -> str:
    return (
        f"outcome_goal:\n{goal}\n\n"
        f"node_owned_files:\n{json.dumps(sorted(owned))}\n\n"
        f"upstream_diff:\n{diff}"
    )


def upstream_view(state: dict | None, current: dict[str, Content]) -> dict[str, Content]:
    """*current* with this model's own leftovers from its last run taken back out.

    A workspace that persists between runs still holds what this model wrote
    last time.  A file still exactly as this model left it (or still gone, if
    it deleted it) is not an upstream change, so it reads as it was received.
    A file upstream has since changed differs from both and stays as it is.
    """
    if not state:
        return current
    received = state.get("last_received_files") or {}
    view = dict(current)
    for path, content in (state.get("cached_replacements") or {}).items():
        if current.get(path) == content:
            _restore(view, path, received)
    for path in state.get("deleted_files") or []:
        if path not in current:
            _restore(view, path, received)
    return view


def _restore(view: dict[str, Content], path: str, received: dict[str, Content]) -> None:
    if path in received:
        view[path] = received[path]
    else:
        view.pop(path, None)


def decide(
    state: dict | None,
    rendered: str,
    phash: str,
    current: dict[str, Content],
    call: ModelCall,
) -> str:
    """``"hit"``, ``"jev_hit"`` or ``"miss"`` for this run.

    *current* is the upstream view of the workspace (:func:`upstream_view`).
    """
    if _truthy(call.spec.config.get("force_refresh")):
        return "miss"
    if not state or state.get("prompt_hash") != phash:
        return "miss"
    previous = state.get("last_received_files") or {}
    upstream = changed_paths(previous, current)
    if not upstream:
        return "hit"
    owned = set(state.get("cached_replacements") or {}) | set(state.get("deleted_files") or [])
    if upstream & owned:
        return "miss"  # write collision: replaying would clobber an upstream edit

    classify = resolve_classifier(call)
    if classify is None:
        return "miss"
    threshold = _threshold(call.spec.config.get("jev_cache_threshold"))
    try:
        score = float(classify(
            jev_state(rendered, list(state.get("cached_replacements") or {}), unified_diff(previous, current)),
            JEV_QUESTION,
        ))
    except Exception:
        log.warning("%s '%s': classifier failed; re-running the agent", NAME, call.spec.name, exc_info=True)
        return "miss"
    return "jev_hit" if score >= threshold else "miss"


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _threshold(value: Any) -> float:
    if value is None or value == "":
        return DEFAULT_THRESHOLD
    threshold = float(value)
    if not 0 <= threshold <= 1:
        raise ValueError(f"jev_cache_threshold must be between 0 and 1; got {threshold}.")
    return threshold


# ---------------------------------------------------------------------------
# exec_fn
# ---------------------------------------------------------------------------

async def run_cached_agent(rendered: str, call: ModelCall) -> dict:
    """The ``agent`` kind, replaying its last workspace changes when still valid."""
    config = call.spec.config
    if not config.get("agent_dir"):
        raise ValueError(
            f"Model '{call.spec.name}': model_type=\"{NAME}\" needs "
            "config(agent_dir=\"...\") — the shared workspace the agent works in."
        )
    workdir = Path(str(config["agent_dir"])).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    spath = state_path(call, workdir)
    state_dir = spath.parent

    phash = prompt_hash(rendered, config)
    state = load_state(spath)
    current = upstream_view(state, snapshot(workdir, exclude=state_dir))
    verdict = decide(state, rendered, phash, current, call)

    if verdict != "miss":
        assert state is not None
        replacements = state.get("cached_replacements") or {}
        deleted = state.get("deleted_files") or []
        replay(workdir, replacements, deleted)
        state["last_received_files"] = current
        save_state(spath, state)
        output = state.get("output", "")
        if call.spec.output_format == "json" and isinstance(output, str):
            from promptmap.executor.run_context import parse_json_output

            output = parse_json_output(output)
        return {
            "output": output,
            "logs": [],
            "time_run": 0.0,
            "cache": verdict,
            "changed_files": sorted(replacements),
            "deleted_files": sorted(deleted),
        }

    result = await builtin_kinds.run_agent(rendered + OUTCOME_CONTRACT, _workspace_keyed(call, current))
    after = snapshot(workdir, exclude=state_dir)
    replacements = {p: c for p, c in after.items() if current.get(p) != c}
    deleted = sorted(p for p in current if p not in after)
    raw_output = result.get("output", "")
    save_state(spath, {
        "prompt_hash": phash,
        "last_received_files": current,
        "cached_replacements": replacements,
        "deleted_files": deleted,
        "output": raw_output if isinstance(raw_output, str) else json.dumps(raw_output),
    })
    return {
        **result,
        "cache": "miss",
        "changed_files": sorted(replacements),
        "deleted_files": deleted,
    }


def _workspace_keyed(call: ModelCall, current: dict[str, Content]) -> ModelCall:
    """*call*, with compute keyed on the incoming workspace and always running.

    The ``agent`` kind memoises on the prompt alone, which would hand back an
    old answer without the agent touching the workspace — leaving nothing to
    diff.  Keying on the workspace too, and running the work anyway on a hit,
    keeps the before/after diff honest.
    """
    fingerprint = hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest()

    async def compute(key: str, compute: Callable[[], Any]) -> Any:
        ran = False

        def tracked():
            nonlocal ran
            ran = True
            return compute()

        value = await call.compute(f"{key}\x00workspace:{fingerprint}", compute=tracked)
        if not ran:
            value = compute()
            if hasattr(value, "__await__"):
                value = await value
        return value

    return dataclasses.replace(call, compute=compute)


#: The ``agent`` kind's record, renamed, with this exec_fn and the extra keys —
#: so it inherits ``agent``'s config keys, secrets and global-instruction setting.
KIND = register_model_kind(dataclasses.replace(
    builtin_kinds.AGENT,
    name=NAME,
    exec_fn=run_cached_agent,
    config_keys=builtin_kinds.AGENT.config_keys | {"jev_cache_threshold", "force_refresh", "state_dir"},
))
