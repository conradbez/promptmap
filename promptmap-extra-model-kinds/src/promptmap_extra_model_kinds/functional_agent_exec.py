"""
``local_functional_agent`` — a coding agent as a pure function: Dir in, Dir out.

The model receives a directory tree, lets the local ``agent`` kind work on a
throwaway copy of it (where it may build, run tests, install packages), and
returns the tree with its changes already applied.  The real repository is
never touched; a later node decides what to do with the result.

Input: ``promptfiles=["fix"]`` — an upstream model's :class:`~promptmap.files.Dir`.
If that model is another functional agent, its ``result`` is used;
``"fix.result"`` names it outright.  Where the tree comes from is the upstream
model's business (an ``execute_python`` node returning ``Dir("../app")``, an
earlier functional agent, …).  Where the agent works is nobody's: a scratch
copy that exists only to find out which files to create, edit or delete.

Output::

    {"output":  <the agent's final answer>,
     "result":  Dir — the input tree with this model's changes applied,
     "changes": Dir — only the files this model created or modified,
     "deleted": [paths this model removed],
     "cache":   "hit" | "replay" | "merged" | "miss",
     ...the agent kind's "logs" and "time_run" on a miss}

so the next functional agent takes ``promptfiles=["this_model"]``.

Reuse
-----
Per model, ``<state_dir>/<model>.json`` keeps the prompt, the tree it last
received (as file hashes) and its sparse change set; file bytes live in
``<state_dir>/blobs``.  With the same prompt:

``hit``     the input tree is unchanged: re-apply the change set.  No LLM call.
``replay``  upstream changed files this model did not touch: apply the change
            set to the new tree, then verify (one call).
``merged``  upstream also changed some of this model's files (at most
            ``merge_max_files``): one LLM call merges those files three-way,
            then verify (a second call).
``miss``    anything else, or a failed merge or verification: run the agent
            on the new tree, telling it its previous change, its previous
            answer, what changed upstream and why the old change was not reused.

Verification asks the classifier (``classify_call`` in client.py, else hosted
Jev when ``TYPESAFE_API_KEY`` is set) and falls back to a YES/NO LLM call.
It passes when the score reaches ``jev_cache_threshold``.

Config keys, beyond the ``agent`` kind's (``agent_dir`` is not used — the
agent always works in a scratch copy):

``jev_cache_threshold``  score needed to reuse over a changed input (0.85)
``merge_max_files``      most colliding files to merge rather than re-run (5)
``context_max_chars``    cap on each diff put into a prompt (20000)
``force_refresh``        always run the agent (false)
``state_dir``            default ``.promptmap/state``, relative to the
                         project root (the folder holding client.py)
"""

from __future__ import annotations

import asyncio
import dataclasses
import difflib
import hashlib
import inspect
import json
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from promptmap.executor import builtin_kinds
from promptmap.files import Dir, File, select_path, split_model_path
from promptmap.model_types import ModelCall, register_model_kind

log = logging.getLogger("promptmap")

NAME = "local_functional_agent"
DEFAULT_THRESHOLD = 0.85
DEFAULT_MERGE_MAX = 5
DEFAULT_CONTEXT_MAX = 20_000
DEFAULT_STATE_DIR = ".promptmap/state"

#: Appended to every agent task, so the change set stays sparse.
OUTCOME_CONTRACT = (
    "\n\nInspect the workspace and achieve the requested outcome. Modify or "
    "create only the files necessary to satisfy the outcome and leave all "
    "unrelated files untouched. The workspace is a scratch copy: build and run "
    "tests freely."
)

#: The Noul question put to the classifier when verifying a reuse.
VERIFY_QUESTION = (
    "Given upstream_diff, does node_change applied to the new workspace still "
    "completely satisfy outcome_goal without requiring any further code or file changes?"
)

#: Build and cache folders the agent may create while testing in its scratch copy.
_IGNORED_DIRS = frozenset({
    ".promptmap", "__pycache__", "node_modules", ".venv", ".pytest_cache", ".mypy_cache",
})
_IGNORED_PREFIXES = (".promptmap_out_",)


# ---------------------------------------------------------------------------
# Trees
# ---------------------------------------------------------------------------

def snapshot_dir(root: Path, keep: "set[str] | frozenset[str]" = frozenset(), name: str | None = None) -> Dir:
    """The scratch copy at *root* as a Dir.

    Build and cache folders the agent made while testing (``node_modules``,
    ``.venv``, ``__pycache__``, …) are left out, unless the input already had
    files under them (*keep* is the input's paths); ``.git`` always is.
    """
    root = root.resolve()
    kept_dirs = {PurePosixPath(p).parent.as_posix() for p in keep}
    kept_dirs = {str(d) for p in kept_dirs for d in [p, *map(str, PurePosixPath(p).parents)]}
    entries: dict[str, bytes] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        dirnames[:] = [
            d for d in dirnames
            if d != ".git" and not d.startswith(_IGNORED_PREFIXES)
            and (d not in _IGNORED_DIRS or (here / d).relative_to(root).as_posix() in kept_dirs)
        ]
        for f in filenames:
            path = here / f
            if path.is_file() and not path.is_symlink():
                entries[path.relative_to(root).as_posix()] = path.read_bytes()
    return Dir(entries, name=name or root.name or "tree")


def apply_changes(base: Dir, changes: dict[str, bytes], deleted: list[str]) -> Dir:
    """*base* with *changes* written over it and *deleted* removed."""
    entries: dict[str, File | bytes] = {p: f for p, f in base.entries.items() if p not in set(deleted)}
    entries.update(changes)
    return Dir(entries, name=base.name)


def changed_paths(before: dict[str, str], after: dict[str, str]) -> set[str]:
    """Paths whose hash differs between two ``{path: sha256}`` maps."""
    return {p for p in before.keys() | after.keys() if before.get(p) != after.get(p)}


def hashes(tree: Dir) -> dict[str, str]:
    return {p: f.sha256 for p, f in tree.entries.items()}


def unified_diff(paths: list[str], old: Callable[[str], bytes | None], new: Callable[[str], bytes | None]) -> str:
    chunks: list[str] = []
    for path in paths:
        a, b = old(path), new(path)
        ta, tb = _text(a), _text(b)
        if (a is not None and ta is None) or (b is not None and tb is None):
            chunks.append(f"Binary file {path} changed\n")
            continue
        lines = list(difflib.unified_diff(
            (ta or "").splitlines(keepends=True), (tb or "").splitlines(keepends=True),
            fromfile=f"a/{path}" if a is not None else "/dev/null",
            tofile=f"b/{path}" if b is not None else "/dev/null",
        ))
        chunks += [line if line.endswith("\n") else line + "\n" for line in lines]
    return "".join(chunks)


def _text(data: bytes | None) -> str | None:
    if data is None:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… [{len(text) - limit} more characters cut]\n"


# ---------------------------------------------------------------------------
# State: a JSON record per model plus a content-addressed blob folder
# ---------------------------------------------------------------------------

class StateStore:
    def __init__(self, path: Path):
        self.path = path
        self.blobs = path.parent / "blobs"

    def load(self) -> dict | None:
        try:
            return json.loads(self.path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def save(self, state: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=1, sort_keys=True))
        os.replace(tmp, self.path)

    def put(self, data: bytes) -> str:
        sha = hashlib.sha256(data).hexdigest()
        target = self.blobs / sha
        if not target.exists():
            self.blobs.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(f".tmp{os.getpid()}")
            tmp.write_bytes(data)
            os.replace(tmp, target)
        return sha

    def put_tree(self, tree: Dir) -> None:
        for f in tree.entries.values():
            if not (self.blobs / f.sha256).exists():
                self.put(f.read_bytes())

    def get(self, sha: str | None) -> bytes | None:
        if sha is None:
            return None
        try:
            return (self.blobs / sha).read_bytes()
        except FileNotFoundError:
            return None


def state_path(call: ModelCall) -> Path:
    configured = Path(str(call.spec.config.get("state_dir") or DEFAULT_STATE_DIR)).expanduser()
    if not configured.is_absolute():
        configured = (_project_root(call) or Path.cwd()) / configured
    return configured / f"{call.spec.name}.json"


def _project_root(call: ModelCall) -> Path | None:
    path = call.spec.path
    if not path.is_file():
        return None
    for parent in path.resolve().parents:
        if (parent / "client.py").is_file():
            return parent
    return path.resolve().parent


def prompt_hash(rendered: str, config: dict) -> str:
    salt = json.dumps({k: str(config.get(k, "")) for k in ("agent_backend", "agent_model")}, sort_keys=True)
    return hashlib.sha256((rendered + "\x00" + salt).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

def resolve_base(call: ModelCall) -> Dir:
    """The input tree: the first promptfile holding a Dir."""
    for name in call.spec.promptfiles_used:
        found = split_model_path(name, call.outputs)
        if found is None:
            continue
        model, path = found
        value = select_path(call.outputs[model], path, name)
        if isinstance(value, dict) and isinstance(value.get("result"), Dir):
            value = value["result"]
        if isinstance(value, Dir):
            return value
    raise ValueError(
        f"Model '{call.spec.name}': model_type=\"{NAME}\" needs an input tree — "
        "config(promptfiles=[\"<upstream model>\"]) naming a model whose output is a Dir "
        "(e.g. an execute_python model ending in output = Dir(\"path\"))."
    )


# ---------------------------------------------------------------------------
# LLM helpers: text-only calls, verification, merging
# ---------------------------------------------------------------------------

async def text_llm(call: ModelCall, prompt: str) -> str:
    """Send *prompt* to the run's LLM backend *without* the promptfiles.

    ``call.llm`` attaches the model's promptfiles — here, the whole input
    tree — to every call.  A merge or a verification needs only the text, so
    the backend is called directly, still through the cache.
    """
    backend = getattr(getattr(getattr(call.llm, "func", None), "__self__", None), "llm_call", None)
    if backend is None:
        return await call.llm(prompt)

    def invoke():
        params = inspect.signature(backend).parameters
        kwargs: dict = {}
        if "config" in params:
            kwargs["config"] = call.spec.config
        if "secrets" in params:
            kwargs["secrets"] = call.secrets
        return backend(prompt, **kwargs)

    async def run():
        out = await asyncio.to_thread(invoke)
        return await out if inspect.isawaitable(out) else out

    return await call.compute(f"{prompt}\x00{NAME}:text_llm", compute=run)


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


async def verify(call: ModelCall, goal: str, owned: list[str], upstream_diff: str, change_diff: str) -> float:
    """P(the reused change still satisfies *goal*) — classifier, else LLM YES/NO."""
    limit = _int(call.spec.config.get("context_max_chars"), DEFAULT_CONTEXT_MAX)
    state = (
        f"outcome_goal:\n{goal}\n\n"
        f"node_owned_files:\n{json.dumps(sorted(owned))}\n\n"
        f"upstream_diff:\n{_clip(upstream_diff, limit)}\n\n"
        f"node_change:\n{_clip(change_diff, limit)}"
    )
    classify = resolve_classifier(call)
    if classify is not None:
        return float(await asyncio.to_thread(classify, state, VERIFY_QUESTION))
    reply = await text_llm(call, f"{state}\n\nQuestion: {VERIFY_QUESTION}\nAnswer with exactly YES or NO.")
    return 1.0 if str(reply).strip().upper().startswith("YES") else 0.0


_FILE_BLOCK = re.compile(r"<<<FILE (?P<path>[^>\n]+)>>>\n(?P<body>.*?)<<<END>>>", re.DOTALL)
_DELETE_LINE = re.compile(r"<<<DELETE (?P<path>[^>\n]+)>>>")

MERGE_INSTRUCTIONS = """\
You are merging two independent edits to the same files.

GOAL of "your" edit:
{goal}

For each file below you get ORIGINAL (what both edits started from), YOURS
(the earlier edit made for the goal) and UPSTREAM (a newer edit made by
someone else). Produce the merged file: keep every UPSTREAM change and
re-apply what YOURS changed relative to ORIGINAL so the GOAL still holds.

{files}
Reply with one block per file and nothing else:

<<<FILE path/to/file>>>
<full merged file content>
<<<END>>>

or, if the file should not exist after merging, the single line:

<<<DELETE path/to/file>>>
"""


def _merge_section(path: str, original: bytes | None, yours: bytes | None, upstream: bytes | None) -> str:
    def show(label: str, data: bytes | None, missing: str) -> str:
        return f"--- {label} ---\n{missing if data is None else data.decode('utf-8')}\n"

    return (
        f"=== {path} ===\n"
        + show("ORIGINAL", original, "<file did not exist>")
        + show("YOURS", yours, "<you deleted this file>")
        + show("UPSTREAM", upstream, "<upstream deleted this file>")
        + "\n"
    )


async def merge(call: ModelCall, goal: str, sides: dict[str, tuple]) -> dict[str, bytes | None] | None:
    """Three-way merge *sides* (path → original, yours, upstream) with one LLM call.

    Returns path → merged bytes (None = delete), or None if any side is
    binary or the reply does not cover every file.
    """
    if any(d is not None and _text(d) is None for trio in sides.values() for d in trio):
        return None
    files = "".join(_merge_section(p, *sides[p]) for p in sorted(sides))
    reply = str(await text_llm(call, MERGE_INSTRUCTIONS.format(goal=goal, files=files)))
    merged: dict[str, bytes | None] = {}
    for m in _FILE_BLOCK.finditer(reply):
        merged[m.group("path").strip()] = m.group("body").encode("utf-8")
    for m in _DELETE_LINE.finditer(reply):
        merged[m.group("path").strip()] = None
    if set(merged) != set(sides):
        log.info("%s '%s': merge reply did not cover %s", NAME, call.spec.name, sorted(set(sides) - set(merged)))
        return None
    return merged


# ---------------------------------------------------------------------------
# exec_fn
# ---------------------------------------------------------------------------

async def run_functional_agent(rendered: str, call: ModelCall) -> dict:
    config = call.spec.config
    base = resolve_base(call)
    store = StateStore(state_path(call))
    store.put_tree(base)
    state = store.load()
    phash = prompt_hash(rendered, config)
    base_hashes = hashes(base)

    if _truthy(config.get("force_refresh")):
        reason = "a refresh was forced"
    elif state is None:
        reason = None  # first run: nothing to reuse, nothing to explain
    elif state.get("prompt_hash") != phash:
        reason = "the task itself changed"
    else:
        reused = await _try_reuse(rendered, call, store, state, base, base_hashes)
        if isinstance(reused, dict):
            return reused
        reason = reused

    return await _run_agent(rendered, call, store, state, base, base_hashes, phash, reason)


async def _try_reuse(rendered, call, store, state, base, base_hashes) -> dict | str:
    """The reused result, or the reason it could not be reused."""
    prev = state.get("base") or {}
    changes: dict[str, str] = state.get("changes") or {}
    deleted: list[str] = state.get("deleted") or []
    upstream = changed_paths(prev, base_hashes)

    def finish(verdict: str, new_changes: dict[str, bytes], new_deleted: list[str]) -> dict:
        return _finish(call, store, state, base, base_hashes, verdict, new_changes, new_deleted)

    current = {p: store.get(s) for p, s in changes.items()}
    if any(v is None for v in current.values()):
        return "its saved change set is incomplete"
    if not upstream:
        return finish("hit", current, deleted)

    owned = set(changes) | set(deleted)
    collisions = sorted(upstream & owned)
    new_changes = {p: b for p, b in current.items() if p not in collisions}
    new_deleted = [p for p in deleted if p not in collisions]
    verdict = "replay"

    if collisions:
        limit = _int(call.spec.config.get("merge_max_files"), DEFAULT_MERGE_MAX)
        if len(collisions) > limit:
            return f"upstream changed {len(collisions)} of the files it had changed (merge limit {limit})"
        sides = {
            p: (store.get(prev.get(p)), current.get(p), store.get(base_hashes.get(p)))
            for p in collisions
        }
        merged = await merge(call, rendered, sides)
        if merged is None:
            return f"its changes to {', '.join(collisions)} could not be merged with upstream's"
        for path, data in merged.items():
            if data is None:
                if path in base_hashes:
                    new_deleted.append(path)
            elif data != store.get(base_hashes.get(path)):
                new_changes[path] = data
        verdict = "merged"

    upstream_diff = unified_diff(sorted(upstream), lambda p: store.get(prev.get(p)), lambda p: store.get(base_hashes.get(p)))
    touched = sorted(set(new_changes) | set(new_deleted))
    change_diff = unified_diff(
        touched,
        lambda p: store.get(base_hashes.get(p)),
        lambda p: None if p in new_deleted else new_changes.get(p),
    )
    threshold = _threshold(call.spec.config.get("jev_cache_threshold"))
    try:
        score = await verify(call, rendered, touched, upstream_diff, change_diff)
    except Exception:
        log.warning("%s '%s': verification failed; running the agent", NAME, call.spec.name, exc_info=True)
        return "verifying the reused change failed"
    if score < threshold:
        return f"the reused change was judged not to hold after upstream's edits (score {score:.2f} < {threshold})"
    return finish(verdict, new_changes, sorted(set(new_deleted)))


def _finish(call, store, state, base, base_hashes, verdict, changes, deleted) -> dict:
    deleted = sorted(p for p in deleted if p in base_hashes)
    result = apply_changes(base, changes, deleted)
    store.save({
        **state,
        "base": base_hashes,
        "changes": {p: store.put(b) for p, b in changes.items()},
        "deleted": deleted,
    })
    return _shape(call, state.get("output", ""), result, changes, deleted, verdict, base.name)


def _shape(call, output, result: Dir, changes: dict[str, bytes], deleted, verdict, name) -> dict:
    if call.spec.output_format == "json" and isinstance(output, str):
        from promptmap.executor.run_context import parse_json_output

        output = parse_json_output(output)
    return {
        "output": output,
        "result": result,
        "changes": Dir(dict(changes), name=f"{name}-changes"),
        "deleted": list(deleted),
        "cache": verdict,
    }


async def _run_agent(rendered, call, store, state, base, base_hashes, phash, reason) -> dict:
    task = rendered + OUTCOME_CONTRACT + _previous_attempt(call, store, state, base_hashes, reason)

    def work() -> dict:
        scratch = Path(tempfile.mkdtemp(prefix="promptmap-functional-"))
        try:
            root = base.save(scratch)
            spec = call.spec.derive(config={**call.spec.config, "agent_dir": str(root)})
            raw = builtin_kinds._exec_agent(task, dataclasses.replace(call, spec=spec))
            result = json.loads(raw) if isinstance(raw, str) else raw
            result["result"] = snapshot_dir(root, keep=set(base.entries), name=base.name)
            return result
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    # Keyed on the task (which carries the previous attempt) and the input
    # tree, so an identical situation is memoised by promptmap's own cache.
    value = await call.compute(f"{task}\x00base:{base.sha256}", compute=lambda: asyncio.to_thread(work))
    value = json.loads(value) if isinstance(value, str) else dict(value)
    result: Dir = value["result"]
    after = hashes(result)
    changes = {p: result.entries[p].read_bytes() for p, s in after.items() if base_hashes.get(p) != s}
    deleted = sorted(p for p in base_hashes if p not in after)
    raw_output = value.get("output", "")
    store.save({
        "prompt_hash": phash,
        "base": base_hashes,
        "changes": {p: store.put(b) for p, b in changes.items()},
        "deleted": deleted,
        "output": raw_output if isinstance(raw_output, str) else json.dumps(raw_output),
    })
    shaped = _shape(call, raw_output, result, changes, deleted, "miss", base.name)
    return {**{k: v for k, v in value.items() if k not in shaped}, **shaped}


def _previous_attempt(call, store, state, base_hashes, reason) -> str:
    """What the agent should know about its last run, for an informed re-run."""
    if not state:
        return ""
    limit = _int(call.spec.config.get("context_max_chars"), DEFAULT_CONTEXT_MAX)
    prev = state.get("base") or {}
    changes = state.get("changes") or {}
    deleted = state.get("deleted") or []
    own_diff = unified_diff(
        sorted(set(changes) | set(deleted)),
        lambda p: store.get(prev.get(p)),
        lambda p: None if p in deleted else store.get(changes.get(p)),
    )
    upstream = sorted(changed_paths(prev, base_hashes))
    upstream_diff = unified_diff(upstream, lambda p: store.get(prev.get(p)), lambda p: store.get(base_hashes.get(p)))
    parts = [
        "\n\n## Previous attempt",
        f"You worked on this before. It could not simply be reused because {reason or 'it was out of date'}.",
        "Start from your previous change: re-apply what still holds, adapt it to the "
        "current workspace, and fix only what no longer works.",
    ]
    if state.get("output"):
        parts.append(f"\n### Your previous answer\n{_clip(str(state['output']), limit)}")
    parts.append(
        "\n### Your previous change (against the workspace you had then)\n"
        + (f"```diff\n{_clip(own_diff, limit)}```" if own_diff else "(no file changes)")
    )
    if upstream_diff:
        parts.append(
            "\n### What changed in the workspace since then (not by you)\n"
            f"```diff\n{_clip(upstream_diff, limit)}```"
        )
    return "\n".join(parts) + "\n"


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _int(value: Any, default: int) -> int:
    return default if value in (None, "") else int(value)


def _threshold(value: Any) -> float:
    if value in (None, ""):
        return DEFAULT_THRESHOLD
    threshold = float(value)
    if not 0 <= threshold <= 1:
        raise ValueError(f"jev_cache_threshold must be between 0 and 1; got {threshold}.")
    return threshold


#: The ``agent`` kind's record, renamed, with this exec_fn and its own keys.
KIND = register_model_kind(dataclasses.replace(
    builtin_kinds.AGENT,
    name=NAME,
    exec_fn=run_functional_agent,
    config_keys=(builtin_kinds.AGENT.config_keys - {"agent_dir"}) | {
        "jev_cache_threshold", "merge_max_files", "context_max_chars",
        "force_refresh", "state_dir",
    },
))
