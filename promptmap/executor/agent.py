"""
``agent`` — a coding agent as a pure function: Dir in, Dir out.

The harness is pi (https://pi.dev, ``npm i -g @mariozechner/pi-coding-agent``),
run headless (``pi --print --mode json``).  The model receives a directory
tree, lets pi work on a throwaway copy of it (where it may build, run tests, install packages), and
returns the tree with its changes already applied.  The real repository is
never touched; a later node decides what to do with the result.

Input (optional): ``promptfiles="fix"`` — an upstream model's :class:`~promptmap.files.Dir`.
Without one the agent starts from an empty folder named after the model, so
it creates the tree rather than editing one.
If that model is another ``agent``, its ``result`` is used;
``"fix.result"`` names it outright.  Where the tree comes from is the upstream
model's business (a custom kind returning ``Dir("../app")``, an earlier
``agent``, …).  Where the agent works is nobody's: a scratch
copy that exists only to find out which files to create, edit or delete.

Output::

    {"output":  <the agent's final answer>,
     "result":  Dir — the input tree with this model's changes applied,
     "changes": {path: text} — only the files this model created or modified,
     "deleted": [paths this model removed],
     "base":    {"name", "hashes": {path: sha256}, "originals": {path: text}}
                — the input tree it worked from, as hashes plus the original
                text of only the files this model changed or deleted,
     "sum_of_changes": {"ast_summary": [...], "unified_diff": "..."} — result vs input,
     "cache":   "hit" | "replay" | "merged" | "miss",
     "merged_by": {path: "git" | "llm"} — when files were merged,
     ...the agent kind's "logs" and "time_run" on a miss}

so the next ``agent`` takes ``promptfiles="this_model"``.

``result`` is the only file output, so the run's ``outputs/<model>/`` folder
holds exactly the complete patched tree.  Everything else is plain data: text
files as strings, binary files (images, video, PDFs, …) only as
``{"sha256": ...}``.  The next run reads a changed binary's bytes from the
previous ``result``; an *original* binary is known only by its hash, which is
enough to tell whether upstream changed it — a binary file both sides changed
is never merged, so the agent re-runs.

Reuse
-----
This kind keeps no state of its own.  Its output carries what the next run
needs — the input tree it worked from (``base``), its sparse change set
(``changes``, ``deleted``) and their ``sum_of_changes`` — and promptmap hands
that back as ``call.previous``: the model's last successful result from a run
of the same template (promptmap >= 0.5.17).  Editing the ``.prompt`` file
therefore starts afresh; deleting the run database does too.  With the same
rendered prompt:

``hit``     upstream changed nothing that means anything — the tree is the
            same, or only comments/formatting changed in files this model
            did not touch: re-apply the change set.  No LLM call.
``replay``  upstream changed the meaning of files this model did not touch:
            apply the change set to the new tree, then verify (one call).
``merged``  upstream also edited some of this model's files (at most
            ``merge_max_files``): each is merged three-way with ``git merge-file``,
            and whatever conflicts goes to one LLM merge call,
            then verify.  When upstream's edits there were
            only comments/formatting and the merge means exactly what this
            model's version meant, the verify call is skipped.
``miss``    anything else, or a failed merge or verification: run the agent
            on the new tree, telling it its previous change, its previous
            answer, what changed upstream and why the old change was not reused.

"Meaning" is judged by :mod:`._ast_diff`: files are parsed with
tree-sitter-language-pack and compared with comments (bar directives such as
``# type: ignore``) and formatting stripped; changes are summarised per symbol
(``ADDED_SYMBOL``, ``REMOVED_SYMBOL``, ``SIGNATURE_CHANGED``, ``BODY_CHANGED``).
Files without a grammar, or that do not parse, are compared byte for byte.

The verifier is shown three things: ``upstream_input_ast_diff`` (what changed
in the input since last time, as that symbol summary plus a unified diff),
``previous_run_sum_of_changes`` (everything this model changed on its last
run, against the input it had then) and ``replayed_changes`` (what reuse would
now apply to the new input).  Verification asks the classifier (``classify_call`` in client.py, else hosted
Jev when ``TYPESAFE_API_KEY`` is set) and falls back to a YES/NO LLM call.
It passes when the score reaches ``jev_cache_threshold``.

Config keys (the agent always works in a scratch copy, never in place):

``agent_model``          pi model, e.g. ``anthropic/claude-sonnet-4-5``; defaults
                         to client.py's ``agent_model``, then pi's own default
``agent_step_limit``     max agent turns, 0 for none (default 0)
``agent_cost_limit``     max spend in dollars, 0 for none (default 3)

``jev_cache_threshold``  score needed to reuse over a changed input (0.85)
``merge_max_files``      most colliding files to merge rather than re-run (5)
``context_max_chars``    cap on each diff put into a prompt (20000)
``force_refresh``        always run the agent (false)
"""

from __future__ import annotations

import asyncio
import dataclasses
from dataclasses import dataclass
import difflib
import hashlib
import inspect
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable

import time

from promptmap.files import Dir, File, FileOutputError, select_path, split_model_path
from promptmap.model_types import ModelCall, ModelKind, register_model_kind, report_progress

from . import _ast_diff

log = logging.getLogger("promptmap")

NAME = "agent"
DEFAULT_THRESHOLD = 0.85
DEFAULT_MERGE_MAX = 5
DEFAULT_CONTEXT_MAX = 20_000

#: Appended to every agent task, so the change set stays sparse.
OUTCOME_CONTRACT = (
    "\n\nInspect the workspace and achieve the requested outcome. Modify or "
    "create only the files necessary to satisfy the outcome and leave all "
    "unrelated files untouched. The workspace is a scratch copy: build and run "
    "tests freely."
)

#: The Noul question put to the classifier when verifying a reuse.
VERIFY_QUESTION = (
    "On the previous run, this node achieved outcome_goal by applying "
    "previous_run_sum_of_changes to its input folder. Since then, the input folder "
    "has changed by upstream_input_ast_diff. Does re-applying the node's changes "
    "(replayed_changes) on top of the new input still completely satisfy "
    "outcome_goal, without the agent having to produce a new diff?"
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


def sum_of_changes(
    paths: "Iterable[str]", old: Callable[[str], bytes | None], new: Callable[[str], bytes | None]
) -> dict:
    """``{"ast_summary": [...], "unified_diff": "..."}`` for the files in *paths*."""
    paths = sorted(p for p in paths if old(p) != new(p))
    return {
        "ast_summary": _ast_diff.summary(_ast_diff.diff_trees(paths, old, new)),
        "unified_diff": unified_diff(paths, old, new),
    }


def _render_sum(changes: dict, limit: int) -> str:
    summary = "\n".join(changes["ast_summary"]) or "(none)"
    diff = changes["unified_diff"]
    return f"{summary}\n\n```diff\n{_clip(diff, limit)}```" if diff else summary


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
# The last run — read back from promptmap (``call.previous``), not stored here
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Last:
    """What this model returned on its last successful run of this template."""

    prompt: str
    answer: Any
    #: The tree it produced last time.
    result: Dir
    #: The input it worked from: every path's hash, and the original bytes of
    #: the files it changed or deleted (the rest are as in ``result``).
    base_hashes: dict[str, str]
    originals: dict[str, bytes]
    changes: dict[str, bytes]
    deleted: tuple[str, ...]
    sum_of_changes: dict | None = None

    def was(self, path: str) -> bytes | None:
        """*path* as this model received it last time."""
        if path in self.originals:
            return self.originals[path]
        if path in self.owned or path not in self.base_hashes:
            return None
        return _read(self.result, path)

    def ours(self, path: str) -> bytes | None:
        """*path* as this model left it last time."""
        if path in self.deleted:
            return None
        if path in self.changes:
            return self.changes[path]
        return self.was(path)

    @property
    def owned(self) -> set[str]:
        return set(self.changes) | set(self.deleted)


def encode_bytes(data: bytes) -> str | dict:
    """File content as JSON data: text as a string, anything else as its hash."""
    text = _text(data)
    return text if text is not None else {"sha256": hashlib.sha256(data).hexdigest()}


#: Stands in for a binary file whose bytes were not kept.  Not valid UTF-8,
#: so it is treated as binary everywhere (never merged, diffed as "Binary
#: file changed"); equal only to a stand-in for the same content.
_OPAQUE_PREFIX = b"\xff\xfepromptmap-binary sha256:"


def opaque(sha256: str) -> bytes:
    return _OPAQUE_PREFIX + sha256.encode("ascii")


def decode_bytes(value: str | dict, source: Dir | None = None, path: str | None = None) -> bytes:
    """Bytes back from :func:`encode_bytes`.

    A binary is read from *source* at *path* when it holds that content,
    else stands in as :func:`opaque`.
    """
    if not isinstance(value, dict):
        return value.encode("utf-8")
    sha = value["sha256"]
    file = source.entries.get(path) if source is not None and path is not None else None
    if file is not None and file.sha256 == sha:
        return file.read_bytes()
    return opaque(sha)


def last_run(call: ModelCall) -> Last | None:
    """The previous result, if it is one this kind produced (and can reuse)."""
    previous = getattr(call, "previous", None)  # promptmap >= 0.5.17
    out = getattr(previous, "output", None)
    if not isinstance(out, dict):
        return None
    base, changes = out.get("base"), out.get("changes")
    if not (isinstance(out.get("result"), Dir) and isinstance(changes, dict)
            and isinstance(base, dict) and isinstance(base.get("hashes"), dict)):
        return None
    summary = out.get("sum_of_changes")
    return Last(
        prompt=previous.prompt,
        answer=out.get("output", ""),
        result=out["result"],
        base_hashes=dict(base["hashes"]),
        originals={p: decode_bytes(v) for p, v in (base.get("originals") or {}).items()},
        changes={p: decode_bytes(v, out["result"], p) for p, v in changes.items()},
        deleted=tuple(out.get("deleted") or ()),
        sum_of_changes=summary if isinstance(summary, dict) and "ast_summary" in summary else None,
    )


def _read(tree: Dir, path: str) -> bytes | None:
    file = tree.entries.get(path)
    return None if file is None else file.read_bytes()


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

def resolve_base(call: ModelCall) -> Dir:
    """The input tree: the first promptfile holding a Dir, else an empty one."""
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
    # No input tree: the agent starts from an empty folder and creates one.
    return Dir({}, name=call.spec.name)


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


async def verify(call: ModelCall, goal: str, upstream: dict, previous: dict, replayed: dict) -> float:
    """P(the replayed change still satisfies *goal*) — classifier, else LLM YES/NO.

    Each of *upstream*, *previous* and *replayed* is a :func:`sum_of_changes`.
    """
    limit = _int(call.spec.config.get("context_max_chars"), DEFAULT_CONTEXT_MAX)
    state = (
        f"outcome_goal:\n{goal}\n\n"
        f"upstream_input_ast_diff:\n{_render_sum(upstream, limit)}\n\n"
        f"previous_run_sum_of_changes:\n{_render_sum(previous, limit)}\n\n"
        f"replayed_changes:\n{_render_sum(replayed, limit)}"
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


def git_merge(original: bytes | None, yours: bytes | None, upstream: bytes | None) -> bytes | None:
    """A clean textual three-way merge by ``git merge-file``, else None.

    None on a conflict, when either side deleted the file, for binary
    content, or when git is not installed — the LLM merge takes those.
    """
    git = shutil.which("git")
    if git is None or yours is None or upstream is None:
        return None
    if any(d is not None and _text(d) is None for d in (original, yours, upstream)):
        return None
    with tempfile.TemporaryDirectory(prefix="promptmap-merge-") as tmp:
        paths = []
        for name, data in (("yours", yours), ("original", original or b""), ("upstream", upstream)):
            path = Path(tmp) / name
            path.write_bytes(data)
            paths.append(str(path))
        try:
            proc = subprocess.run([git, "merge-file", "-p", *paths], capture_output=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            return None
    return proc.stdout if proc.returncode == 0 else None


async def merge(
    call: ModelCall, goal: str, sides: dict[str, tuple]
) -> tuple[dict[str, bytes | None], dict[str, str]] | None:
    """Three-way merge *sides* (path → original, yours, upstream).

    Each file is tried with ``git merge-file`` first; whatever does not merge
    cleanly goes to the LLM in one call.  Returns (path → merged bytes, None
    meaning delete; path → "git" or "llm"), or None if the LLM merge failed.
    """
    merged: dict[str, bytes | None] = {}
    for path, (original, yours, upstream) in sides.items():
        clean = git_merge(original, yours, upstream)
        if clean is not None:
            merged[path] = clean
    by = {path: "git" for path in merged}
    rest = {path: trio for path, trio in sides.items() if path not in merged}
    if rest:
        llm_merged = await llm_merge(call, goal, rest)
        if llm_merged is None:
            return None
        merged.update(llm_merged)
        by.update({path: "llm" for path in rest})
    return merged, by


async def llm_merge(call: ModelCall, goal: str, sides: dict[str, tuple]) -> dict[str, bytes | None] | None:
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

async def run_agent(rendered: str, call: ModelCall) -> dict:
    base = resolve_base(call)
    last = last_run(call)

    if _truthy(call.spec.config.get("force_refresh")):
        reason = "a refresh was forced"
    elif last is None:
        reason = None  # first run: nothing to reuse, nothing to explain
    elif last.prompt != rendered:
        reason = "the task itself changed"
    else:
        try:
            reused = await _try_reuse(rendered, call, last, base)
        except FileOutputError:
            reused = "the files of its previous result are no longer in the blob store"
        if isinstance(reused, dict):
            return reused
        reason = reused

    try:
        return await _run_agent(rendered, call, last, base, reason)
    except FileOutputError:
        # The previous result's files are gone: run without that context.
        return await _run_agent(rendered, call, None, base, reason)


async def _try_reuse(rendered: str, call: ModelCall, last: Last, base: Dir) -> dict | str:
    """The reused result, or the reason it could not be reused.

    Upstream changes are judged semantically (:mod:`._ast_diff`): edits that
    only touch comments or formatting never need a verdict.  But any upstream
    edit to a file this model also changed is merged, never overwritten.
    """
    base_hashes = hashes(base)
    upstream = changed_paths(last.base_hashes, base_hashes)
    current = dict(last.changes)
    deleted = list(last.deleted)

    def finish(verdict: str, new_changes: dict[str, bytes], new_deleted: list[str], **extra) -> dict:
        return _finish(call, last, base, verdict, new_changes, new_deleted, **extra)

    if not upstream:
        return finish("hit", current, deleted)

    def now(p):
        return _read(base, p)

    upstream_changes = _ast_diff.diff_trees(upstream, last.was, now)
    semantic = {c.path for c in upstream_changes if c.semantic}
    collisions = sorted(upstream & last.owned)
    if not semantic and not collisions:
        return finish("hit", current, deleted)  # comments/formatting elsewhere only

    new_changes = {p: b for p, b in current.items() if p not in collisions}
    new_deleted = [p for p in deleted if p not in collisions]
    verdict, merged_by = "replay", {}
    merge_kept_meaning = False

    if collisions:
        limit = _int(call.spec.config.get("merge_max_files"), DEFAULT_MERGE_MAX)
        if len(collisions) > limit:
            return f"upstream changed {len(collisions)} of the files it had changed (merge limit {limit})"
        sides = {p: (last.was(p), last.ours(p), now(p)) for p in collisions}
        result = await merge(call, rendered, sides)
        if result is None:
            return f"its changes to {', '.join(collisions)} could not be merged with upstream's"
        merged, merged_by = result
        for path, data in merged.items():
            if data is None:
                if path in base_hashes:
                    new_deleted.append(path)
            elif data != now(path):
                new_changes[path] = data
        verdict = "merged"
        # Upstream only reformatted or re-commented our files, and the merge
        # means exactly what our version meant: nothing left to judge.
        merge_kept_meaning = not (semantic & set(collisions)) and all(
            _same_meaning(p, merged[p], last.ours(p)) for p in collisions
        )

    extra = {"merged_by": merged_by} if merged_by else {}
    if not semantic and merge_kept_meaning:
        return finish(verdict, new_changes, sorted(set(new_deleted)), **extra)

    replayed = sum_of_changes(
        set(new_changes) | set(new_deleted), now, lambda p: None if p in new_deleted else new_changes.get(p)
    )
    threshold = _threshold(call.spec.config.get("jev_cache_threshold"))
    try:
        score = await verify(
            call, rendered,
            upstream={"ast_summary": _ast_diff.summary(upstream_changes),
                      "unified_diff": unified_diff(sorted(semantic), last.was, now)},
            previous=_previous_sum(last),
            replayed=replayed,
        )
    except Exception:
        log.warning("%s '%s': verification failed; running the agent", NAME, call.spec.name, exc_info=True)
        return "verifying the reused change failed"
    if score < threshold:
        return f"the reused change was judged not to hold after upstream's edits (score {score:.2f} < {threshold})"
    return finish(verdict, new_changes, sorted(set(new_deleted)), **extra)


def _same_meaning(path: str, a: bytes | None, b: bytes | None) -> bool:
    if a is None or b is None:
        return a is b
    return _ast_diff.analyse(path, a).fingerprint == _ast_diff.analyse(path, b).fingerprint


def _previous_sum(last: Last) -> dict:
    """What this model changed on its last run, against the input it had then."""
    if last.sum_of_changes is not None:
        return last.sum_of_changes
    return sum_of_changes(last.owned, last.was, last.ours)


def _finish(call, last: Last, base: Dir, verdict: str, changes, deleted, **extra) -> dict:
    deleted = sorted(p for p in deleted if p in base.entries)
    result = apply_changes(base, changes, deleted)
    return {**_shape(call, last.answer, base, result, changes, deleted, verdict), **extra}


def _shape(call, output, base: Dir, result: Dir, changes: dict[str, bytes], deleted, verdict) -> dict:
    """The output — and, read back as ``call.previous`` next run, the state."""
    if call.spec.output_format == "json" and isinstance(output, str):
        from promptmap.executor.run_context import parse_json_output

        output = parse_json_output(output)
    owned = set(changes) | set(deleted)
    return {
        "output": output,
        "result": result,
        "changes": {p: encode_bytes(b) for p, b in sorted(changes.items())},
        "deleted": list(deleted),
        "base": {
            "name": base.name,
            "hashes": hashes(base),
            "originals": {p: encode_bytes(f.read_bytes()) for p, f in base.entries.items() if p in owned},
        },
        "sum_of_changes": sum_of_changes(
            set(changes) | set(deleted),
            lambda p: _read(base, p),
            lambda p: None if p in deleted else changes.get(p),
        ),
        "cache": verdict,
    }


async def _run_agent(rendered: str, call: ModelCall, last: Last | None, base: Dir, reason: str | None) -> dict:
    task = rendered + OUTCOME_CONTRACT + _previous_attempt(call, last, base, reason)

    def work() -> dict:
        scratch = Path(tempfile.mkdtemp(prefix="promptmap-agent-"))
        try:
            root = base.save(scratch)
            spec = call.spec.derive(config={**call.spec.config, "agent_dir": str(root)})
            raw = _exec_agent(task, dataclasses.replace(call, spec=spec))
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
    base_hashes = hashes(base)
    after = hashes(result)
    changes = {p: result.entries[p].read_bytes() for p, s in after.items() if base_hashes.get(p) != s}
    deleted = sorted(p for p in base_hashes if p not in after)
    shaped = _shape(call, value.get("output", ""), base, result, changes, deleted, "miss")
    return {**{k: v for k, v in value.items() if k not in shaped}, **shaped}


def _previous_attempt(call: ModelCall, last: Last | None, base: Dir, reason: str | None) -> str:
    """What the agent should know about its last run, for an informed re-run."""
    if last is None:
        return ""
    limit = _int(call.spec.config.get("context_max_chars"), DEFAULT_CONTEXT_MAX)
    upstream = sum_of_changes(
        changed_paths(last.base_hashes, hashes(base)), last.was, lambda p: _read(base, p)
    )
    parts = [
        "\n\n## Previous attempt",
        f"You worked on this before. It could not simply be reused because {reason or 'it was out of date'}.",
        "Start from your previous change: re-apply what still holds, adapt it to the "
        "current workspace, and fix only what no longer works.",
    ]
    if last.answer:
        answer = last.answer if isinstance(last.answer, str) else json.dumps(last.answer)
        parts.append(f"\n### Your previous answer\n{_clip(answer, limit)}")
    previous = _previous_sum(last)
    parts.append(
        "\n### Your previous change (against the workspace you had then)\n"
        + (_render_sum(previous, limit) if previous["ast_summary"] else "(no file changes)")
    )
    if upstream["ast_summary"]:
        parts.append(
            "\n### What changed in the workspace since then (not by you)\n"
            + _render_sum(upstream, limit)
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


# ---------------------------------------------------------------------------
# The harness: pi, headless
# ---------------------------------------------------------------------------

def _exec_agent(task: str, call: ModelCall) -> dict:
    """Run pi in ``config["agent_dir"]`` with *task*; ``{"output", "logs", "time_run"}``.

    pi's final assistant text is the output; its JSON events (bar the
    streaming deltas) are the log.  Step and cost limits are enforced by
    watching ``turn_end`` / ``message_end`` events and stopping the process.
    """
    binary = _pi_bin()
    if not binary:
        raise FileNotFoundError(
            'model_type="agent" needs the pi coding agent: '
            "npm i -g @mariozechner/pi-coding-agent (or set PROMPTMAP_PI_BIN)"
        )
    config = call.spec.config
    workdir = Path(config["agent_dir"])
    cmd = [binary, "--print", "--mode", "json", "--no-session"]
    model = _agent_model_name(call.spec)
    if model:
        cmd += ["--model", model]
    cmd.append(task)

    step_limit = call.spec.config_int("agent_step_limit", 0)
    cost_limit = float(config.get("agent_cost_limit", 3.0))

    started = time.monotonic()
    proc = subprocess.Popen(
        cmd, cwd=str(workdir), text=True,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    events: list = []
    output, error = "", None
    steps, cost, stopped = 0, 0.0, None
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            events.append({"type": "raw", "text": line})
            continue
        kind = event.get("type")
        if kind in _NOISY_EVENTS:
            continue
        if kind == "agent_end":
            events.append({"type": "agent_end"})  # its messages repeat the log
            continue
        events.append(event)
        message = event.get("message") or {}
        if kind == "message_end" and message.get("role") == "assistant":
            cost += float(((message.get("usage") or {}).get("cost") or {}).get("total") or 0)
            text = _assistant_text(message)
            if text:
                output = text
            if message.get("stopReason") == "error":
                error = message.get("errorMessage") or "pi reported an error"
        elif kind == "turn_end":
            steps += 1
            report_progress(call.spec.name, steps, step_limit)
        if kind in ("turn_end", "message_end"):
            if step_limit and steps >= step_limit:
                stopped = f"step limit ({step_limit}) reached"
            elif cost_limit and cost >= cost_limit:
                stopped = f"cost limit (${cost_limit}) reached"
            if stopped:
                proc.terminate()
                break
    stderr = proc.stderr.read() if proc.stderr else ""
    proc.wait()

    if stopped:
        events.append({"type": "exit", "reason": stopped})
    elif error:
        raise RuntimeError(f"pi: {error}")
    elif proc.returncode != 0:
        raise RuntimeError(f"pi exited with {proc.returncode}: {stderr.strip() or 'no error reported'}")
    return {
        "output": output,
        "logs": json.loads(json.dumps(events, default=str)),
        "time_run": round(time.monotonic() - started, 3),
    }


#: Streaming events: every delta of every message.  ``message_end`` and
#: ``tool_execution_end`` carry the finished versions.
_NOISY_EVENTS = frozenset({"message_update", "tool_execution_update"})


def _assistant_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    return "".join(
        part.get("text", "") for part in content or ()
        if isinstance(part, dict) and part.get("type") == "text"
    )


def _pi_bin() -> str | None:
    """Path to the pi binary — a seam for tests."""
    return os.getenv("PROMPTMAP_PI_BIN") or shutil.which("pi")


def _agent_model_name(spec) -> str | None:
    """config(agent_model=...), else client.py's ``agent_model``, else pi's default."""
    if spec.config.get("agent_model"):
        return str(spec.config["agent_model"])
    from promptmap.llm import client_module_for

    module = client_module_for(spec.path)
    name = getattr(module, "agent_model", None) if module is not None else None
    return str(name) if name else None


#: A coding agent: Dir in, patched Dir out.
KIND = register_model_kind(ModelKind(
    name=NAME,
    exec_fn=run_agent,
    config_keys=frozenset({
        "agent_model", "agent_step_limit", "agent_cost_limit",
        "jev_cache_threshold", "merge_max_files", "context_max_chars", "force_refresh",
    }),
))
