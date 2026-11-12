"""
The agent step cache: every LLM response a mini-swe-agent run received, keyed
on the exact request that produced it.

promptmap's own cache stores an agent's *finished* result.  An agent that is
cut off mid-run (Ctrl-C, a crash, a closed laptop) has no finished result, so
the restart would pay for every step again.  With this cache it doesn't: the
restart sends the same task, gets the same first reply back from here, runs
the same command (the commands always run for real, so the scratch folder ends
up in the same state), sees the same output, and so on — word for word — until
the first point where the conversation differs (where the old run stopped, or
a command printed something new).  From there on the LLM is called as usual.

It lives next to the run database (``.promptmap/agent_steps.db``), separate
from the prompt cache, and is cleared with it: ``promptmap run --clear-cache``
and ``promptmap clear-cache`` forget the steps too.

``PROMPTMAP_AGENT_STEP_CACHE`` overrides the path; ``0`` / ``off`` turns the
cache off.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

#: Steps not used for this long are dropped when the cache is opened.
_MAX_AGE_DAYS = 30


def cache_path() -> Path | None:
    """Where the cache lives, or None when it is turned off."""
    override = os.getenv("PROMPTMAP_AGENT_STEP_CACHE", "").strip()
    if override.lower() in ("0", "off", "false", "no"):
        return None
    if override:
        return Path(override).expanduser()
    from promptmap import db

    return db.db_path().parent / "agent_steps.db"


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS agent_steps ("
        " key TEXT PRIMARY KEY, model_name TEXT NOT NULL,"
        " response TEXT NOT NULL, used_at REAL NOT NULL)"
    )
    return conn


def step_key(llm_name: str, messages: list, kwargs: dict) -> str:
    """The exact request, as a hash: model, every message, every parameter."""
    blob = json.dumps({"model": llm_name, "messages": messages, "kwargs": kwargs},
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def wrap_query(model: Any, model_name: str) -> None:
    """Route *model*'s litellm call (``_query``) through the cache, in place.

    *model_name* is the promptmap model, so ``clear-cache MODEL`` can forget
    just its steps.  A model without ``_query`` (a test double, a non-litellm
    backend) is left alone.
    """
    path = cache_path()
    query: Callable | None = getattr(model, "_query", None)
    if path is None or query is None:
        return
    llm_name = str(getattr(getattr(model, "config", None), "model_name", ""))
    _prune(path)

    def cached_query(messages: list, **kwargs: Any) -> Any:
        key = step_key(llm_name, messages, kwargs)
        hit = _get(path, key)
        if hit is not None:
            return hit
        response = query(messages, **kwargs)
        _put(path, key, model_name, response)
        return response

    model._query = cached_query


def _get(path: Path, key: str) -> Any:
    with _connect(path) as conn:
        row = conn.execute("SELECT response FROM agent_steps WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        conn.execute("UPDATE agent_steps SET used_at = ? WHERE key = ?", (time.time(), key))
    import litellm

    return litellm.ModelResponse(**json.loads(row[0]))


def _put(path: Path, key: str, model_name: str, response: Any) -> None:
    try:
        data = json.dumps(response.model_dump(mode="json"))
    except Exception:  # an odd response object: just don't cache it
        return
    with _connect(path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO agent_steps (key, model_name, response, used_at) VALUES (?, ?, ?, ?)",
            (key, model_name, data, time.time()),
        )


def _prune(path: Path) -> None:
    with _connect(path) as conn:
        conn.execute("DELETE FROM agent_steps WHERE used_at < ?",
                     (time.time() - _MAX_AGE_DAYS * 86400,))


def clear(models: list[str] | None = None) -> int:
    """Forget the steps of *models* (all when None); the number removed."""
    path = cache_path()
    if path is None or not path.exists():
        return 0
    with _connect(path) as conn:
        if models is None:
            return conn.execute("DELETE FROM agent_steps").rowcount
        if not models:
            return 0
        marks = ",".join("?" * len(models))
        return conn.execute(f"DELETE FROM agent_steps WHERE model_name IN ({marks})", models).rowcount


@contextmanager
def scratch_dir(prefix: str, key: str) -> Iterator[Path]:
    """A scratch folder at the same path every time for the same *key*.

    The agent sees its working directory (``pwd``, error messages, the task
    text), so a random ``mkdtemp`` name would make every restart differ from
    the first step and never hit the step cache.  The folder is emptied
    first, which also clears what a killed run left there.  A lock keeps two
    runs with the same key apart: the second gets a random folder instead.
    """
    path = Path(tempfile.gettempdir()) / f"{prefix}{hashlib.sha256(key.encode()).hexdigest()[:16]}"
    lock = _try_lock(path.with_name(path.name + ".lock"))
    if lock is None:
        path = Path(tempfile.mkdtemp(prefix=prefix))
    else:
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
        if lock is not None:
            lock.close()  # releases the flock


def _try_lock(lock_path: Path):
    """An exclusive non-blocking lock on *lock_path* (an open file), or None."""
    try:
        import fcntl
    except ImportError:  # Windows: no shared path, just random folders
        return None
    handle = open(lock_path, "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle
