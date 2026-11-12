"""
``agent`` — the mini-swe-agent coding agent, text in, text out.

The rendered template is the task.  The agent works in a fresh scratch directory and
its final reply is the model's output, like an LLM call that can run tools
along the way.  Nothing on your disk is touched.

Files work as they do for any model:

* **In:** upstream files named in ``promptfiles`` are copied into
  ``inputs/`` in the workspace, and the task says so.
* **Out:** anything the agent writes into ``outputs/`` becomes this model's
  files (an :class:`~promptmap.files.Output`: its reply plus those files).

For a folder in, patched folder out — with diff-aware reuse — use
``coding_agent``.

Config keys: ``agent_model``, ``agent_step_limit``, ``agent_cost_limit``
(see :mod:`promptmap.executor.mini_swe`).  Results go through the prompt cache, keyed
on the task and the input files, so an unchanged task does not re-run.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any

from promptmap.files import (
    Dir, File, FileOutputError, Output, iter_files, safe_relpath, select_path, split_model_path,
)
from promptmap import agent_step_cache
from promptmap.model_types import ModelCall, ModelKind, register_model_kind

from . import mini_swe

NAME = "agent"

_FILES_NOTE = """

Your working directory is {workdir}.{inputs}
To hand files on to the next step, write them into {workdir}/outputs/.
When you are done, your final reply is the output passed on to the next step.
"""


def input_files(call: ModelCall) -> list[tuple[str, File]]:
    """``(relative path, File)`` for every upstream file named in ``promptfiles``."""
    found: list[tuple[str, File]] = []
    for name in call.spec.promptfiles_used:
        hit = split_model_path(name, call.outputs)
        if hit is None:
            continue  # a run-level --promptfile, not an upstream model
        model, path = hit
        value = select_path(call.outputs[model], path, name)
        found += [(safe_relpath(f"{model}/{rel}"), f) for rel, f in iter_files(value)]
    return found


async def run_agent(rendered: str, call: ModelCall) -> Any:
    inputs = input_files(call)
    digest = hashlib.sha256("\n".join(f"{p}:{f.sha256}" for p, f in inputs).encode()).hexdigest()
    key = f"{rendered}\x00inputs:{digest}"
    return await call.compute(
        key, compute=lambda: asyncio.to_thread(_work, rendered, call, inputs, key),
    )


def _work(rendered: str, call: ModelCall, inputs: list[tuple[str, File]], key: str = "") -> Any:
    # Same key, same folder: a restarted run replays its steps (agent_step_cache).
    with agent_step_cache.scratch_dir("promptmap-agent-", f"{call.spec.name}\x00{key}") as workdir:
        for rel, file in inputs:
            target = workdir / "inputs" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(file.read_bytes())
        (workdir / "outputs").mkdir()
        listed = "".join(f"\n  inputs/{rel}" for rel, _ in inputs)
        note = _FILES_NOTE.format(
            workdir=workdir,
            inputs=f"\nInput files are in {workdir}/inputs/:{listed}" if inputs else "",
        )
        result = _exec_agent(rendered + note, call, workdir)
        files = _collect(workdir / "outputs")  # File reads its bytes now
    text = result["output"]
    return Output(text, files=files) if files else text


def _exec_agent(task: str, call: ModelCall, workdir: Path) -> dict:
    """Run mini-swe-agent — a seam for tests."""
    return mini_swe.run_mini_swe(task, call, workdir)


def _collect(out_dir: Path) -> list:
    items = []
    for path in sorted(out_dir.iterdir()):
        if path.is_symlink():
            raise FileOutputError(f"outputs/: refusing symlink '{path.name}'.")
        items.append(Dir(path) if path.is_dir() else File(path))
    return items


#: The mini-swe-agent agent, text in, text out.
KIND = register_model_kind(ModelKind(name=NAME, exec_fn=run_agent, config_keys=mini_swe.CONFIG_KEYS))
