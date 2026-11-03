"""
``agent_multi_harness`` — a coding agent working in ``agent_dir``, with the
rendered text as its task, on one of several harnesses: mini-swe-agent
(default), opencode (which brings MCP servers) or pi.

Unlike core's ``agent`` / ``coding_agent`` (scratch copies), it works in place
in ``agent_dir``.  Importing this module registers the kind.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from promptmap.files import Dir, File, FileOutputError
from promptmap.executor import mini_swe
from promptmap.model_types import ModelCall, ModelKind, register_model_kind

from . import pi_harness

NAME = "agent_multi_harness"


def _collect_out_dir(out_dir: Path) -> list:
    """The files and directories left in *out_dir*, as File/Dir objects."""
    items = []
    for path in sorted(out_dir.iterdir()):
        if path.is_symlink():
            raise FileOutputError(f"out_dir: refusing symlink '{path.name}'.")
        items.append(Dir(path) if path.is_dir() else File(path))
    return items



async def run_agent(rendered: str, call: ModelCall) -> dict:
    """Run a coding agent with the rendered template as its task.

    The agent works in ``agent_dir`` (created if missing), running shell
    commands there until it submits.  The output is a dict::

        {"output": <what the agent submitted>,
         "logs": <the full message trajectory>,
         "time_run": <seconds the agent ran>}

    so a downstream model reads ``ref('fix')['output']``.  With
    ``output_format="json"`` the submitted text is parsed, so a downstream
    model reads ``ref('fix').output.key``.

    The agent is also given an empty output directory.  Anything it writes
    there becomes this model's files, under ``"files"`` (``File`` / ``Dir``
    objects, stored in the blob store, not left on disk), so downstream
    models read or attach them like any other model's files.

    Config keys:

    ``agent_dir``         working directory (required)
    ``agent_backend``     ``mini-swe-agent`` (default), ``opencode`` or ``pi``
    ``agent_model``       model name; for mini-swe-agent defaults to
                          client.py's ``agent_model``, then ``MSWEA_MODEL_NAME``;
                          opencode's / pi's own config otherwise
    ``agent_step_limit``  max LLM calls, 0 for none (default 0)
    ``agent_cost_limit``  max spend in dollars, 0 for none (default 3)
    ``agent_mcp``         opencode only: MCP servers the agent may use, as a
                          dict (opencode's ``mcp`` block) or a path to an
                          ``opencode.json``

    mini-swe-agent needs ``pip install mini-swe-agent``; opencode needs the
    ``opencode`` binary on PATH (``npm i -g opencode-ai``) or
    ``PROMPTMAP_OPENCODE_BIN``; pi needs the ``pi`` binary
    (``npm i -g @mariozechner/pi-coding-agent``) or ``PROMPTMAP_PI_BIN``.  Results are cached on the rendered prompt like any
    other kind, so an unchanged task does not re-run.
    """
    raw = await call.compute(
        rendered, compute=lambda: asyncio.to_thread(_exec_agent, rendered, call)
    )
    result = json.loads(raw) if isinstance(raw, str) else raw
    if call.spec.output_format == "json":
        from promptmap.executor.run_context import parse_json_output

        result["output"] = parse_json_output(result["output"])
    return result


_AGENT_BACKENDS = ("mini-swe-agent", "opencode", "pi")


def _exec_agent(task: str, call: ModelCall) -> str:
    config = call.spec.config
    workdir = config.get("agent_dir")
    if not workdir:
        raise ValueError(
            f"Model '{call.spec.name}': model_type=\"agent_multi_harness\" needs "
            "config(agent_dir=\"...\") — the directory the agent works in."
        )
    workdir = Path(workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    backend = str(config.get("agent_backend") or _AGENT_BACKENDS[0])
    if backend not in _AGENT_BACKENDS:
        raise ValueError(
            f"Model '{call.spec.name}': agent_backend must be one of "
            f"{', '.join(_AGENT_BACKENDS)}, not '{backend}'."
        )

    # Unique per run, so agents fanned out over one agent_dir never share it.
    # Inside agent_dir, because a backend may refuse to write outside it.
    out_dir = workdir / f".promptmap_out_{uuid.uuid4().hex[:8]}"
    out_dir.mkdir()
    task += _FILES_INSTRUCTION.format(out_dir=out_dir)
    started = time.monotonic()
    try:
        if backend == "opencode":
            output, logs = _exec_opencode(task, call, workdir)
        elif backend == "pi":
            output, logs = _exec_pi(task, call, workdir)
        else:
            output, logs = _exec_mini_swe_agent(task, call, workdir)
        files = _collect_out_dir(out_dir)  # File reads its bytes now
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
    time_run = round(time.monotonic() - started, 3)

    result = {
        "output": output,
        "logs": json.loads(json.dumps(logs, default=str)),
        "time_run": time_run,
    }
    if files:
        result["files"] = files
        return result  # the cache stores files as blobs plus a manifest
    return json.dumps(result)


_FILES_INSTRUCTION = """

To pass files on to the next step (images, notes, data), write them into
{out_dir}. Everything in that directory is handed on with your final output.
"""


# ---------------------------------------------------------------------------
# Backend: mini-swe-agent
# ---------------------------------------------------------------------------

def _exec_mini_swe_agent(task: str, call: ModelCall, workdir: Path) -> tuple[str, list]:
    result = mini_swe.run_mini_swe(task, call, workdir)
    return result["output"], result["logs"]


# ---------------------------------------------------------------------------
# Backend: pi (https://pi.dev) — needs Node
# ---------------------------------------------------------------------------

def _exec_pi(task: str, call: ModelCall, workdir: Path) -> tuple[str, list]:
    result = pi_harness.run_pi(task, call, workdir)
    return result["output"], result["logs"]


# ---------------------------------------------------------------------------
# Backend: opencode (https://opencode.ai) — brings MCP servers
# ---------------------------------------------------------------------------

def _exec_opencode(task: str, call: ModelCall, workdir: Path) -> tuple[str, list]:
    """Run ``opencode run`` headless in *workdir* and collect its JSON events.

    opencode's final reply (the text parts of its last message) is the output;
    every event it printed is the log.  ``agent_mcp`` is handed over through
    ``OPENCODE_CONFIG_CONTENT``, which opencode merges over its own config, so
    the project's ``opencode.json`` still applies.  Step and cost limits are
    enforced by watching ``step_finish`` events and stopping the process.
    """
    binary = _opencode_bin()
    if not binary:
        raise FileNotFoundError(
            "agent_backend=\"opencode\" needs the opencode binary: "
            "npm i -g opencode-ai (or set PROMPTMAP_OPENCODE_BIN)"
        )

    config = call.spec.config
    cmd = [binary, "run", "--format", "json", "--auto", "--dir", str(workdir)]
    if config.get("agent_model"):
        cmd += ["--model", str(config["agent_model"])]
    cmd.append(task + _OPENCODE_INSTRUCTION.format(workdir=workdir))

    env = os.environ.copy()
    for promptmap_key, opencode_key in _OPENCODE_KEY_ALIASES.items():
        if env.get(promptmap_key) and not env.get(opencode_key):
            env[opencode_key] = env[promptmap_key]
    mcp = _opencode_mcp_servers(config.get("agent_mcp"), call.spec.path)
    if mcp:
        env["OPENCODE_CONFIG_CONTENT"] = _merged_opencode_config(env.get("OPENCODE_CONFIG_CONTENT"), mcp)

    step_limit = call.spec.config_int("agent_step_limit", 0)
    cost_limit = float(config.get("agent_cost_limit", 3.0))

    proc = subprocess.Popen(
        cmd, cwd=str(workdir), env=env, text=True,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    events: list = []
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
        events.append(event)
        if event.get("type") == "step_finish":
            steps += 1
            cost += float((event.get("part") or {}).get("cost") or 0)
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
    elif proc.returncode != 0:
        raise RuntimeError(
            f"opencode exited with {proc.returncode}: {stderr.strip() or _last_error(events)}"
        )

    return _opencode_final_text(events), events


#: promptmap's provider keys under the names opencode's SDKs read them from.
#: OPENAI_API_KEY and ANTHROPIC_API_KEY already match.
_OPENCODE_KEY_ALIASES = {"GEMINI_API_KEY": "GOOGLE_GENERATIVE_AI_API_KEY"}


#: Smaller models tend to guess a path for new files; pin them to agent_dir.
_OPENCODE_INSTRUCTION = """

Your working directory is {workdir}. Create and edit files there (relative
paths resolve against it). When you are done, your final reply is the output
passed on to the next step.
"""


def _opencode_bin() -> str | None:
    """Path to the opencode binary — a seam for tests."""
    return os.getenv("PROMPTMAP_OPENCODE_BIN") or shutil.which("opencode")


def _opencode_mcp_servers(value: Any, model_path: Path) -> dict:
    """``agent_mcp`` as opencode's ``mcp`` dict: a dict, JSON text, or a path."""
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    text = str(value).strip()
    if not text.startswith("{"):
        path = Path(text).expanduser()
        if not path.is_absolute():
            path = (model_path.parent / path) if model_path.name != "<inline>" else path
        text = path.read_text()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("agent_mcp must be a JSON object of MCP servers")
    # A whole opencode.json is fine too: take its "mcp" block.
    if "mcp" in data and all(not isinstance(v, dict) or "type" not in v for v in data.values()):
        return data["mcp"]
    return data


def _merged_opencode_config(existing: str | None, mcp: dict) -> str:
    base: dict = {}
    if existing:
        try:
            base = json.loads(existing)
        except json.JSONDecodeError:
            base = {}
    base["mcp"] = {**base.get("mcp", {}), **mcp}
    return json.dumps(base)


def _opencode_final_text(events: list) -> str:
    """The text parts of the last message opencode wrote."""
    texts = [e["part"] for e in events if e.get("type") == "text" and isinstance(e.get("part"), dict)]
    if not texts:
        return ""
    last_message = texts[-1].get("messageID")
    return "".join(p.get("text", "") for p in texts if p.get("messageID") == last_message)


def _last_error(events: list) -> str:
    for event in reversed(events):
        if event.get("type") == "error":
            return str(event.get("error") or (event.get("part") or {}).get("error"))
    return "no error reported"


#: A coding agent working in a directory, with the rendered text as its task.
KIND = register_model_kind(ModelKind(
    name=NAME,
    exec_fn=run_agent,
    config_keys=frozenset({
        "agent_dir", "agent_backend", "agent_model", "agent_step_limit", "agent_cost_limit", "agent_mcp",
    }),
))
