"""
The pi harness (https://pi.dev): run the pi coding agent headless in a
directory and collect its answer.  Shared by the ``agent`` and
``coding_agent`` kinds.

Needs the ``pi`` binary (``npm i -g @mariozechner/pi-coding-agent``) on PATH,
or at ``PROMPTMAP_PI_BIN``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from promptmap.model_types import ModelCall, report_progress

#: The config keys the harness reads.
CONFIG_KEYS = frozenset({"agent_model", "agent_step_limit", "agent_cost_limit"})


def run_pi(task: str, call: ModelCall, workdir: Path) -> dict:
    """Run pi in *workdir* with *task*; ``{"output", "logs", "time_run"}``.

    pi's final assistant text is the output; its JSON events (bar the
    streaming deltas) are the log.  Step and cost limits are enforced by
    watching ``turn_end`` / ``message_end`` events and stopping the process.
    """
    binary = _pi_bin()
    if not binary:
        raise FileNotFoundError(
            f'model_type="{call.spec.model_type}" needs the pi coding agent: '
            "npm i -g @mariozechner/pi-coding-agent (or set PROMPTMAP_PI_BIN)"
        )
    config = call.spec.config
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
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        # Merged into stdout: a separate stderr pipe is never drained while we
        # read stdout, so pi blocks once it writes >16 KB there (deadlock).
        # Non-JSON lines land in the log as {"type": "raw"} events.
        stderr=subprocess.STDOUT,
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
    proc.wait()
    stderr = "\n".join(e["text"] for e in events if e.get("type") == "raw")[-2000:]

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


