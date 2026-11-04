"""
The mini-swe-agent harness (https://mini-swe-agent.com): a minimal bash-only
agent, run in-process in a directory.  Shared by the ``agent`` and
``coding_agent`` kinds (and the extras' ``agent_multi_harness``).

Pure Python (``pip install mini-swe-agent``, a promptmap dependency), so no
Node toolchain is needed.  The LLM is picked by litellm name from, in order:
``config(agent_model=...)``, client.py's ``agent_model``, ``MSWEA_MODEL_NAME``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from promptmap.model_types import ModelCall, report_progress

#: The config keys the harness reads.
CONFIG_KEYS = frozenset({"agent_model", "agent_step_limit", "agent_cost_limit"})


def run_mini_swe(task: str, call: ModelCall, workdir: Path) -> dict:
    """Run mini-swe-agent in *workdir* with *task*; ``{"output", "logs", "time_run"}``.

    The output is what the agent submitted; the log is its message trajectory.
    On hitting ``agent_step_limit`` / ``agent_cost_limit`` the agent stops and
    the output is empty (the log's last message says why).
    """
    # Its import banner (version, v2 migration link) is noise in a promptmap run.
    os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")
    try:
        import yaml
        from minisweagent import package_dir
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.environments.local import LocalEnvironment
    except ImportError as exc:
        raise ImportError(
            f'model_type="{call.spec.model_type}" needs mini-swe-agent: pip install mini-swe-agent'
        ) from exc

    step_limit = call.spec.config_int("agent_step_limit", 0)

    class _Agent(DefaultAgent):
        def query(self) -> dict:
            message = super().query()
            report_progress(call.spec.name, self.n_calls, step_limit)
            return message

    defaults = yaml.safe_load((package_dir / "config" / "default.yaml").read_text())
    model_cfg = defaults["model"] | {
        "observation_template": defaults["model"]["observation_template"] + _BUDGET_TEMPLATE,
    }
    agent = _Agent(
        _agent_model(agent_model_name(call.spec), model_cfg),
        LocalEnvironment(cwd=str(workdir), **defaults["environment"]),
        **defaults["agent"] | {
            "step_limit": step_limit,
            "cost_limit": float(call.spec.config.get("agent_cost_limit", 3.0)),
        },
    )
    started = time.monotonic()
    result = agent.run(task + _SUBMIT_INSTRUCTION)
    return {
        "output": result.get("submission", ""),
        "logs": json.loads(json.dumps(agent.messages, default=str)),
        "time_run": round(time.monotonic() - started, 3),
    }


#: mini-swe-agent's own prompt says to submit with a bare
#: ``echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT``, which submits nothing.  promptmap
#: wants a final output, which is whatever the agent prints after that marker.
_SUBMIT_INSTRUCTION = """

When you are done, submit your final output (the answer to pass on to the next
step) by printing COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT on the first line and
the output after it, e.g.:

    printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\n%s\\n' "your final output"
"""


#: Appended to every command result so the agent knows what it has left.
#: Template vars come from the agent: n_model_calls / model_cost so far, and
#: its step_limit / cost_limit (0 = none).
_BUDGET_TEMPLATE = """
{%- if step_limit or cost_limit %}
<budget>
{%- if step_limit %} Step {{ n_model_calls }} of {{ step_limit }} ({{ step_limit - n_model_calls }} left).{% endif %}
{%- if cost_limit %} ${{ '%.2f' % model_cost }} of ${{ '%.2f' % cost_limit }} spent.{% endif %}
{%- if step_limit and step_limit - n_model_calls <= 1 %} Your next response is your last: submit your final output now.
{%- else %} Submit your final output before you run out; work cut off by a limit is lost.{% endif %}
</budget>
{%- endif %}"""


def agent_model_name(spec) -> str | None:
    """config(agent_model=...), else client.py's ``agent_model``, else MSWEA_MODEL_NAME."""
    if spec.config.get("agent_model"):
        return str(spec.config["agent_model"])
    from promptmap.llm import client_module_for

    module = client_module_for(spec.path)
    name = getattr(module, "agent_model", None) if module is not None else None
    return str(name) if name else os.getenv("MSWEA_MODEL_NAME")


def _agent_model(name: str | None, model_cfg: dict):
    """The mini-swe-agent model to drive the agent — a seam for tests."""
    from minisweagent.models import get_model

    return get_model(name, model_cfg)
