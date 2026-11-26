"""
The mini-swe-agent harness (https://mini-swe-agent.com): a minimal bash-only
agent, run in-process in a directory.  Shared by the ``agent`` and
``coding_agent`` kinds (and the extras' ``agent_multi_harness``).

Pure Python (``pip install mini-swe-agent``, a promptmap dependency), so no
Node toolchain is needed.  The LLM is picked by litellm name from, in order:
``config(agent_model=...)``, settings.yml's ``provider``/``model``, ``MSWEA_MODEL_NAME``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from promptmap import agent_step_cache
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

    # Same litellm hygiene as the default backend: no message logging,
    # callbacks or debug output (mini-swe-agent calls litellm directly).
    from promptmap.llm import quiet_litellm

    quiet_litellm()

    step_limit = call.spec.config_int("agent_step_limit", 0)

    class _Agent(DefaultAgent):
        def query(self) -> dict:
            message = super().query()
            report_progress(call.spec.name, self.n_calls, step_limit)
            return message

    defaults = submit_with_output(yaml.safe_load((package_dir / "config" / "default.yaml").read_text()))
    model_cfg = defaults["model"] | {
        "observation_template": defaults["model"]["observation_template"] + _BUDGET_TEMPLATE,
    }
    model = _agent_model(agent_model_name(call.spec), model_cfg)
    # Replay of identical steps after a cut-off run (see agent_step_cache).
    agent_step_cache.wrap_query(model, call.spec.name)
    agent = _Agent(
        model,
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
#: :func:`submit_with_output` puts this command in place of the bare echo in
#: mini-swe's own templates, so the agent is told one way to submit, not two
#: (two made agents chain the submit onto a heredoc, writing it into a file).
SUBMIT_COMMAND = r'''printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n%s\n' "<your final output>"'''
_BARE_SUBMIT = "`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`"

_SUBMIT_INSTRUCTION = f"""

When you are done, submit your final output (the answer to pass on to the next
step): COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT on the first line, the output after
it.  Use this command on its own, never chained with && or ; to another command
(finish writing and checking files first):

    {SUBMIT_COMMAND}
"""


def submit_with_output(defaults: dict) -> dict:
    """mini-swe-agent's default config with its bare-echo submit replaced by :data:`SUBMIT_COMMAND`."""
    out = {section: dict(values) if isinstance(values, dict) else values for section, values in defaults.items()}
    for section, key in (("agent", "instance_template"), ("agent", "system_template"), ("model", "format_error_template")):
        text = out.get(section, {}).get(key)
        if isinstance(text, str):
            out[section][key] = text.replace(_BARE_SUBMIT, f"`{SUBMIT_COMMAND}`")
    return out


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
    """config(agent_model=...), else settings.yml's model, else MSWEA_MODEL_NAME.

    A name that is a settings.yml alias (``agent_model="high"``) becomes the
    litellm ``provider/model`` it points at.  settings.yml comes before the
    env var because mini-swe-agent loads its own global ``.env`` (which may
    set ``MSWEA_MODEL_NAME``) on import: the project's choice should win.
    """
    if spec.config.get("agent_model"):
        return resolve_alias(spec, str(spec.config["agent_model"]))
    from promptmap.llm import litellm_model
    from promptmap.settings import resolve_model, settings_for

    target = resolve_model(settings_for(spec.path)) if getattr(spec, "path", None) else None
    if target is not None:
        return litellm_model(*target)
    return os.getenv("MSWEA_MODEL_NAME") or None


def resolve_alias(spec, name: str | None) -> str | None:
    """*name* as a litellm model string when it is a settings.yml alias, else unchanged."""
    if not name:
        return name
    from promptmap.llm import litellm_model
    from promptmap.settings import model_aliases, settings_for

    aliases = model_aliases(settings_for(spec.path))
    return litellm_model(*aliases[name]) if name in aliases else name


def _agent_model(name: str | None, model_cfg: dict):
    """The mini-swe-agent model to drive the agent — a seam for tests."""
    from minisweagent.models import get_model

    return get_model(name, model_cfg)
