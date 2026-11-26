"""
pm-code's agent: mini-swe-agent's loop, with promptmap tools instead of bash.

mini-swe-agent has three parts, and pm-code swaps two of them:

* **Model** — :class:`ToolModel` is its ``LitellmModel`` with pm-code's tool
  list in place of the single ``bash`` tool.  A reply with no tool call is not
  a format error here: it is the answer to the user.
* **Environment** — :class:`PromptmapEnvironment` executes an action by calling
  the tool on a :class:`~promptmap.workspace.Workspace` instead of running a
  shell command.  There is no shell: pm-code cannot touch files directly.
* **Agent** — :class:`PmCodeAgent` is ``DefaultAgent`` kept across turns: each
  :meth:`~PmCodeAgent.send` adds the user's message to the same conversation
  and steps until the model replies without calling a tool.
"""

from __future__ import annotations

import json
import os
import platform
import time
from typing import Any, Callable

os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")
os.environ.setdefault("MSWEA_COST_TRACKING", "ignore_errors")

from minisweagent.agents.default import DefaultAgent  # noqa: E402
from minisweagent.exceptions import FormatError, LimitsExceeded  # noqa: E402
from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig  # noqa: E402

from promptmap_code.tools import TOOLS, Toolbox  # noqa: E402

SYSTEM_PROMPT = """\
You are pm-code, an assistant that builds software by building a promptmap pipeline.

A promptmap pipeline is a DAG of nodes, each a .prompt file under map/. You never
write files yourself: you change the pipeline with your tools, and code is written
only when coding_agent nodes run.

Node types you create:
- chat: an LLM answer (a spec, a plan, a review). Refine it with chat_with_node.
- coding_agent: a coding agent that does a task and writes the code into the code
  directory ({code_dir}). Its working folder is the root of that directory, so a
  task names paths from there ("create hello.py"), not "in the code directory".
  A coding_agent that continues another's code uses inherit_coding_folder (or
  connect_nodes with as_folder=true); otherwise it starts from an empty folder.

Nodes read each other through Jinja: connect_nodes inserts {{{{ ref('upstream') }}}}.
list_jinja_functions shows everything else a template can do (previous runs,
conditional skips, promptdata, ...); add_jinja puts it into a node.

How to work:
1. Look first: list_nodes and read_code_dir before changing things.
2. Prefer small, focused nodes wired together over one huge prompt. Group related
   nodes in folders (e.g. 1_plan/, 2_build/).
3. Run what you changed (run_nodes) and check results with read_node_output and
   read_code_dir. Fix the pipeline, not the code: edit prompts and rerun.
4. When you are done, or need the user, reply in plain text without calling a tool.
"""

#: Put before the promptmap skill: it is written for an agent with a shell.
SKILL_PREAMBLE = """# Reference: the promptmap skill

Below is promptmap's own guide. It was written for an agent with a shell and a
file editor; you have neither. Follow its advice on *how to shape a pipeline*,
and where it says to run a command or edit a file, use the tool instead:

| The guide says | You use |
|---|---|
| create / edit `map/x.prompt` | create_chat_node, create_coding_agent_node, set_node_prompt, add_jinja |
| `{{{{ config(...) }}}}` keys for a node | the create_* tools' arguments, connect_nodes(as_folder=true) |
| `promptmap ls` / reading map files | list_nodes, show_node |
| `promptmap rename` | rename_node |
| `promptmap run` | run_nodes |
| `promptmap chat` | chat_with_node |
| `promptmap show-result` | read_node_output |
| reading the code / outputs/ | read_code_dir |
| `promptmap test`, `.testprompt`, init, docs, serve | not available to you: tell the user if one is needed |

Where the guide and the instructions above it disagree, the instructions above win.
"""


def _strip_frontmatter(text: str) -> str:
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            return text[end + 4:].lstrip("\n")
    return text


def load_skill() -> str | None:
    """promptmap's SKILL.md, as copied into this package (kept in sync by a test)."""
    from pathlib import Path

    here = Path(__file__).resolve().parent
    path = here / "promptmap_skill.md"
    return _strip_frontmatter(path.read_text(encoding="utf-8")) if path.is_file() else None


#: Per-project instructions, read from the project root (like CLAUDE.md).
PROJECT_FILE = "PM_CODE.md"


def build_system_prompt(code_dir, root=None, skill: bool = True) -> str:
    """pm-code's rules, then the project's PM_CODE.md, then the promptmap skill as reference."""
    from pathlib import Path

    parts = [SYSTEM_PROMPT.format(code_dir=code_dir)]
    project = Path(root) / PROJECT_FILE if root is not None else None
    if project is not None and project.is_file():
        parts.append(f"# Project instructions ({PROJECT_FILE})\n\n{project.read_text(encoding='utf-8').strip()}\n")
    text = load_skill() if skill else None
    if text:
        parts.append(SKILL_PREAMBLE.format() + "\n" + text)
    return "\n\n".join(parts)


class ToolModelConfig(LitellmModelConfig):
    observation_template: str = (
        "{% if output.returncode %}<error>\n{% endif %}{{ output.output }}{% if output.returncode %}\n</error>{% endif %}"
    )
    cost_tracking: str = "ignore_errors"


class ToolModel(LitellmModel):
    """mini-swe-agent's LiteLLM model, calling pm-code's tools instead of bash."""

    def __init__(self, tools: list[dict] = TOOLS, **kwargs: Any) -> None:
        super().__init__(config_class=ToolModelConfig, **kwargs)
        self.tools = tools

    def _query(self, messages: list[dict], **kwargs: Any):
        import litellm

        return litellm.completion(
            model=self.config.model_name, messages=messages, tools=self.tools,
            **(self.config.model_kwargs | kwargs),
        )

    def _parse_actions(self, response) -> list[dict]:
        """One action per tool call; none means the model is answering the user."""
        actions = []
        for call in response.choices[0].message.tool_calls or []:
            try:
                args = json.loads(call.function.arguments or "{}")
            except ValueError as exc:
                raise FormatError({
                    "role": "user",
                    "content": f"Tool call arguments for {call.function.name} are not valid JSON: {exc}",
                    "extra": {"interrupt_type": "FormatError"},
                }) from None
            actions.append({"tool": call.function.name, "args": args if isinstance(args, dict) else {},
                            "tool_call_id": call.id})
        return actions


class PromptmapEnvironment:
    """A mini-swe-agent environment whose actions are promptmap tool calls."""

    def __init__(self, toolbox: Toolbox, on_action: Callable[[dict, dict], None] | None = None) -> None:
        self.toolbox = toolbox
        self.on_action = on_action

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        result = self.toolbox.call(action.get("tool", ""), action.get("args", {}))
        output = {"output": result.output, "returncode": 0 if result.ok else 1, "exception_info": ""}
        if self.on_action is not None:
            self.on_action(action, output)
        return output

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        ws = self.toolbox.ws
        return {"root": str(ws.root), "code_dir": str(ws.code_dir), "system": platform.system(), **kwargs}

    def serialize(self) -> dict:
        return {"info": {"config": {"environment_type": f"{__name__}.{type(self).__name__}"}}}


class PmCodeAgent(DefaultAgent):
    """``DefaultAgent`` as a conversation: :meth:`send` one user message at a time."""

    def __init__(self, model: ToolModel, env: PromptmapEnvironment, *, max_steps: int = 40,
                 cost_limit: float = 0.0, skill: bool = True, **kwargs: Any) -> None:
        super().__init__(model, env, system_template=SYSTEM_PROMPT, instance_template="{{ task }}",
                         step_limit=0, cost_limit=cost_limit, **kwargs)
        self.max_steps = max_steps
        ws = env.toolbox.ws
        self.add_messages(model.format_message(
            role="system", content=build_system_prompt(ws.code_dir, ws.root, skill=skill)))

    def send(self, text: str) -> str:
        """Add *text* as the user's message; step until the model answers. Returns the answer."""
        self.add_messages(self.model.format_message(role="user", content=text))
        format_errors = 0
        for _ in range(self.max_steps):
            try:
                message = self.query()
            except FormatError as exc:
                format_errors += 1
                self.add_messages(*exc.messages)
                if format_errors >= self.config.max_consecutive_format_errors:
                    return "Stopped: the model kept making malformed tool calls."
                continue
            except LimitsExceeded:
                return f"Stopped: cost limit (${self.config.cost_limit:.2f}) reached."
            format_errors = 0
            if not message.get("extra", {}).get("actions"):
                return message.get("content") or ""
            self.execute_actions(message)
        note = f"Stopped after {self.max_steps} tool steps this turn; say 'continue' to keep going."
        self.add_messages(self.model.format_message(role="user", content=note))
        return note

    def log(self) -> list[dict]:
        return json.loads(json.dumps(self.messages, default=str))


def build_agent(workspace, model_name: str, *, max_steps: int = 40, cost_limit: float = 0.0, skill: bool = True,
                on_action: Callable[[dict, dict], None] | None = None,
                model_kwargs: dict | None = None) -> PmCodeAgent:
    from promptmap.llm import quiet_litellm

    quiet_litellm()
    model = ToolModel(model_name=model_name, model_kwargs=model_kwargs or {})
    env = PromptmapEnvironment(Toolbox(workspace), on_action=on_action)
    agent = PmCodeAgent(model, env, max_steps=max_steps, cost_limit=cost_limit, skill=skill)
    agent._start_time = time.time()
    return agent
