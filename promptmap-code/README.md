# promptmap-code (`pm-code`)

A terminal coding assistant that builds software by building a
[promptmap](https://pypi.org/project/promptmap/) pipeline. It works like Claude
Code, with one difference: it never edits files itself. Its tools change the
pipeline (`map/*.prompt`). Code is written only when `coding_agent` nodes run.

```bash
pip install "promptmap[code]"          # or: pip install promptmap-code
pm-code -m anthropic/claude-sonnet-5-5  # chat in the terminal
pm-code "build a todo CLI in python"   # one request, then exit
pm-code -i "build a todo CLI"          # one request, then keep chatting
```

Defaults:

- **Project:** the current directory (`--root`). Nodes go in `map/`, run history in `.promptmap/`.
- **Code:** `coding_agent` nodes write into the current directory (`--code-dir`), through their `project=` config key.
- **API keys:** read from the environment and the project's `.env`, like `promptmap run`. A missing key stops pm-code at startup, with the variable's name.
- **Model:** `--model` → `settings.yml` → `$PM_CODE_MODEL` / `$MSWEA_MODEL_NAME`. If the project has no `settings.yml`, pm-code writes one so the nodes use the same model.

## Tools

| Tool | Does |
|---|---|
| `list_nodes(prefix?)` | Nodes with type and dependencies. `prefix="1_plan/2_api/*"` filters by path under `map/` |
| `show_node(name)` | A node's template text |
| `create_chat_node(name, request, folder?, chat_instruction?)` | A `model_type="chat"` node |
| `create_coding_agent_node(name, task, folder?, inherit_coding_folder?)` | A `coding_agent` node that writes to the code dir |
| `set_node_prompt(name, prompt)` | Replace a node's text, keeping its `config()` |
| `rename_node(old, new)` | Rename a node, its `ref()`s and its run history |
| `connect_nodes(upstream, downstream, as_folder?)` | Add a `ref()`, or continue another coding_agent's code |
| `list_jinja_functions()` | Every Jinja function a template can call, with help text |
| `add_jinja(name, jinja, position?)` | Add Jinja to a node. The syntax is checked first |
| `run_nodes(names?)` | Run nodes and everything upstream of them |
| `chat_with_node(name, message)` | Send a follow-up to a chat node |
| `read_node_output(name)` | Latest output. For a coding_agent: its answer and the files it changed |
| `read_code_dir(path?)` | List or read the code dir. Read-only, and limited to the code dir |

## What the model is told

The system prompt is built in three layers. The earlier layer wins where they disagree:

1. **pm-code's rules:** you only have these tools, how to work, and where code goes.
2. **`PM_CODE.md`** in the project root, if there is one. Use it for per-project instructions, like a CLAUDE.md.
3. **promptmap's skill guide** (`skills/promptmap/SKILL.md`) as reference: how to shape a pipeline.
   It comes with a table that maps each `promptmap …` command or file edit to the tool that replaces it.
   `--no-skill` leaves it out.

mini-swe-agent's own `default.yaml` prompts are not used: they teach a bash
workflow, and pm-code has no shell.

The coding agents that `coding_agent` nodes run are told something different:
mini-swe's bash prompts (with promptmap's submit command in place of its bare
`echo`), the node's task, and the code folder's `AGENTS.md` (else `CLAUDE.md`).
`project_instructions=false` on a node leaves that file out. Editing `AGENTS.md`
alone does not re-run a node: use `force_refresh` or `promptmap clear-cache NODE`.

While nodes run, pm-code prints each node's start and finish, with a spinner
showing coding agents' step counts.

The guide ships as `promptmap_code/promptmap_skill.md`, a copy of the repo
file. A test fails if the two differ.

## How it is built

pm-code is [mini-swe-agent](https://mini-swe-agent.com) with two parts swapped
(`promptmap_code/agent.py`):

- **Model:** `ToolModel` is mini-swe's `LitellmModel` with the tools above
  instead of `bash`. A reply with no tool call is the answer to the user.
- **Environment:** `PromptmapEnvironment` runs each action as a call on
  `promptmap.workspace.Workspace` instead of a shell command.

`promptmap.workspace.Workspace` is promptmap's Python API for editing, running
and reading nodes. It works without pm-code:

```python
from promptmap.workspace import Workspace

ws = Workspace(".")
ws.add_chat_node("spec", "Spec a todo CLI.", folder="1_plan")
ws.add_coding_agent_node("build", "Implement the spec as todo.py.")
ws.connect("spec", "build")
ws.run(["build"])
print(ws.output("build"))
ws.chat("spec", "Add due dates")
```
