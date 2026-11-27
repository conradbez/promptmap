# promptmap-code (`pm-code`)

A terminal tool for building software as a [promptmap](https://pypi.org/project/promptmap/)
pipeline, one chat at a time. You pick a chat and talk it through. When you are
done, its final answer creates the next nodes: 1 to 5 coding agents, and more
chats for parts that are still open. Each new node `ref()`s the chat it came from.

```bash
pip install "promptmap[code]"                 # or: pip install promptmap-code
pm-code -m anthropic/claude-sonnet-5-5        # pick a chat, or name a new one
pm-code todo                                  # go straight into chat 'todo' (new name: starts it)
```

## How it works

1. **Pick a chat.** pm-code lists the project's chat nodes, indented by folder. Type a
   number or name, or a new name to start a chat.
2. **Original intent.** A new chat asks what you want from it. That text is the chat's
   `.prompt` file and the first message of its conversation. It stays as you wrote it.
3. **Refine.** Each line you type after that refines the intent. The chat replies with
   its complete, updated answer every time.
4. **Tools.** A line starting with `/` is a tool. In a terminal, typing `/` shows them in a
   dropdown, and a unique prefix is enough (`/con`).
5. **`/done`.** pm-code evaluates the chat's **last** reply only and calls the pm tools it
   names (below). Earlier replies are never evaluated.
6. **Go deeper.** An empty line returns to the list. A child chat works the same way, and
   its children go one folder deeper (`map/<chat>/<child>/`).

### Slash tools (you, in a chat)

| Tool | Does |
|---|---|
| `/done` | call the pm tools in the last reply, then offer to run new coding agents |
| `/connect [node]` | make this chat read another node (a `ref()` after its intent). Searchable list, nearest in the map first |
| `/model [provider/model]` | change this chat's model (`provider=`/`model=` in its config); no argument opens the provider and model pickers; `/model default` resets |
| `/run` | run this chat's coding agents |
| `/redo` | answer again now, e.g. after `/connect` or `/model` |
| `/map` | where this chat sits in the pipeline |
| `/intent`, `/history` | the original intent; the intent, refinements and replies |
| `/back`, empty line | back to the chat list (Ctrl-D quits) |

### pm tools (the chat, in its last reply)

```markdown
### coding_agent: build_cli
Create todo.py with add, list and done commands.

### chat: storage
Work out how tasks are stored.

### connect: storage -> build_cli
### model: build_cli -> anthropic/claude-sonnet-5-5
```

- `coding_agent` / `chat` create a node in the chat's folder (`map/<chat>/build_cli.prompt`)
  with `{{ ref('<chat>') }}`, so it reads the chat's whole answer. At most 5 of each.
  A later `/done` updates them from the newer reply and keeps their connected inputs.
  A name already used elsewhere in the map is skipped.
- `connect` adds a `ref()` from one node to another.
- `model` sets a node's model (`agent_model=` for a coding agent).
- A reply cannot connect into, or change the model of, its own chat. Use `/connect` and `/model` for that.

## The map

Every node pm-code writes includes `{{ org_outline(up=0, used_by=False) }}`: a bullet
tree of the node and the nodes it was built from, so each chat and coding agent
knows where it sits. `/map` prints the whole pipeline in the same format, with the
current chat marked `[you are here]`. Top bullets are the final outputs and each
node's inputs are under it:

```text
implement_todo_cli
- todo [you are here]
todo_readme (also refs todo)
todo_tests (also refs todo)
```

`used_by=False` keeps a chat's prompt unchanged when children are added to it.
Otherwise running a child would regenerate the chat's reply.

## Defaults

- **Project:** the current directory (`--root`). Nodes go in `map/`, run history in `.promptmap/`.
- **Code:** coding agents write into the current directory (`--code-dir`), through their `project=` config key.
- **API keys:** read from the environment and the project's `.env`. A missing key stops pm-code at startup, with the variable's name.
- **Model:** the nodes use `settings.yml`. `--model` writes one when the project has none; else `$PM_CODE_MODEL`.

## Python API

pm-code is a thin layer over `promptmap_code.Session`, which works on
`promptmap.workspace.Workspace`:

```python
from promptmap.workspace import Workspace
from promptmap_code import Session

s = Session(Workspace("."))
s.create_chat("todo", "A tiny python todo CLI.")
print(s.reply("todo"))                  # first reply
print(s.reply("todo", "Finish now."))   # a refinement
s.connect_into("todo", "constraints")   # /connect
s.set_model("todo", "openai", "gpt-5")  # /model
done, plan = s.apply("todo")            # /done: pm tools in the last reply
s.run([c.name for c, status in done if c.tool == "coding_agent"])
print(s.outline("todo"))
```
