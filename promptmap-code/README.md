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
   number or name, or a new name to start a chat (you then say what it should work out).
2. **Talk.** Each line is a follow-up (`promptmap chat`). The chat replies with its
   complete, updated answer every time.
3. **`/done`.** pm-code reads the chat's **final** reply for these headings:

   ```markdown
   ### coding_agent: build_cli
   Create todo.py with add, list and done commands.

   ### chat: storage
   Work out how tasks are stored.
   ```

   Each heading becomes a node in the chat's folder (`map/<chat>/build_cli.prompt`),
   with `{{ ref('<chat>') }}` so it reads the chat's whole answer. At most 5 coding
   agents and 5 chats per chat. Running `/done` again updates those nodes from the
   newer reply. A name already used elsewhere in the map is skipped.
4. **Run.** pm-code offers to run the new coding agents. They write code into
   `--code-dir`. `/run` runs this chat's coding agents again later.
5. **Go deeper.** `/back` (or an empty line) returns to the list. A child chat works the
   same way, and its children go one folder deeper (`map/<chat>/<child>/`).

| In a chat | Does |
|---|---|
| any text | send a follow-up |
| `/done` | create or update the nodes the final reply names |
| `/run` | run this chat's coding agents |
| `/map` | show where this chat sits in the pipeline |
| `/history` | print the conversation |
| `/back`, empty line | back to the chat list (Ctrl-D quits) |

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
print(s.reply("todo", "Finish now."))   # follow-up
created, plan = s.spawn("todo")         # nodes from the final reply
s.run([c.name for c, status in created if c.kind == "coding_agent"])
print(s.outline("todo"))
```
