# promptmap-extra-model-kinds

Extra model kinds for [promptmap](https://pypi.org/project/promptmap/):

| `model_type`   | Module          | Runs                                                   |
|----------------|-----------------|--------------------------------------------------------|
| `python_modal` | `modal_exec`    | the bullet's upstream Python in a Modal sandbox        |
| `agent_modal`  | `agent_exec`    | an OpenCode agent (plus optional MCP server) on Modal  |
| `agent37`      | `agent37_exec`  | a Hermes agent on an Agent37 workspace (`agent37.md`)  |
| `coding_agent` | `coding_agent_exec` | the local `agent` kind as a function: a `Dir` in, a patched `Dir` out, reusing its last change when it still holds |

And RAG kinds, called from any template:

| Template call                  | Module         | Returns                                         |
|--------------------------------|----------------|-------------------------------------------------|
| `rag_brave_search(query, count=8)` | `brave_search` | web results via Brave Search (`BRAVE_API_KEY`) |
| `rag_gemini_search(query, instructions=..., model=...)` | `gemini_search` | a Gemini Flash answer grounded in Google Search, then its sources (`GEMINI_API_KEY`, env or `.env`) |
| `rag_openai_search(query, instructions=..., model=...)` | `openai_search` | an OpenAI answer using its `web_search` tool, then its sources (`OPENAI_API_KEY`, env or `.env`) |
| `rag_anthropic_search(query, instructions=..., model=...)` | `anthropic_search` | a Claude answer using its `web_search` tool, then its sources (`ANTHROPIC_API_KEY`, env or `.env`) |

The three LLM searches return `[answer, "title\nurl", ...]`. Override their models with
`GEMINI_SEARCH_MODEL` (default `gemini-3.6-flash`), `OPENAI_SEARCH_MODEL` (`gpt-5.4-mini`)
and `ANTHROPIC_SEARCH_MODEL` (`claude-opus-5-5`, effort `ANTHROPIC_SEARCH_EFFORT=low`).

```jinja
{% set hits = rag_brave_search(ref('topic'), count=5) %}
Use these sources:
{{ hits | join("\n\n") }}
```

Installing the package is enough: promptmap finds the kinds through the
`promptmap.model_kinds` and `promptmap.rag_kinds` (promptmap >= 0.5.16) entry point
groups and loads them on first use. Nothing has to import it.

Configuration is read from the environment — see each module's docstring.

## `coding_agent`

A coding agent as a pure function: a `Dir` in, the patched `Dir` out. The agent
works on a scratch copy of its input, free to build and run tests; the copy only
exists to find out which files to create, edit or delete, and where it lives is
irrelevant. The input always comes from upstream through `promptfiles`, so getting
a real repo into a `Dir` is a static node's job:

```jinja
{# map/repo.prompt — a static node decides what the input tree is #}
{{ config(model_type="execute_python", cache=False) }}
output = Dir("../app")

{# map/api.prompt #}
{{ config(model_type="coding_agent", promptfiles="repo") }}
Add a /health endpoint to the API.

{# map/tests.prompt — takes the previous coding_agent's patched tree #}
{{ config(model_type="coding_agent", promptfiles="api") }}
Add tests for every endpoint.
```

Output: `output` (the agent's answer), `result` (the patched `Dir`), `changes`
(a `Dir` of only the files it created or modified), `deleted`, `sum_of_changes`
(`ast_summary` lines plus a `unified_diff` of result vs input) and `cache`.
Build and cache folders the agent creates while testing (`node_modules`, `.venv`,
`__pycache__`, …) are not counted as changes.

The node keeps no state of its own: its output also carries `base` (the input
tree it worked from), and promptmap hands the whole output back next run as the
model's previous result (`call.previous`, promptmap ≥ 0.5.17) — the same row
templates see as `{{ previous }}`. Editing the `.prompt` file, or deleting the
run database, starts it afresh.

With the same prompt it reuses that change instead of running the agent:

| `cache`  | when                                                        | LLM calls          |
|----------|-------------------------------------------------------------|--------------------|
| `hit`    | input unchanged, or only comments/formatting changed in files this model didn't touch | none |
| `replay` | upstream changed the meaning of files this model didn't touch | verify           |
| `merged` | upstream also edited some of this model's files             | `git merge-file` per file, one LLM merge call for any that conflict, then verify (skipped if upstream's edit there was comments/formatting only and the merge keeps this model's meaning) |
| `miss`   | otherwise, or merge/verify fails                            | the agent          |

"Meaning" comes from [tree-sitter-language-pack](https://pypi.org/project/tree-sitter-language-pack/):
files are compared with comments (except directives like `# type: ignore`) and
formatting stripped, and changes are summarised per symbol — `ADDED_SYMBOL`,
`REMOVED_SYMBOL`, `SIGNATURE_CHANGED`, `BODY_CHANGED`. Grammars download on first
use (into `~/.cache/tree-sitter-language-pack`); files with no grammar, or that
don't parse, are compared byte for byte.

The verifier sees the upstream change (`upstream_input_ast_diff`), everything this
model changed last run (`previous_run_sum_of_changes`) and what reuse would apply now
(`replayed_changes`), each as symbol summary plus unified diff.
Verification uses `classify_call` in client.py (or hosted Jev with
`TYPESAFE_API_KEY`), else a YES/NO LLM call. On a miss the agent is told its
previous change and answer, what changed upstream and why it is re-running.
Extra config keys: `jev_cache_threshold` (0.85), `merge_max_files` (5),
`context_max_chars` (20000), `force_refresh`.
