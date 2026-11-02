# promptmap-extra-model-kinds

Extra model kinds for [promptmap](https://pypi.org/project/promptmap/):

| `model_type`   | Module          | Runs                                                   |
|----------------|-----------------|--------------------------------------------------------|
| `python_modal` | `modal_exec`    | the bullet's upstream Python in a Modal sandbox        |
| `agent_modal`  | `agent_exec`    | an OpenCode agent (plus optional MCP server) on Modal  |
| `agent37`      | `agent37_exec`  | a Hermes agent on an Agent37 workspace (`agent37.md`)  |
| `local_functional_agent` | `functional_agent_exec` | the local `agent` kind as a function: a `Dir` in, a patched `Dir` out, reusing its last change when it still holds |

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

## `local_functional_agent`

A coding agent as a pure function. It receives a directory tree, works on a
scratch copy (free to build and run tests) and returns the tree with its
changes already applied. The real repository is never touched.

```jinja
{# map/api.prompt — input: a folder on disk #}
{{ config(model_type="local_functional_agent", base_dir="../app") }}
Add a /health endpoint to the API.

{# map/tests.prompt — input: the previous agent's patched tree #}
{{ config(model_type="local_functional_agent", promptfiles=["api"]) }}
Add tests for every endpoint.
```

Output: `output` (the agent's answer), `result` (the patched `Dir`), `changes`
(a `Dir` of only the files it created or modified), `deleted` and `cache`.
`base_dir` snapshots with `git ls-files` in a git checkout.

Each model remembers its last input tree and change set (`.promptmap/state`).
With the same prompt it reuses that change instead of running the agent:

| `cache`  | when                                               | LLM calls          |
|----------|----------------------------------------------------|--------------------|
| `hit`    | the input tree is unchanged                        | none               |
| `replay` | upstream changed only files this model didn't      | verify             |
| `merged` | upstream also changed some of this model's files   | 3-way merge + verify |
| `miss`   | otherwise, or merge/verify fails                   | the agent          |

Verification uses `classify_call` in client.py (or hosted Jev with
`TYPESAFE_API_KEY`), else a YES/NO LLM call. On a miss the agent is told its
previous change and answer, what changed upstream and why it is re-running.
Extra config keys: `base_dir`, `jev_cache_threshold` (0.85), `merge_max_files` (5),
`context_max_chars` (20000), `force_refresh`, `state_dir`.
