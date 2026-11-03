# promptmap-extra-model-kinds

Extra model kinds for [promptmap](https://pypi.org/project/promptmap/):

| `model_type`   | Module          | Runs                                                   |
|----------------|-----------------|--------------------------------------------------------|
| `python_modal` | `modal_exec`    | the bullet's upstream Python in a Modal sandbox        |
| `agent_modal`  | `agent_exec`    | an OpenCode agent (plus optional MCP server) on Modal  |
| `agent37`      | `agent37_exec`  | a Hermes agent on an Agent37 workspace (`agent37.md`)  |
| `execute_python` | `python_exec` | the rendered template as Python, in-process; output is `stdout` or an `output` variable |
| `agent_multi_harness` | `agent_multi_harness_exec` | a coding agent in place in `agent_dir`: mini-swe-agent (default) or opencode with MCP servers (`agent_backend`) |

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

## Moved here from promptmap core (promptmap ≥ 0.6)

- `execute_python` — unchanged, just packaged here.
- `agent_multi_harness` — promptmap's old `agent` kind (mini-swe-agent / opencode,
  working in `agent_dir`). Rename `model_type="agent"` to
  `model_type="agent_multi_harness"`; config keys are unchanged.

`coding_agent` moved the other way: it is now in promptmap core, on the
[pi](https://pi.dev) harness, under the same name. See promptmap's
[agent docs](https://github.com/conradbez/promptmap/blob/main/docs/ADVANCED_USAGE.md#agent-models-model_typeagent).
