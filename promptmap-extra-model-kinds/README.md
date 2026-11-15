# promptmap-extra-model-kinds

Extra model kinds for [promptmap](https://pypi.org/project/promptmap/):

| `model_type`   | Module          | Runs                                                   |
|----------------|-----------------|--------------------------------------------------------|
| `python_modal` | `modal_exec`    | the bullet's upstream Python in a Modal sandbox        |
| `agent_modal`  | `agent_exec`    | an OpenCode agent (plus optional MCP server) on Modal  |
| `agent37`      | `agent37_exec`  | a Hermes agent on an Agent37 workspace (`agent37.md`)  |
| `execute_python` | `python_exec` | the rendered template as Python, in-process; output is `stdout` or an `output` variable |
| `agent_multi_harness` | `agent_multi_harness_exec` | a coding agent in place in `agent_dir`: mini-swe-agent (default), opencode with MCP servers, or pi (`agent_backend`) |
| `browser_use` | `browser_use_exec` | a [browser-use](https://github.com/browser-use/browser-use) agent doing the task in a real browser — local and visible by default, cloud / headless on request (see below) |

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

## `browser_use` — a browser agent

The rendered prompt is the task; the agent's answer is the output, and anything
the browser downloads becomes the model's files.

```jinja
{{ config(model_type="browser_use") }}
Find the cheapest direct flight from London to Lisbon next Friday. Reply with airline, time and price.
```

By default it drives a **local** Chromium (fresh temporary profile) in a
**visible window**, so you can watch it. Change that per model or per project:

```jinja
{{ config(model_type="browser_use", browser_headless=True) }}   {# no window #}
{{ config(model_type="browser_use", browser_cloud=True) }}      {# Browser Use Cloud, BROWSER_USE_API_KEY #}
```

```yaml
# settings.yml — project defaults; config() wins over these
browser_use:
  headless: true
  cloud: false
  model: low          # a settings.yml alias, provider/model, or a Pydantic AI string
  max_steps: 50
```

| Config key | Default | |
|---|---|---|
| `browser_model` | settings.yml's provider/model | alias (`"low"`), `provider/model`, or `"openai:gpt-4o"`; with no model anywhere, browser-use's `bu-fast` (`BROWSER_USE_API_KEY`) |
| `browser_cloud` | `False` | `True`: a Browser Use Cloud browser instead of a local one |
| `browser_headless` | `False` | `True`: no window. On Linux with no display it runs headless anyway |
| `browser_max_steps` | `100` | step cap |
| `browser_allowed_domains` | — | `["example.com", "*.example.org"]` |

`PROMPTMAP_BROWSER_HEADLESS=1` / `PROMPTMAP_BROWSER_CLOUD=1` set the defaults
from the environment (e.g. CI). Results are cached like any prompt.

Install with `pip install "promptmap[browser]"` (or
`"promptmap-extra-model-kinds[browser]"`): browser-use is pinned exactly,
as it asks, and like promptmap's litellm pin. A local browser needs Chrome,
Chromium or Edge. browser-use's telemetry is off unless you set
`ANONYMIZED_TELEMETRY` yourself. A provider key passed as
`run(secrets={"OPENAI_API_KEY": ...})` reaches openai / anthropic / gemini
models; otherwise keys come from the environment or `.env`.

## Moved here from promptmap core (promptmap ≥ 0.6)

- `execute_python` — unchanged, just packaged here.
- `agent_multi_harness` — promptmap's old `agent` kind (mini-swe-agent / opencode,
  working in `agent_dir`). Rename `model_type="agent"` to
  `model_type="agent_multi_harness"`; config keys are unchanged.

`coding_agent` moved the other way: it is now in promptmap core, on the
[mini-swe-agent](https://mini-swe-agent.com) harness, under the same name.
The [pi](https://pi.dev) harness lives here, as `agent_backend="pi"` on
`agent_multi_harness` (needs Node: `npm i -g @mariozechner/pi-coding-agent`,
or install with the `[npm]` extra and pi / opencode are npm-installed on first
use at a pinned version; Python ≥ 3.12). See promptmap's
[agent docs](https://github.com/conradbez/promptmap/blob/main/docs/ADVANCED_USAGE.md#agent-models-model_typeagent).
