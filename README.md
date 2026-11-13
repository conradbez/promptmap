# promptmap

A **data-engineering-inspired** prompt orchestration tool for LLMs (think `dbt` for prompts).

**Why `promptmap`?** Treating complex LLM workflows as giant one-shot prompts is like navigating a city by staring at your feet, one wrong turn and you're lost. `promptmap` turns your prompts into a visual map of connected hypotheses (`models`) and reality-check landmarks (`tests`), with built-in tools to inspect runs (`promptmap docs`).

**Mechanically, this means you:** write modular prompts in Jinja2, chain them into a dependency graph with `ref()`, and validate outputs against reality with `promptmap test`. Iterate on complex LLM workflows without losing context or rewriting from scratch. *(Read more on the philosophy in [motivation.md](motivation.md)).*

![DAG visual](promptmap_docs.png)

---

## Quick start

```bash
uv tool install promptmap
promptmap init my_project   # asks for provider, model + API key; writes settings.yml and .env
# or: promptmap init my_project --provider anthropic --model claude-sonnet-4-5 --api-key sk-...
# or: promptmap init my_project --no-interactive   (blank settings.yml/.env to edit)
```

promptmap calls your LLM through [LiteLLM](https://github.com/BerriAI/litellm) (100+ providers). The provider and model
live in `settings.yml` at the project root — change them with `promptmap run --provider X --model Y`
or edit the file by hand (values aren't validated). Need full control over the call? Add a
[`client.py`](docs/INTERMEDIATE_USAGE.md#customising-the-llm-backend-clientpy).

Edit your `.prompt` files in `map/`, then run the pipeline, test expectations, and inspect the run:

```bash
cd my_project   # API key lives in .env (init adds the right variable)

promptmap run     # Execute the prompt DAG
promptmap test    # Validate outputs against map/*.testprompt
promptmap docs    # Generate HTML report of the run and run history
promptmap serve   # Launch the interactive local UI / HTTP server
promptmap clear-cache MODEL  # Make one model start cold on the next run
```

**Claude Code skill** (teaches Claude the promptmap workflow, prompt style and testing approach):

```bash
mkdir -p ~/.claude/skills/promptmap && curl -fsSL https://raw.githubusercontent.com/conradbez/promptmap/main/skills/promptmap/SKILL.md -o ~/.claude/skills/promptmap/SKILL.md
```

---

## Core concepts

### 1. Models (`map/*.prompt`)

Each `.prompt` file is a node in your DAG. Use `ref('model_name')` to inject the output of an upstream model:

```jinja
{# map/outline.prompt #}
Based on this topic, create a detailed outline:

{{ ref('topic') }}

```

### 2. Model types (`config(model_type=...)`)

Not every node in your pipeline needs to be a standard LLM call. Override `model_type` at the top of a `.prompt` file for deterministic reshaping or coding agents:

| Model Type | What it does | Declaration |
| --- | --- | --- |
| **LLM** *(default)* | Sends the rendered Jinja prompt to your configured LLM. | Standard `.prompt` file |
| **`template`** | Renders Jinja directly with no LLM call (for reshaping upstream outputs). | `{{ config(model_type="template") }}` |
| **`agent`** | Hands the prompt to [mini-swe-agent](https://mini-swe-agent.com), which can run shell commands in a scratch folder before replying. Text in, text out. Upstream `promptfiles` are copied in, and files it writes become the model's files. Pure Python, no Node needed. | `{{ config(model_type="agent") }}` |
| **`coding_agent`** | mini-swe-agent as a pure function: an upstream folder in (or empty), the patched folder plus a diff out. Unchanged inputs are reused without re-running the agent; a run cut off mid-way replays its finished steps on restart instead of paying for them again. | `{{ config(model_type="coding_agent", promptfiles="repo") }}` |

*(More model kinds, such as `execute_python`, `agent_multi_harness` (mini-swe-agent / opencode / pi), `python_modal`, `agent_modal` and `agent37`, come with `pip install "promptmap[extras]"`.)*

### 3. Tests (`map/*.testprompt`)

Run `promptmap test` to validate the latest run's outputs against your expectations:

* **LLM-as-a-judge:** Prompt tests in `map/*.testprompt` (alongside your models) pass when the LLM returns `{"results": "pass"}`.
* **Deterministic checks:** Compare output fields directly in YAML cases (`map/*.yml`) using `expect` (no LLM judge needed).

### 4. Visualizing & serving (`promptmap docs` & `promptmap serve`)

Inspect prompt graph visually:

* **`promptmap docs`** generates a self-contained HTML report of every run—showing rendered prompts, model outputs, token usage, and test results across the DAG.
* **`promptmap serve`** starts a local server to interactively explore the DAG, trigger runs or inspect previous results.

---

## Documentation

| Category | Guides |
| --- | --- |
| **Inputs & Data** | [Passing variables (`promptdata`)](docs/INTERMEDIATE_USAGE.md#passing-variables-to-templates-promptdata) · [Passing files (`promptfiles`)](docs/INTERMEDIATE_USAGE.md#passing-files-to-models-promptfiles) · [RAG inside prompts](docs/INTERMEDIATE_USAGE.md#rag-inside-prompts-ragpy) · [Global instructions](docs/ADVANCED_USAGE.md#global-instructions-globalprompt) |
| **Workflow & DAG** | [Grouping models in folders](docs/INTERMEDIATE_USAGE.md#grouping-models-in-folders) · [Map & reduce (`each=`)](docs/ADVANCED_USAGE.md#map-and-reduce-each) · [Dynamically skipping models](docs/INTERMEDIATE_USAGE.md#how-to-dynamically-skip-a-model) · [Models that produce files](docs/ADVANCED_USAGE.md#models-that-produce-files-promptmapfile-promptmapdir-promptmapoutput) · [Run IDs & iterating on the last answer (`previous`)](docs/ADVANCED_USAGE.md#run-ids-and-iterating-on-the-last-answer-previous) |
| **Testing & Eval** | [Writing tests](docs/INTERMEDIATE_USAGE.md#promptmap-tests) · [Validation](docs/INTERMEDIATE_USAGE.md#validation-validation) · [Bulk YAML testing & `expect`](docs/ADVANCED_USAGE.md#bulk-testing-with-yaml-cases-promptmap-test) |
| **CLI, UI & API** | [`promptmap docs` (HTML report)](docs/INTERMEDIATE_USAGE.md#promptmap-docs) · [`promptmap serve` & HTTP server](docs/ADVANCED_USAGE.md#promptmap-serve) · [`promptmap ls`](docs/ADVANCED_USAGE.md#promptmap-ls) · [Python API & callbacks](docs/ADVANCED_USAGE.md#python-api) · [Token usage](docs/ADVANCED_USAGE.md#token-usage-promptmapllmresult) |
| **Customization** | [Provider/model (`settings.yml`) & custom backend (`client.py`)](docs/INTERMEDIATE_USAGE.md#customising-the-llm-backend-clientpy) · [Output format & `config()`](docs/ADVANCED_USAGE.md#output-format-config-config) · [Agent models](docs/ADVANCED_USAGE.md#agent-models-model_typeagent) · [Custom model kinds](docs/ADVANCED_USAGE.md#writing-your-own-model-kind-model_type) · [Architecture](docs/ARCHITECTURE.md) |
