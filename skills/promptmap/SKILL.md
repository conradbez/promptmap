---
name: promptmap
description: Build, test and iterate on LLM pipelines with promptmap (dbt for prompts). Use when working in a project with a map/ folder of .prompt / .testprompt files, when the user mentions promptmap, ref(), promptmap run/test/docs/serve, or wants to split a big one-shot prompt into a DAG of smaller tested steps.
---

# promptmap

promptmap turns a big LLM task into a DAG of small Jinja prompts (`map/*.prompt`)
joined with `ref()`, plus tests (`map/*.testprompt`, `map/**/*.yml`) that check
each step against reality.

Docs live in the repo: https://github.com/conradbez/promptmap
(`README.md`, `docs/INTERMEDIATE_USAGE.md`, `docs/ADVANCED_USAGE.md`).

## Motivation: the map and the landmarks

An LLM is a drunken tourist following the line on the phone: it can walk 30
minutes back to the pub it started from without noticing. The fix is the
human loop: make a prediction, check it against the world, update the map.

| Idea | In promptmap |
|------|--------------|
| The map: your understanding of the problem | the DAG of models in `map/` |
| A step: change one prompt, assumption or link | edit one `.prompt`, `promptmap run` |
| A landmark: "I expect the river on my left" | a test |
| Keeping landmarks across iterations | `promptmap test` |

What this means for you as the agent:

- **Never one-shot rewrite the pipeline.** Find the node whose hypothesis is
  wrong and change that node. Keep wrong working; build off it.
- **Predict before you run.** Before `promptmap run`, say what you expect a
  model to output. A surprise is information, so act on it.
- **Turn every real mistake into a test.** When a run drifted off the map,
  fix the node, then add a test that would have caught it.
- **You are the CEO, not the factory floor.** Trust the prompts you write and
  verify them with promptmap tests. Your employees (the sub-prompts) exist for
  a reason: give them direction. Micromanaging them (reading every model's
  output, hand-checking code line by line) will not improve the outcome. A good
  CEO sets checks and balances without inspecting every widget: overall
  dashboards (`promptmap docs`/`serve`), overall status (`promptmap run`
  results), and audits (`promptmap test`). If something is wrong, an audit
  should catch it; if none does, write the audit.

## Workflow

```bash
uv tool install promptmap --with anthropic   # or openai | gemini
promptmap init --provider anthropic
promptmap run        # run the DAG (add -s <model> to run one model + upstream)
promptmap test       # run tests against the latest run
promptmap docs       # HTML report: rendered prompts, outputs, tokens, tests
promptmap serve      # local UI to explore and trigger runs
```

Loop: edit one node → `promptmap run -s <model>` → read the output (or
`promptmap docs`) → `promptmap test` → repeat.

## Writing prompts: outcome-focused, one screen max

Each `.prompt` is one hypothesis with one job. Rules:

1. **Fits on one laptop screen (~40 lines).** Longer means it does two jobs.
   Split it into two models and `ref()` one from the other.
2. **Lead with the outcome**, not the procedure: what the output is for and
   who reads it next. "Return the 5 parts whose polarity can be wrong, as
   JSON" beats "Think step by step about the parts…".
3. **State the output shape** at the end. Use `output_format="json"` when a
   downstream model or test reads fields.
4. **Say what to do with missing evidence** ("write MISSING", "flag it") so
   gaps show up instead of being papered over.
5. **Put the hard-won lesson in the prompt** when a node exists because of a
   past failure (one sentence of why).

```jinja
{# map/outline.prompt — one job: an outline the article writer can follow #}
{{ config(output_format="json") }}
Write an outline for an article on {{ promptdata('topic') }} for
{{ promptdata('audience') }}. The next step writes one section per heading.

Background research:
{{ ref('research') }}

Return only JSON: {"headings": ["..."], "key_claim": "..."}
```

## Model types: `prompt`, `template`, `agent`, `coding_agent`

Set with `{{ config(model_type="...") }}` at the top of the file.

| Use | When | Declaration |
|-----|------|-------------|
| **prompt** (default LLM call) | the step needs judgement or generation | plain `.prompt` file |
| **`template`** | reshaping, joining or selecting upstream outputs; no LLM needed | `{{ config(model_type="template") }}` |
| **`agent`** | the step needs tools (run code, fetch, inspect files) but the result is an answer | `{{ config(model_type="agent") }}` |
| **`coding_agent`** | the step produces or patches a file tree (code, a board, a repo) | `{{ config(model_type="coding_agent", promptfiles="repo") }}` |

Prefer the cheapest type that works: `template` before prompt, prompt before
agent. Deterministic steps should never cost an LLM call.

```jinja
{# map/brief_for_writer.prompt — no LLM, just reshape #}
{{ config(model_type="template") }}
Topic: {{ promptdata('topic') }}
Headings:
{% for h in ref('outline').headings %}- {{ h }}
{% endfor %}
```

**`agent`** runs the [pi](https://pi.dev) agent (`npm i -g
@mariozechner/pi-coding-agent`) in a scratch folder: text in, text out.
Upstream `promptfiles` are copied to `inputs/`; files it writes to `outputs/`
become the model's files.

**`coding_agent`** runs pi as a pure function: a `Dir` in, the patched `Dir` out.
The input tree comes from upstream via `promptfiles`; leave it out and the
agent creates the tree from scratch. Chain agents by passing one's output to
the next. It caches its change and only re-runs the agent when upstream
changed the meaning of the code.

```jinja
{# map/api.prompt — repo is an upstream model whose output is a Dir #}
{{ config(model_type="coding_agent", promptfiles="repo") }}
Add a /health endpoint to the API.

{# map/api_tests.prompt — works on api's patched tree #}
{{ config(model_type="coding_agent", promptfiles="api") }}
Add tests for every endpoint.
```

Output keys: `output` (answer), `result` (patched `Dir`), `changes`,
`deleted`, `sum_of_changes`, `cache`. Downstream: `{{ ref('api').changes }}`.

With `promptmap[extras]`: `execute_python` (run rendered Python, capture
`output`) and `agent_multi_harness` (mini-swe-agent / opencode in place in an
`agent_dir`). See
`docs/ADVANCED_USAGE.md#agent-models-model_typeagent`.

## Writing tests: test the process, not the product

A test is a landmark: a prediction that must hold on **any** input. Test that
each stage is honest about its evidence, not that the final artefact is good.

- Bad: "Is the board safe to ship?" Fails on a known-fault input even when QA
  correctly caught the fault, so a catch looks like a regression.
- Good: "Given the pads the board actually has, did the polarity check's
  verdict match?" Passes on clean and faulty inputs alike; a failure names the
  stage that drifted.

Have the judge work out the answer from the evidence first, then compare with
the stage's verdict.

**LLM-judged test** — `map/**/*.testprompt`, passes on `{"results": "pass"}`:

```jinja
{# map/tests/outline_matches_research.testprompt #}
Research: {{ ref('research') }}
Outline key claim: {{ ref('outline').key_claim }}

First, list the claims the research supports. Then: is the key claim one of them?
Reply with only JSON: {"results": "pass"} or {"results": "fail"}.
```

Tests are **gates** by default: `promptmap run` stops before downstream models
if one fails. Use `{{ config(gate=False) }}` for subjective checks.

**YAML cases** — `map/**/*.yml`: run the pipeline over named inputs, pin a
model's output with `given` (replay a real failure), assert fields with
`expect` (no LLM judge):

```yaml
# map/tests/reversed_flyback_diode.yml
baselines:
  default:
    promptdata: {brief: 12 V motor driver}
cases:
  - name: Reversed flyback diode
    given:
      board: {output: {pad_nets: [{ref: D1, pads: [{pad: "1", net: MOTOR_DRAIN}]}]}}
    expect:
      polarity:
        - {ref: D1, built_pads_agree: false}   # QA must catch it
```

Recipe for every real mistake: (1) save the bad upstream output as `given`,
(2) `expect` the stage that should catch it to flag it, (3) fix that stage's
prompt until `promptmap test --case "<name>"` passes.

## Going further: which doc to read

Intermediate — `docs/INTERMEDIATE_USAGE.md`:

| Need | Feature | Section |
|------|---------|---------|
| Pass runtime inputs (topic, audience) | `promptdata('key')`, `--promptdata k=v` | `#passing-variables-to-templates-promptdata` |
| Attach PDFs/images to a model | `promptfiles` | `#passing-files-to-models-promptfiles` |
| Pull search/RAG results into a prompt | `rag.py`, `rag_*()` kinds | `#rag-inside-prompts-ragpy` |
| Organise many models | subfolders in `map/` | `#grouping-models-in-folders` |
| Skip a model conditionally | `skip_and_set_to_value()` | `#how-to-dynamically-skip-a-model` |
| Swap LLM provider / custom call | `client.py` | `#customising-the-llm-backend-clientpy` |
| Schema-check an output in Python | `validation/<model>.py` | `#validation-validation` |
| Yes/no classifier judge, gate tests | `.testprompt` options | `#promptmap-tests` |

Advanced — `docs/ADVANCED_USAGE.md`:

| Need | Feature | Section |
|------|---------|---------|
| Run a model once per item, then reduce | `each=` | `#map-and-reduce-each` |
| Replay a known input / assert fields | YAML `given`, `expect` | `#bulk-testing-with-yaml-cases-promptmap-test` |
| Models that write files or folders | `promptmap.File`, `Dir`, `Output` | `#models-that-produce-files-promptmapfile-promptmapdir-promptmapoutput` |
| Refine last run's answer | `{{ previous }}`, `cache_check` | `#run-ids-and-iterating-on-the-last-answer-previous` |
| Shared instructions for every model | `global.prompt` | `#global-instructions-globalprompt` |
| JSON output, caching and other config keys | `config()` | `#output-format-config-config` |
| Agents / MCP tools | `agent`, `coding_agent`; `agent_multi_harness` + `agent_mcp` | `#agent-models-model_typeagent` |
| Your own model type | custom model kinds | `#writing-your-own-model-kind-model_type` |
| Call from Python / live progress | `promptmap.run()` | `#python-api` |
| Token counts | `LLMResult` | `#token-usage-promptmapllmresult` |

Worked example of the whole philosophy: `examples/pcb_design/` (staged DAG,
`each=` per part, a `given` fault case in `map/tests/`).
