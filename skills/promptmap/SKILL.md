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
uv tool install promptmap
promptmap init my_project --no-interactive --provider anthropic --model claude-sonnet-4-5
promptmap run        # run the DAG (add -s <model> to run one model + upstream)
promptmap test       # run tests against the latest run
promptmap docs       # HTML report: rendered prompts, outputs, tokens, tests
promptmap serve      # local UI to explore and trigger runs
```

`init` writes `settings.yml` (provider + model) and `.env` (API key). As an
agent, always pass `--no-interactive` (the prompts need a human) plus the
flags you know; run `promptmap init --help` for the valid options. Never
paste a user's key into a command — leave it blank in `.env` for them.
One prompt can use another model: `{{ config(model="...") }}` — a model name or
an alias from settings.yml's `aliases:` block (`low: openai/gpt-4o-mini`).

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

## Folders: a file named like its folder sums it up

Folders group models; the folder is not part of the name (`2_build/parts.prompt`
is `ref('parts')`), and names stay unique across the map. A file named like
its folder, inside it (`node_1/node_1.prompt`) or next to it (`node_1.prompt`
beside `node_1/`), is that **folder model**: every model directly inside the folder is
appended to its end as `{{ ref('child') }}` unless it already refs that child.
A sub-folder counts as one child, its own folder model.

```
map/
  node_1/
    node_1.prompt        # gets research, critique, node_a appended
    research.prompt
    critique.prompt
    node_a/
      node_a.prompt      # gets fact_1, fact_2 appended
      fact_1.prompt
      fact_2.prompt
  final.prompt           # {{ ref('node_1') }} = the whole folder's result
```

Write the folder model as the instruction only ("Merge the findings below into
one recommendation"); the inputs arrive after it. Files on disk are never
edited. A child must not `ref()` its own folder model (cycle).

## Ref tags: on by default

Every `{{ ref('x') }}` an LLM sees arrives tagged so it can tell inputs apart:
`[x]\n<output>\n[/x]`, so no need for your own delimiters around refs. Text you
process stays plain: `ref('x') | upper`, `.split()`, JSON paths
(`ref('m.key')`), `each=` items, and non-LLM kinds (`template`,
`execute_python`). Off: `promptmap run --no-wrap-refs`; custom tags:
`wrap_refs=lambda name, is_open: ...` in Python.

## Model types: `prompt`, `template`, `agent`, `coding_agent`

Set with `{{ config(model_type="...") }}` at the top of the file.

| Use | When | Declaration |
|-----|------|-------------|
| **prompt** (default LLM call) | the step needs judgement or generation | plain `.prompt` file |
| **`template`** | reshaping, joining or selecting upstream outputs; no LLM needed | `{{ config(model_type="template") }}` |
| **`agent`** | the step needs tools (run code, fetch, inspect files) but the result is an answer | `{{ config(model_type="agent") }}` |
| **`chat`** *(promptmap[extras])* | a person refines the answer over several turns (follow-ups via `promptmap chat MODEL "..."` or `run(chat_input=...)`) | `{{ config(model_type="chat") }}` |
| **`coding_agent`** | the step produces or patches a file tree (code, a board, a repo) | `{{ config(model_type="coding_agent", inherit_coding_folder="repo") }}` |

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

**`agent`** runs [mini-swe-agent](https://mini-swe-agent.com) (pure Python,
LLM from `config(agent_model=...)`, else the `settings.yml` model, else `MSWEA_MODEL_NAME`) in a scratch folder: text in, text out.
`promptfiles` (upstream models' files and run-level `--promptfile name=path`) are copied to `inputs/<name>/`; files it writes to `outputs/`
become the model's files.

**`coding_agent`** runs mini-swe-agent as a pure function: a `Dir` in, the patched `Dir` out.
The tree it works on comes from upstream via `inherit_coding_folder="<model>"`;
leave it out and the agent creates the tree from scratch. `promptfiles` keeps
its ordinary meaning: reference files (a spec, a screenshot) copied into
`.promptmap_inputs/` for the agent to read, never part of `result`; changing
one re-runs the agent. Do not use `promptfiles` to pass the tree. Independent agents (neither refs the
other) run in parallel, so split unrelated work into sibling agents rather than
one long chain (`promptmap run --threads N` caps how many run at once,
default 4). Chain agents by passing one's output to
the next. It caches its change and only re-runs the agent when upstream
changed the meaning of the code.

```jinja
{# map/api.prompt — repo is an upstream model whose output is a Dir #}
{{ config(model_type="coding_agent", inherit_coding_folder="repo") }}
Add a /health endpoint to the API.

{# map/api_tests.prompt — works on api's patched tree, reads api_spec's files #}
{{ config(model_type="coding_agent", inherit_coding_folder="api", promptfiles="api_spec") }}
Add tests for every endpoint.
```

Output keys: `output` (answer), `result` (patched `Dir`), `changes`,
`deleted`, `sum_of_changes`, `cache`. Downstream: `{{ ref('api').changes }}`.

**Where the tree goes.** A `coding_agent` does not write `outputs/<model>/`
(the tree stays in the run database, stored once however many steps share it);
`write_folder=true` brings that folder back. To land the work in a real
project, put `project="../app"` on the **last** agent of the chain:

```jinja
{{ config(model_type="coding_agent", inherit_coding_folder="api", project="../app") }}
```

After every run it writes the files that differ and removes the files this
model deleted; anything else there (`.git`, `node_modules`, untracked files)
is left alone, and the output reports `project: {path, written, deleted}`.
Relative paths are from where `promptmap run` started. It overwrites without
asking, so tell the user to commit hand edits first. Files deleted by an
*earlier* agent are not removed from the project. Keep the intermediate agents
pure (no `project=`), so reuse and parallel siblings keep working.

**Tweaking a `coding_agent` prompt keeps its work.** `coding_agent` has
`warm_start` on by default. After you edit its `.prompt` file, the next run
starts from the previous change set, sees a diff of the old and new prompt and
adjusts only what the edit asks for (`cache: "warm"`), instead of
cold-starting. So refine a mostly-right agent prompt in small edits. When an
edit changes what the model is *for*, the old result would mislead it: run
`promptmap clear-cache <model>` once for a clean start, or set
`{{ config(warm_start=false) }}` to always start cold. `promptmap run` prints
the verdict after each model (`· cached`, `· warm · continued previous run`,
`· miss · agent ran`), so you can see what a prompt edit cost. To rename a
model, use `promptmap rename <old> <new>` (try `--dry-run` first), never a
plain file rename: the command also rewrites `ref()` / `promptfiles` and moves
the model's run history, so the agent's previous work is not lost. Other models can opt in
with `warm_start=true` (it fills `{{ previous }}` across edits). See
`#editing-a-prompt-without-a-cold-start-warm_start`.

With `promptmap[extras]`: `execute_python` (run rendered Python, capture
`output`) and `agent_multi_harness` (mini-swe-agent / opencode / pi in place in an
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
| Organise many models | subfolders in `map/`; `x/x.prompt` or `x.prompt` beside `x/` auto-refs the folder's models | `#grouping-models-in-folders` |
| Skip a model conditionally | `skip_and_set_to_value()` | `#how-to-dynamically-skip-a-model` |
| Change provider / model, model aliases | `settings.yml` (`aliases:`), `config(model=...)` | `#choosing-the-provider-and-model-settingsyml` |
| Fully custom LLM call | `client.py` | `#customising-the-llm-backend-clientpy` |
| Schema-check an output in Python | `validation/<model>.py` | `#validation-validation` |
| Yes/no classifier judge, gate tests | `.testprompt` options | `#promptmap-tests` |

Advanced — `docs/ADVANCED_USAGE.md`:

| Need | Feature | Section |
|------|---------|---------|
| Run a model once per item, then reduce | `each=` | `#map-and-reduce-each` |
| Replay a known input / assert fields | YAML `given`, `expect` | `#bulk-testing-with-yaml-cases-promptmap-test` |
| Models that write files or folders | `promptmap.File`, `Dir`, `Output` | `#models-that-produce-files-promptmapfile-promptmapdir-promptmapoutput` |
| Refine last run's answer | `{{ previous }}`, `previous_of('name')`, `cache_check`; for follow-ups from a person, `model_type="chat"` | `#run-ids-and-iterating-on-the-last-answer-previous` |
| Edit a `coding_agent` prompt, keep its previous run | `warm_start` (default on for `coding_agent`) | `#editing-a-prompt-without-a-cold-start-warm_start` |
| Write a `coding_agent`'s tree to a real folder / `outputs/` | `project="../app"`, `write_folder=true` | INTERMEDIATE_USAGE `#where-file-outputs-go-write_folder-project` |
| Make one model start cold next run | `promptmap clear-cache MODEL` | `#editing-a-prompt-without-a-cold-start-warm_start` |
| Rename a model, keep its history | `promptmap rename OLD NEW` | `#renaming-a-model-promptmap-rename` |
| Shared instructions for every model | `global.prompt` | `#global-instructions-globalprompt` |
| JSON output, caching and other config keys | `config()` | `#output-format-config-config` |
| Agents / MCP tools | `agent`, `coding_agent`; `agent_multi_harness` + `agent_mcp` | `#agent-models-model_typeagent` |
| Your own model type | custom model kinds | `#writing-your-own-model-kind-model_type` |
| Your own Jinja functions; tell a model where it sits (`org_outline()`) | `jinja_template_functions.py`, `@promptmap.template_function` | `#your-own-jinja-functions-jinja_template_functionspy` |
| Call from Python / live progress | `promptmap.run()` | `#python-api` |
| Token counts | `LLMResult` | `#token-usage-promptmapllmresult` |

Worked example of the whole philosophy: `examples/pcb_design/` (staged DAG,
`each=` per part, a `given` fault case in `map/tests/`).
