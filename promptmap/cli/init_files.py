"""
promptmap/cli/init_files.py

Scaffold templates and the `promptmap init` Click command.
"""

from pathlib import Path

from promptmap.settings import SETTINGS_FILE

# ---------------------------------------------------------------------------
# Scaffold file templates
# ---------------------------------------------------------------------------

INIT_FILES = {
    "README.md": """\
# Getting Started

| File / Directory | Purpose |
|-----------------|---------|
| `settings.yml` | Which provider + model to call (edit freely) |
| `map/` | START HERE: Prompt files (`.prompt`) and LLM-as-judge tests (`.testprompt`) |
| `validation/` | Pre-pass quality gates |
| `outputs/` | Generated outputs from `promptmap run` (auto-created) |

Run: `promptmap run` or `promptmap run --promptdata topic="your topic"`

---

## map/

Write your prompts here. Each `.prompt` file defines one step in your pipeline.

You can:
- Reference other prompt outputs:  `{{ ref('other_prompt_name') }}`
- Access passed-in data:           `{{ promptdata("key") }}`
- Configure model behaviour:       `{{ config(output_format="json") }}`

Example chain — `map/topic.prompt` → `map/article.prompt` → `map/summary.prompt`:

    # topic.prompt
    Generate a catchy blog post topic about AI.

    # article.prompt
    Write a detailed article about: {{ ref('topic') }}

    # summary.prompt
    {{ config(output_format="json") }}
    Summarise this article in 3 bullet points. Return JSON: {"bullets": ["...", "...", "..."]}.
    {{ ref('article') }}

---

## Tests (`map/*.testprompt`)

Used when you change prompts to make sure everything still works and passes quality standards.

Each `.testprompt` file in `map/` (next to the models it checks) is an LLM-as-judge test.
The prompt should reference model outputs via `{{ ref('model_name') }}` and return JSON:
  - Pass: `{"results": "pass"}`
  - Fail: `{"results": "fail"}`

Tests are also gates: `promptmap run` judges each one right after the models it
reads, and stops the run if it fails. Add `{{ config(gate=False) }}` to a
test that should only run under `promptmap test`.

Example `map/summary_has_bullets.testprompt`:

    Does the following text contain at least 3 bullet points (lines starting with - or •)?

    {{ ref('summary') }}

    Reply with only valid JSON: {"results": "pass"} or {"results": "fail"}.

---

## validation/

Optional Python code that post-processes each model's LLM output before it is
stored and passed to downstream models via `ref()`.

Each `.py` file must expose a `validate(prompt: str, result: str) -> Any` function.
- Return `False` (or raise) to stop the pipeline and report a validation error.
- Return any other value to use it as the model's output — this replaces the raw
  LLM text for both storage and downstream `ref()` calls.

Example `validation/article.py` — fail short outputs, pass the rest unchanged:

    def validate(prompt: str, result: str) -> Any:
        \"\"\"Article must be at least 200 characters and contain a markdown header.\"\"\"
        if len(result) < 200 or "#" not in result:
            return False
        return result

Example with post-processing — parse and return a cleaned dict:

    import json

    def validate(prompt: str, result: str) -> Any:
        data = json.loads(result)          # raises → model fails
        data.pop("debug_info", None)       # strip internal keys
        return data                        # dict replaces raw JSON string downstream

---

## settings.yml

Which LLM `promptmap run` calls, through [LiteLLM](https://github.com/BerriAI/litellm):

    provider: anthropic
    model: claude-sonnet-4-5

Set it with `promptmap init --provider <name> --model <name>`, or edit it by hand —
values are not validated. API keys come from the environment or `.env`
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, ...); `promptmap init` adds the
right one to `.env`.

For full control over the call (retries, system prompts, a custom API), add a
`client.py` with `llm_call(prompt: str) -> str`; it replaces settings.yml.
See docs/INTERMEDIATE_USAGE.md.
""",
    "map/articles.prompt": """\
{{ config(output_format="json") }}
{# Generate an article on a topic. Pass --promptdata topic=... or leave blank for a generic article. #}
{% if promptdata("topic") %}
Write a detailed, engaging article about: {{ promptdata("topic") }}
{% else %}
Write a detailed, engaging article about a fascinating topic of your choice.
{% endif %}

Structure it with an introduction, 3-4 body sections, and a conclusion.
Use markdown formatting with headers.

Return only valid JSON in this exact format:
{
  "content": "<full article text in markdown>",
  "author": "<a plausible author name>",
  "audience": "<intended audience, e.g. 'general public', 'developers', 'students'>"
}
""",
    "map/summaries.prompt": """\
{{ config(output_format="json") }}
You are given a JSON article object. Produce a concise summary.

Article:
{{ ref('articles') }}

Return only valid JSON in this exact format:
{
  "summaries": [
    {
      "title": "<short title for the summary>",
      "summary": "<2-3 sentence summary of the article>",
      "key_points": ["<point 1>", "<point 2>", "<point 3>"]
    }
  ]
}
""",
    "map/README.md": """\
# promptmap models (The Map)

Rather than treating prompts as isolated pass/fail one-shots — or saving an
unorganized mess of every prompt ever tried — promptmap models structures prompts as
a Directed Acyclic Graph (DAG) of lower- and higher-order hypotheses. Like
keeping failed working in physics, it externalizes your mental map so you can
tweak underlying assumptions rather than doing blind, one-shot rewrites.

## Tests (The Landmarks)

`*.testprompt` files live here too. Tests preserve hard-earned domain lessons
across iterations (such as catching a reversed diode where logical pin names
matched, but physical pad connections were backward). They act as smoke tests
to verify that a change actually moved the project toward the destination.
""",
    "map/summary_has_bullets.testprompt": """\
{# gate=False: summaries is the last model; nothing downstream for a gate to protect #}
{{ config(gate=False) }}
Does the following JSON summaries object contain at least 3 key points in the first summary entry?

{{ ref('summaries') }}

Reply with only valid JSON: {"results": "pass"} or {"results": "fail"}.
""",
}

# ---------------------------------------------------------------------------
# CLI command — registered onto the main Click group from cli/__init__.py
# ---------------------------------------------------------------------------

def register_command(main) -> None:
    """Attach the `promptmap init` command to *main* Click group."""
    import click
    from rich.console import Console
    from promptmap.cli.vscode import is_running_in_vscode, setup_vscode_associations

    console = Console()

    @main.command("init")
    @click.argument("project_name", required=False)
    @click.option("--force", is_flag=True, default=False, help="Overwrite existing files.")
    @click.option(
        "--secrets", "with_secrets", is_flag=True, default=False,
        help="Also write secrets.yml + secrets-template.yml and gitignore secrets.yml.",
    )
    @click.option("--data", "with_data", is_flag=True, default=False, help="Also write promptdata.yml.")
    @click.option("--files", "with_files", is_flag=True, default=False, help="Also write promptfiles.yml.")
    @click.option(
        "--map-dir",
        default="map",
        show_default=True,
        help="Models whose keys --secrets/--data/--files write (relative to the project).",
    )
    @click.option(
        "--provider", default=None,
        help="LiteLLM provider (e.g. anthropic, openai, gemini). Asked interactively if omitted.",
    )
    @click.option("--model", default=None, help="Model name. Asked interactively if omitted.")
    @click.option(
        "--api-key", default=None,
        help="Value for the provider's API key variable (e.g. GROQ_API_KEY), written to .env.",
    )
    @click.option(
        "--env", "env_flags", multiple=True, metavar="NAME=VALUE",
        help="Write NAME=VALUE to .env (e.g. AZURE_API_BASE=...). Repeatable.",
    )
    @click.option(
        "--no-interactive", "--nointeractive", "no_interactive", is_flag=True, default=False,
        help="Ask nothing: anything not given by a flag is left blank in settings.yml / .env.",
    )
    def init(
        project_name: str | None, force: bool, with_secrets: bool, with_data: bool,
        with_files: bool, map_dir: str, provider: str | None, model: str | None,
        api_key: str | None, env_flags: tuple[str, ...], no_interactive: bool,
    ) -> None:
        """Scaffold a starter promptmap project inside PROJECT_NAME/.

        Writes settings.yml with the provider and model `promptmap run` calls
        through LiteLLM, and the API key variables it needs to .env.  Whatever
        --provider / --model / --api-key / --env leave out is asked for;
        Enter at a prompt leaves it blank.  --no-interactive asks nothing and
        leaves it blank.

        --secrets, --data and --files also write secrets.yml (+ secrets-template.yml),
        promptdata.yml and promptfiles.yml, listing the keys the project's
        models use.  Without PROJECT_NAME they are written into the current
        directory instead of scaffolding a project: `promptmap init --secrets`.
        Existing values are kept; missing keys are appended.

        \b
        Examples:
          promptmap init my_project --provider anthropic --model claude-sonnet-4-5
          promptmap init my_project          (asks for provider, model, API key)
          promptmap init my_project --provider groq --model llama-3.3-70b-versatile --api-key gsk_...
          promptmap init my_project --no-interactive   (blank settings.yml + .env to edit)
          promptmap init --provider openai   (in an existing project: rewrite settings.yml)
          promptmap init my_project --secrets --data
          promptmap init --secrets          (in an existing project)
        """
        from promptmap import settings as _settings

        env_values = parse_env_flags(env_flags)
        interactive = not no_interactive and _settings.is_interactive()
        llm_flags = provider is not None or model is not None or api_key is not None or bool(env_values)
        llm_kwargs = dict(interactive=interactive, api_key=api_key, env_values=env_values)
        kinds = [k for k, on in (("secrets", with_secrets), ("data", with_data), ("files", with_files)) if on]
        if project_name is None and (kinds or llm_flags):
            for kind in kinds:
                init_values_file(console, kind, Path("."), Path(map_dir), force)
            if llm_flags:
                write_settings(console, Path("."), provider, model, **llm_kwargs)
            return
        project_name = project_name or "generate_articles_example"
        files = dict(INIT_FILES)
        files["validation/articles.py"] = """\
# Validation files let you post-process and gate LLM outputs before they flow downstream.
# - Name this file after the prompt it validates (e.g. articles.py validates articles.prompt).
# - Return False (or raise) to fail the pipeline; return any value to replace the raw LLM text.
# - Returned values are stored and used by ref() in downstream prompts — parse, clean, or reshape freely.

def validate(prompt: str, result: str):
    \"\"\"Passthrough — replace this with your own logic.\"\"\"
    return result


# Example: parse and validate a JSON article with Pydantic
#
# import json
# from pydantic import BaseModel, ValidationError
#
# class Article(BaseModel):
#     content: str
#     author: str
#     audience: str
#
# def validate(prompt: str, result: str) -> Article | bool:
#     try:
#         data = json.loads(result)
#         article = Article(**data)
#     except (json.JSONDecodeError, ValidationError):
#         return False
#     if len(article.content) < 200:
#         return False
#     return article.model_dump()
"""

        root = Path(project_name)
        created, skipped = [], []

        for rel_path, content in files.items():
            path = root / rel_path
            if path.exists() and not force:
                skipped.append(str(path))
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            created.append(str(path))

        for f in created:
            console.print(f"  [green]created[/green]  {f}")
        for f in skipped:
            console.print(f"  [dim]skipped[/dim]  {f}  [dim](use --force to overwrite)[/dim]")

        for kind in kinds:
            init_values_file(console, kind, root, root / map_dir, force)

        if force or not (root / SETTINGS_FILE).exists() or llm_flags:
            write_settings(console, root, provider, model, **llm_kwargs)
        else:
            console.print(f"  [dim]skipped[/dim]  {root / SETTINGS_FILE}  [dim](use --provider/--model to change)[/dim]")

        if created:
            console.print(f"\nRun [bold cyan]cd {project_name} && promptmap run[/bold cyan], or [bold cyan]promptmap run --promptdata topic='your topic'[/bold cyan]")

        if is_running_in_vscode():
            setup_vscode_associations()
            console.print("  [dim]VS Code detected — configured .vscode/settings.json for .prompt files[/dim]")


# ---------------------------------------------------------------------------
# settings.yml — provider + model for the default LiteLLM backend
# ---------------------------------------------------------------------------

def parse_env_flags(pairs) -> dict[str, str]:
    """--env NAME=VALUE flags -> dict; exits on a malformed one."""
    import sys

    values = {}
    for pair in pairs:
        name, sep, value = pair.partition("=")
        if not sep or not name.strip():
            print(f"Error: --env must be NAME=VALUE, got: {pair!r}", file=sys.stderr)
            sys.exit(1)
        values[name.strip()] = value
    return values


def write_settings(console, root: Path, provider: str | None, model: str | None, *,
                   interactive: bool, api_key: str | None = None,
                   env_values: dict[str, str] | None = None) -> None:
    """Write *root*/settings.yml and .env, asking (if *interactive*) for what flags left out.

    --provider must be a LiteLLM provider; --model is taken as given (the
    list of known models is a menu, not a whitelist).
    """
    import sys

    from rich.markup import escape

    from promptmap.settings import setup_llm

    try:
        notes = setup_llm(root, provider=provider, model=model, interactive=interactive,
                          api_key=api_key, env_values=env_values)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {escape(str(exc))}")
        sys.exit(1)
    for note in notes:
        console.print(f"  [green]wrote[/green]  {escape(note)}")


# ---------------------------------------------------------------------------
# promptmap init --secrets | --data | --files — project-root values files
# ---------------------------------------------------------------------------

_VALUES_HEADERS = {
    "secrets": (
        "Secrets for promptmap run / test / serve — read as call.secrets[\"NAME\"].\n"
        "Never rendered, cached or stored.  Keep this file out of git;\n"
        "commit secrets-template.yml instead.  --secret NAME=VALUE overrides."
    ),
    "template": (
        "Template for secrets.yml (which is gitignored): copy it and fill in values.\n"
        "  cp secrets-template.yml secrets.yml"
    ),
    "data": (
        "Default promptdata() values for promptmap run.\n"
        "--promptdata KEY=VALUE overrides any key here."
    ),
    "files": (
        "Default promptfiles for promptmap run: NAME: path (relative to the project root).\n"
        "--promptfile NAME=PATH overrides any key here."
    ),
}


def _values_keys(kind: str, map_dir: Path) -> list[str]:
    """Keys the project's models use for *kind*; empty if there are no models."""
    from promptmap.executor.graph import get_dag_promptdata, get_dag_promptfiles, load_models
    from promptmap.llm import try_load_client_module
    from promptmap.model_types import get_model_kind

    if not map_dir.is_dir():
        return []
    try_load_client_module(str(map_dir))  # registers project-local kinds and their secrets
    models = load_models(str(map_dir))
    if kind == "data":
        return get_dag_promptdata(models)
    if kind == "files":
        return get_dag_promptfiles(models)
    keys: dict[str, None] = {}
    for spec in models.values():
        model_kind = get_model_kind(spec.model_type)
        for name in sorted(model_kind.required_secrets if model_kind is not None else ()):
            keys[name] = None
    return list(keys)


def _write_values_file(path: Path, keys: list[str], header: str, force: bool) -> tuple[str, list[str]]:
    """Write *keys* to *path*; keep an existing file's values and append missing keys.

    Returns (action, keys added).
    """
    import yaml

    from promptmap.project_values import render_values_file

    if not path.exists() or force:
        action = "overwrote" if path.exists() else "created"
        path.write_text(render_values_file(keys, header), encoding="utf-8")
        return action, keys
    existing = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    present = {str(k) for k in existing} if isinstance(existing, dict) else set()
    missing = [k for k in keys if k not in present]
    if missing:
        text = path.read_text(encoding="utf-8")
        body = render_values_file(missing, "")
        path.write_text(text + ("" if text.endswith("\n") or not text else "\n") + body, encoding="utf-8")
        return "updated", missing
    return "unchanged", []


def _ensure_gitignored(root: Path, name: str) -> bool:
    """Add *name* to *root*/.gitignore unless already listed. Returns True if added."""
    path = root / ".gitignore"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if name in {line.strip().lstrip("/") for line in text.splitlines()}:
        return False
    sep = "" if not text or text.endswith("\n") else "\n"
    path.write_text(f"{text}{sep}\n# promptmap secrets — never commit\n{name}\n", encoding="utf-8")
    return True


def init_values_file(console, kind: str, root: Path, map_dir: Path, force: bool) -> None:
    """Write *root*'s values file for *kind* (`promptmap init --secrets|--data|--files`)."""
    import sys

    from rich.markup import escape

    from promptmap.project_values import (
        PROMPTDATA_FILE,
        PROMPTFILES_FILE,
        SECRETS_FILE,
        SECRETS_TEMPLATE_FILE,
    )

    try:
        keys = _values_keys(kind, map_dir)
    except Exception as exc:  # noqa: BLE001 — a broken model shouldn't hide the cause
        console.print(f"[red]Error:[/red] could not read models in {map_dir}: {escape(str(exc))}")
        sys.exit(1)

    if kind == "secrets":
        targets = [(SECRETS_TEMPLATE_FILE, "template"), (SECRETS_FILE, "secrets")]
    else:
        targets = [(PROMPTDATA_FILE if kind == "data" else PROMPTFILES_FILE, kind)]

    for filename, header in targets:
        path = root / filename
        action, added = _write_values_file(path, keys, _VALUES_HEADERS[header], force)
        detail = f" ({', '.join(added)})" if added else ""
        style = "dim" if action == "unchanged" else "green"
        console.print(f"  [{style}]{action}[/{style}]  {escape(str(path))}{escape(detail)}")

    if kind == "secrets" and _ensure_gitignored(root, SECRETS_FILE):
        console.print(f"  [green]updated[/green]  {escape(str(root / '.gitignore'))} ({SECRETS_FILE})")
    if not keys:
        console.print(f"  [dim]No {kind} keys found in {map_dir}/ — add name: value lines by hand.[/dim]")
