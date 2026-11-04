"""
promptmap.cli.pretty_print
--------------------
Rich-based display helpers for the promptmap CLI.

All console instances, table builders, and progress-callback factories live
here so that the main CLI module stays focused on orchestration logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.control import Control
from rich.segment import ControlType
from rich.markup import escape as _escape
from rich.table import Table
from rich import box

from promptmap.executor.executor import ModelRunResult
from promptmap.tester import TestResult


def format_duration(ms: float | int | None) -> str:
    """A duration in one unit, rounded to a whole number: ``850 ms``, ``12 s``, ``3 min``."""
    ms = max(float(ms or 0), 0.0)

    def whole(x: float) -> int:  # half up: 2.5 -> 3 (round() would give 2)
        return int(x + 0.5)

    if whole(ms) < 1000:
        return f"{whole(ms)} ms"
    seconds = whole(ms / 1000)
    if seconds < 60:
        return f"{seconds} s"
    return f"{whole(ms / 60_000)} min"

# ---------------------------------------------------------------------------
# Shared consoles
# ---------------------------------------------------------------------------

console = Console()
err_console = Console(stderr=True, style="bold red")


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_ts(ts: str | None) -> str:
    if not ts:
        return "—"
    return str(ts)[:19].replace("T", " ")


# ---------------------------------------------------------------------------
# promptmap run — header, progress callbacks, summary
# ---------------------------------------------------------------------------

def print_run_header(
    c: Console,
    run_id: str,
    ordered: list,
    select: tuple[str, ...],
    git_sha: str | None,
) -> None:
    c.rule("[bold cyan]promptmap run[/bold cyan]")
    c.print(f"  Run ID   : [dim]{run_id}[/dim]")
    c.print(f"  Models   : {len(ordered)}", end="")
    if select:
        c.print(f"  [dim](select: {sorted(select)})[/dim]")
    else:
        c.print()
    if git_sha:
        c.print(f"  Git SHA  : [dim]{git_sha}[/dim]")
    c.print()


def step_bar(step: int, total: int, width: int = 20) -> str:
    """One square per step: grey until done, then solid.

    With ``agent_step_limit=5`` after 2 steps: ``■ ■ ■ ■ ■ 2/5`` (first two
    solid, the rest grey).  Limits over *width* are scaled to *width* squares.
    With no limit, a solid square per step so far (capped at *width*).
    """
    if total <= 0:
        return f"{_squares(min(step, width), 0)} [dim]{step} steps[/dim]"
    blocks = min(total, width)
    filled = min(blocks, round(step * blocks / total))
    return f"{_squares(filled, blocks - filled)} [dim]{step}/{total}[/dim]"


def _squares(done: int, todo: int) -> str:
    return " ".join(["[bold]■[/bold]"] * done + ["[grey50]■[/grey50]"] * todo)


#: How each cache label reads in `promptmap run`: (short label, colour).
_CACHE_LABELS = {
    "cache": ("cached", "cyan"),
    "cache_check": ("reused · cache_check", "cyan"),
    "hit": ("hit · change re-applied", "cyan"),
    "replay": ("replay · re-applied + verified", "cyan"),
    "merged": ("merged · re-applied + merged", "cyan"),
    "warm": ("warm · continued previous run", "magenta"),
    "miss": ("miss · agent ran", "yellow"),
}


def cache_label(cache: str) -> str:
    """Rich markup for a :attr:`ModelRunResult.cache` value; "" when it just ran."""
    if not cache:
        return ""
    parts = []
    for verdict in cache.split(", "):
        text, colour = _CACHE_LABELS.get(verdict, (verdict, "cyan"))
        parts.append(f"[{colour}]{text}[/{colour}]")
    return " [dim]·[/dim] " + ", ".join(parts)


def make_run_callbacks(
    c: Console,
    results: list[ModelRunResult],
    total: int,
) -> tuple[Callable[[str], None], Callable[[ModelRunResult], None]]:
    """Return (on_start, on_done) callbacks for execute_run.

    Also installs a :data:`~promptmap.model_types.progress_listener` so a model
    that reports steps (an ``agent``) redraws its status line with a step bar.
    On a non-terminal the bar is drawn once, when the model finishes.
    """
    from promptmap.model_types import progress_listener

    line = {"prefix": "", "bar": ""}

    def on_start(name: str) -> None:
        idx = len(results) + 1
        line["prefix"] = f"  [{idx}/{total}] [bold]{name}[/bold] … "
        line["bar"] = ""
        c.print(line["prefix"], end="")

    def on_progress(name: str, step: int, limit: int) -> None:
        line["bar"] = step_bar(step, limit)
        if c.is_terminal:
            c.control(Control.move_to_column(0), Control((ControlType.ERASE_IN_LINE, 2)))
            c.print(f"{line['prefix']}{line['bar']} ", end="")

    def on_done(result: ModelRunResult) -> None:
        results.append(result)
        if line["bar"] and not c.is_terminal:
            c.print(f"{line['bar']} ", end="")
        line["bar"] = ""
        if result.status == "success":
            c.print(
                f"[green]OK[/green] [dim]({format_duration(result.execution_ms)})[/dim]"
                f"{cache_label(getattr(result, 'cache', ''))}"
            )
        elif result.status == "skipped":
            c.print("[yellow]SKIPPED[/yellow]")
        else:
            c.print("[red]ERROR[/red]")
            c.print(f"    [dim]{result.error}[/dim]")

    progress_listener.set(on_progress)
    return on_start, on_done


def print_run_summary(
    c: Console,
    all_results: list[ModelRunResult],
    outputs_dir: Path,
    written: list[str],
    run_id: str,
) -> None:
    successes = sum(1 for r in all_results if r.status == "success")
    errors    = sum(1 for r in all_results if r.status == "error")
    skipped   = sum(1 for r in all_results if r.status == "skipped")

    c.print()
    c.rule()

    summary = Table(box=box.SIMPLE, show_header=False)
    summary.add_row("Done    :", f"[green]{successes}[/green] succeeded")
    if errors:
        summary.add_row("        :", f"[red]{errors}[/red] errored")
    if skipped:
        summary.add_row("        :", f"[yellow]{skipped}[/yellow] skipped")
    cache_counts: dict[str, int] = {}
    for r in all_results:
        if r.status == "success":
            for verdict in (getattr(r, "cache", "") or "ran").split(", "):
                cache_counts[verdict] = cache_counts.get(verdict, 0) + 1
    if set(cache_counts) - {"ran"}:
        order = ["ran", "cache", "cache_check", "hit", "replay", "merged", "warm", "miss"]
        ranked = sorted(cache_counts, key=lambda v: (order.index(v) if v in order else len(order), v))
        summary.add_row("Cache   :", " · ".join(f"{cache_counts[v]} {v}" for v in ranked))
    used = [r.spent_tokens for r in all_results if r.spent_tokens is not None]
    saved = [r.cache_spent_tokens for r in all_results if r.cache_spent_tokens is not None]
    if used or saved:
        summary.add_row(
            "Tokens  :",
            f"{sum(used):,} used [dim]· {sum(saved):,} from cache[/dim]",
        )
    if written:
        summary.add_row("Outputs :", f"[dim]{outputs_dir}/[/dim]  {', '.join(written)}")
    summary.add_row("Run ID  :", f"[dim]{run_id}[/dim]")

    from promptmap import db  # local import to avoid circular at module level
    summary.add_row("DB      :", f"[dim]{db.db_path()}[/dim]")
    c.print(summary)


def print_gates_header(c: Console, gates: dict, first: list) -> None:
    c.print(
        f"  Gates    : {', '.join(sorted(gates))} "
        f"[dim](run first, after {len(first)} upstream model{'s' if len(first) != 1 else ''})[/dim]"
    )
    c.print()


def print_gates_done(c: Console, gated, remaining: int) -> None:
    if gated.gates_passed and not gated.blocked:
        rest = f" — running the remaining {remaining} model{'s' if remaining != 1 else ''}" if remaining else ""
        c.print()
        c.print(f"  [green]Gates passed[/green]{rest}")
        c.print()


def print_gate_failure(c: Console, gated) -> None:
    failed = [r.test_name for r in gated.gate_results if r.status != "pass"]
    reason = f"gate failed: {', '.join(failed)}" if failed else "a model the gates read errored"
    c.print(f"  [red]Pipeline stopped[/red] — {_escape(reason)}")
    if gated.blocked:
        c.print(f"  [dim]Not run: {', '.join(sorted(gated.blocked))}[/dim]")


# ---------------------------------------------------------------------------
# promptmap test — header, progress callbacks, summary
# ---------------------------------------------------------------------------

def print_test_header(
    c: Console,
    map_dir: str,
    tests: dict,
    target_run: dict,
) -> None:
    c.rule("[bold cyan]promptmap test[/bold cyan]")
    c.print(f"  Map dir    : [dim]{map_dir}[/dim]")
    c.print(f"  Tests      : {len(tests)}")
    c.print(f"  Using run  : [dim]{target_run['run_id']}[/dim]  ({target_run['run_date']})")
    c.print()


def make_test_callbacks(
    c: Console,
    test_results: list[TestResult],
    total: int,
) -> tuple[Callable[[str], None], Callable[[TestResult], None]]:
    """Return (on_start, on_done) callbacks for execute_tests."""

    def on_start(name: str) -> None:
        idx = len(test_results) + 1
        c.print(f"  [{idx}/{total}] [bold]{_escape(name)}[/bold] … ", end="")

    def on_done(result: TestResult) -> None:
        test_results.append(result)
        score = f"P(yes)={result.score:.2f}, " if result.score is not None else ""
        if result.status == "pass":
            c.print(f"[green]PASS[/green] [dim]({score}{format_duration(result.execution_ms)})[/dim]")
        elif result.status == "fail" and result.score is not None:
            c.print(f"[red]FAIL[/red] [dim]({score}{format_duration(result.execution_ms)})[/dim]")
        elif result.status == "fail":
            c.print("[red]FAIL[/red]")
            if result.judge == "assertion":
                c.print(f"    {_escape(result.error)}")
            else:
                c.print(f"    LLM returned: [dim]{result.llm_output!r}[/dim]")
        else:
            c.print("[red]ERROR[/red]")
            c.print(f"    [dim]{result.error}[/dim]")

    return on_start, on_done


def print_test_summary(
    c: Console,
    test_results: list[TestResult],
    target_run: dict,
) -> None:
    passed  = sum(1 for r in test_results if r.status == "pass")
    failed  = sum(1 for r in test_results if r.status == "fail")
    errored = sum(1 for r in test_results if r.status == "error")

    c.print()
    c.rule()

    summary = Table(box=box.SIMPLE, show_header=False)
    summary.add_row("Passed  :", f"[green]{passed}[/green]")
    if failed:
        summary.add_row("Failed  :", f"[red]{failed}[/red]")
    if errored:
        summary.add_row("Errors  :", f"[red]{errored}[/red]")
    summary.add_row("Run ID  :", f"[dim]{target_run['run_id']}[/dim]")
    c.print(summary)


# ---------------------------------------------------------------------------
# promptmap ls — models table
# ---------------------------------------------------------------------------

def models_table(ordered: list) -> Table:
    table = Table(
        title="Prompt Models",
        box=box.ROUNDED,
    )
    table.add_column("#", style="dim", justify="right")
    table.add_column("Model", style="bold cyan")
    table.add_column("Depends on")
    table.add_column("promptdata() used", style="dim")
    table.add_column("File", style="dim")

    for i, model in enumerate(ordered, 1):
        deps = ", ".join(model.depends_on) if model.depends_on else "[dim]—[/dim]"
        promptdata_str = ", ".join(model.promptdata_used) if model.promptdata_used else "[dim]—[/dim]"
        table.add_row(str(i), model.name, deps, promptdata_str, str(model.path))

    return table


# ---------------------------------------------------------------------------
# promptmap show-runs — runs table
# ---------------------------------------------------------------------------

_STATUS_STYLES = {
    "success": "green",
    "error": "red",
    "partial": "yellow",
    "running": "cyan",
}


def runs_table(rows: list) -> Table:
    table = Table(title="Recent Runs", box=box.ROUNDED)
    table.add_column("Run ID", style="dim", no_wrap=True)
    table.add_column("Date")
    table.add_column("Status")
    table.add_column("Models", justify="right")
    table.add_column("Created at")
    table.add_column("Completed at")

    for row in rows:
        style = _STATUS_STYLES.get(row["status"], "")
        table.add_row(
            row["run_id"],
            row["run_date"] or "—",
            f"[{style}]{row['status']}[/{style}]",
            str(row["model_count"]),
            _fmt_ts(row["created_at"]),
            _fmt_ts(row["completed_at"]) if row["completed_at"] else "—",
        )

    return table
