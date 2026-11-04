"""on_model_start / on_model_done callbacks on promptmap.run() and promptmap.async_run()."""

from __future__ import annotations

import asyncio

import promptmap
from promptmap.storage import MemoryStorageBackend

MODELS = {
    "topic": "Name one topic.",
    "outline": "Outline {{ ref('topic') }}.",
    "article": "Write about {{ ref('outline') }}.",
}


def _events(llm_call, models=MODELS) -> tuple[list[tuple[str, str]], dict]:
    events: list[tuple[str, str]] = []
    results = promptmap.run(
        models_from_dict=models,
        llm_call=llm_call,
        verbose=False,
        storage_backend=MemoryStorageBackend(),
        on_model_start=lambda name: events.append(("start", name)),
        on_model_done=lambda r: events.append((r.status, r.model_name)),
    )
    return events, results


def test_callbacks_fire_in_dependency_order() -> None:
    events, _ = _events(lambda prompt: "ok")
    assert events == [
        ("start", "topic"), ("success", "topic"),
        ("start", "outline"), ("success", "outline"),
        ("start", "article"), ("success", "article"),
    ]


def test_done_reports_error_and_skipped() -> None:
    def llm(prompt: str) -> str:
        if prompt.startswith("Outline"):
            raise RuntimeError("boom")
        return "ok"

    events, _ = _events(llm)
    assert ("error", "outline") in events
    # article never starts; it is reported as skipped only.
    assert ("start", "article") not in events
    assert ("skipped", "article") in events


def test_done_receives_model_run_result() -> None:
    seen: list[promptmap.ModelRunResult] = []
    promptmap.run(
        models_from_dict={"topic": "Name one topic."},
        llm_call=lambda prompt: "hello",
        verbose=False,
        storage_backend=MemoryStorageBackend(),
        on_model_done=seen.append,
    )
    assert len(seen) == 1
    assert isinstance(seen[0], promptmap.ModelRunResult)
    assert seen[0].llm_output == "hello"


def test_async_run_accepts_callbacks() -> None:
    started: list[str] = []

    async def llm(prompt: str) -> str:
        return "ok"

    asyncio.run(promptmap.async_run(
        models_from_dict=MODELS,
        llm_call=llm,
        verbose=False,
        storage_backend=MemoryStorageBackend(),
        on_model_start=started.append,
    ))
    assert started == ["topic", "outline", "article"]


def test_callbacks_fire_with_verbose_logging() -> None:
    events: list[str] = []
    promptmap.run(
        models_from_dict={"topic": "Name one topic."},
        llm_call=lambda prompt: "ok",
        verbose=True,
        storage_backend=MemoryStorageBackend(),
        on_model_start=events.append,
    )
    assert events == ["topic"]


def test_agent_step_bar_in_run_callbacks():
    """An agent's reported steps show as a block bar of step/limit on its status line."""
    import asyncio
    import io
    from types import SimpleNamespace

    from rich.console import Console

    from promptmap.cli import pretty_print
    from promptmap.model_types import progress_listener, report_progress

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=100)
    token = progress_listener.set(None)
    try:
        on_start, on_done = pretty_print.make_run_callbacks(console, [], total=1)
        on_start("fix")

        async def agent_steps():
            for step in (1, 2, 3):
                await asyncio.to_thread(report_progress, "fix", step, 10)

        asyncio.run(agent_steps())
        on_done(SimpleNamespace(status="success", execution_ms=5))
    finally:
        progress_listener.reset(token)

    assert "■ ■ ■ ■ ■ ■ ■ ■ ■ ■ 3/10 OK" in buf.getvalue()
    done, todo = "[bold]■[/bold]", "[grey50]■[/grey50]"
    assert pretty_print.step_bar(2, 5) == " ".join([done] * 2 + [todo] * 3) + " [dim]2/5[/dim]"
    assert pretty_print.step_bar(3, 0) == " ".join([done] * 3) + " [dim]3 steps[/dim]"
    assert pretty_print.step_bar(50, 100).count(done) == 10  # scaled to 20 squares
