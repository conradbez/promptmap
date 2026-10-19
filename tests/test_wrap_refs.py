"""
wrap_refs — wrapping each ref() output in [name]…[/name] tags.
"""

from __future__ import annotations

import promptmap
from promptmap.storage import MemoryStorageBackend


class RecordingLLM:
    """Stub llm_call that records every prompt it is handed."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, prompt: str, config: dict | None = None) -> str:
        self.prompts.append(prompt)
        if (config or {}).get("output_format") == "json":
            return '{"title": "T"}'
        return "out"


async def _run(models: dict[str, str], llm, **kwargs) -> dict:
    return await promptmap.async_run(
        models_from_dict=models,
        llm_call=llm,
        verbose=False,
        storage_backend=MemoryStorageBackend(),
        **kwargs,
    )


CHAIN = {"topic": "Name a topic.", "essay": "Write about:\n{{ ref('topic') }}"}


async def test_off_by_default() -> None:
    llm = RecordingLLM()
    await _run(CHAIN, llm)
    assert llm.prompts[1] == "Write about:\nout"


async def test_true_wraps_in_named_tags() -> None:
    llm = RecordingLLM()
    await _run(CHAIN, llm, wrap_refs=True)
    assert llm.prompts[1] == "Write about:\n[topic]\nout\n[/topic]"


async def test_callable_writes_the_tags() -> None:
    llm = RecordingLLM()
    tag = lambda name, is_open: f"<{'' if is_open else '/'}{name}>"
    await _run(CHAIN, llm, wrap_refs=tag)
    assert llm.prompts[1] == "Write about:\n<topic>\nout\n</topic>"


async def test_json_output_stays_structured() -> None:
    llm = RecordingLLM()
    models = {
        "meta": '{{ config(output_format="json") }}Give a title.',
        "a": "{{ ref('meta').title }}",
        "b": "{{ ref('meta.title') }}",
    }
    await _run(models, llm, wrap_refs=True)
    assert "T" in llm.prompts
    assert "[meta.title]\nT\n[/meta.title]" in llm.prompts
