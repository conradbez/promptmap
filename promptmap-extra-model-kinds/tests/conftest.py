"""Shared helpers for the extra-kinds tests."""

from __future__ import annotations

import asyncio
import json

from promptmap.executor.executor import execute_run
from promptmap.executor.graph import build_models_from_dict
from promptmap.storage.memory import MemoryStorageBackend

import promptmap_extra_model_kinds  # noqa: F401  (registers the kinds)


def stub_llm(prompt: str, config: dict | None = None) -> str:
    if (config or {}).get("output_format") == "json":
        return json.dumps(["a", "b"])
    return "resp"


def run_models(models: dict[str, str], *, storage=None, llm_call=stub_llm, **kwargs):
    """Execute *models* and return (storage, run_id, {model_name: ModelRunResult})."""
    storage = storage or MemoryStorageBackend()
    storage.init_db()
    specs = list(build_models_from_dict(models).values())
    run_id = storage.create_run(model_count=len(specs))
    results = asyncio.run(execute_run(
        run_id=run_id,
        ordered_models=specs,
        storage_backend=storage,
        llm_call=llm_call,
        **kwargs,
    ))
    return storage, run_id, {r.model_name: r for r in results}
