"""agent_modal / agent37 run on a bullet's config(model=...), else the run's model."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS

import pytest

from promptmap.executor.graph import build_models_from_dict
from promptmap_extra_model_kinds import agent37_exec, agent_exec


class _Stop(Exception):
    pass


def _model_used(module, patch_name, text, run_model="run-model"):
    """The model *module*'s execute hands to *patch_name*, then stop."""
    seen = []

    def capture(*args):
        seen.append(args[-1])
        raise _Stop

    spec = next(iter(build_models_from_dict({"a": text}).values()))
    token = agent_exec.use_run(agent_exec.RunSettings(api_key="k", model=run_model))
    original = getattr(module, patch_name)
    setattr(module, patch_name, capture)
    try:
        with pytest.raises(_Stop):
            asyncio.run(module.execute("Do it", NS(spec=spec)))
    finally:
        setattr(module, patch_name, original)
        agent_exec._settings.reset(token)
    return seen[0]


@pytest.mark.parametrize("module, patch_name, kind", [
    (agent_exec, "_secret_env", "agent_modal"),
    (agent37_exec, "_provider", "agent37"),
])
def test_bullet_model_wins(module, patch_name, kind):
    text = f'{{{{ config(model_type="{kind}", model="bullet-model") }}}}\nDo it'
    assert _model_used(module, patch_name, text) == "bullet-model"


@pytest.mark.parametrize("module, patch_name, kind", [
    (agent_exec, "_secret_env", "agent_modal"),
    (agent37_exec, "_provider", "agent37"),
])
def test_run_model_without_one(module, patch_name, kind):
    text = f'{{{{ config(model_type="{kind}") }}}}\nDo it'
    assert _model_used(module, patch_name, text) == "run-model"
