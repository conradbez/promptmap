"""
The model kinds promptmap ships with.

Each is a :class:`~promptmap.model_types.ModelKind` record: some data, and at most one
function that turns a rendered prompt into a value.  Everything around that —
rendering, the prompt cache, JSON parsing, validation, storage, skip
propagation — belongs to the executor and is identical for every kind,
including the ones you register yourself.

``""`` (plain LLM)   send the rendered prompt to the backend
``template``         the rendered text *is* the output (``exec_fn=None``)
``agent``            a coding agent (the pi harness) as a pure function:
                     Dir in, patched Dir out — see :mod:`promptmap.executor.agent`

``execute_python`` and the multi-harness agent (mini-swe-agent / opencode)
live in ``promptmap[extras]``.

Importing this module registers them; :mod:`promptmap` does that on import.
"""

from __future__ import annotations

from typing import Any

from promptmap.model_types import ModelCall, ModelKind, register_model_kind


async def call_the_llm(rendered: str, call: ModelCall) -> Any:
    """The default: send the rendered prompt to the LLM backend."""
    return await call.llm(rendered)


# ---------------------------------------------------------------------------
# The kinds
# ---------------------------------------------------------------------------

#: The default: render the template, send it to the LLM.
LLM = ModelKind(name="", exec_fn=call_the_llm)

#: Render the template and use it as the output — no LLM call.  For nodes that
#: only reshape what upstream models already produced: a header, a merge of two
#: outputs, a pass-through.  A global instruction is never applied, since the
#: rendered text is the output itself rather than a prompt to answer.
TEMPLATE = ModelKind(name="template", exec_fn=None, accepts_global_instruction=False)

for _kind in (LLM, TEMPLATE):
    register_model_kind(_kind)

from promptmap.executor.agent import KIND as AGENT  # noqa: E402  (registers "agent")
