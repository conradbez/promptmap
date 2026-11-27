"""
``model_type="chat"`` — an answer refined over several turns.

Shipped in ``promptmap[extras]``.  The storage, the commit-on-success rule,
``run(chat_input=..., chat_reset=...)``, ``get_chat()`` and ``promptmap chat``
live in promptmap itself and work for any kind registered with ``chat=True``;
this module is the kind.

The ``.prompt`` file is the **request**, rendered fresh on every run with the
current ``ref()``s and promptdata.  Follow-ups and the replies to them are kept
in the run database (``chat_turns``), never in the file, and each run sends
the LLM real ``user`` / ``assistant`` messages::

    system:    chat_instruction
    user:      <request, rendered now>
    assistant: reply 1
    user:      follow-up 1
    assistant: reply 2
    user:      follow-up 2        <- the new one, if this run brought one

Only the latest reply is the model's output.

The system message is ``chat_instruction``, or the text of the file named by
``chat_instruction_file`` (relative to the directory promptmap runs in — the
project root), read on every run: many chats can share one instructions file,
and editing it changes what all of them are told.  The file's text is part of
the conversation sent, so editing it is a cache miss.  A missing file is an
error, not a silent fallback.

Rules
-----
- A run with a new follow-up (``run(chat_input={"model": "..."})``) appends
  that follow-up and its reply to the thread.
- A run without one regenerates the last reply in place and adds no turn.
- Turns are committed only when the model's result is stored successfully.
- Inputs never enter the history: the request is always rendered from the
  current inputs, while earlier replies stay exactly as they were.
- The cache key is the conversation sent, so a rerun with nothing changed is
  a prompt-cache hit.
"""

from __future__ import annotations

import json
from typing import Any

from promptmap.chat_messages import flatten_messages  # noqa: F401  (re-exported)
from promptmap.model_types import ModelCall, ModelKind, register_model_kind

#: The system message when a model sets no ``chat_instruction``.
DEFAULT_CHAT_INSTRUCTION = (
    "Only your most recent reply is passed on to the rest of the pipeline, so "
    "always reply with the complete, updated answer, never just the changes. "
    "Each follow-up amends your answer; where they conflict, the later one wins."
)

#: ``chat_instruction`` values that send no system message at all.
_NO_INSTRUCTION = frozenset({"", "false", "none", "no", "off", "0"})


def build_messages(
    request: str,
    turns: "list[dict] | tuple[dict, ...]",
    follow_up: str | None = None,
    instruction: str | None = DEFAULT_CHAT_INSTRUCTION,
    max_turns: int | None = None,
) -> tuple[list[dict], bool]:
    """The messages for one run, and whether its reply replaces the last turn.

    *turns* is the current thread (``role`` / ``content``).  With a
    *follow_up* the whole thread is sent and the follow-up appended; without
    one the last reply is left out, since this run regenerates it.

    *max_turns* keeps only that many of the most recent turns before the new
    follow-up (never starting on a follow-up, so the messages still
    alternate after the request).
    """
    history = [{"role": t["role"], "content": t["content"]} for t in turns]
    replace_last = False
    if not follow_up and history and history[-1]["role"] == "assistant":
        history = history[:-1]
        replace_last = True

    if max_turns is not None:
        history = history[-max_turns:] if max_turns > 0 else []
        while history and history[0]["role"] == "user":
            history = history[1:]

    messages: list[dict] = []
    if instruction is not None and str(instruction).strip().lower() not in _NO_INSTRUCTION:
        messages.append({"role": "system", "content": str(instruction)})
    messages.append({"role": "user", "content": request})
    messages.extend(history)
    if follow_up:
        messages.append({"role": "user", "content": follow_up})
    return messages, replace_last


def _max_turns(value: Any) -> int | None:
    if value is None or str(value).strip().lower() in ("", "none"):
        return None
    return int(value)


def chat_instruction(config: dict) -> str | None:
    """The system message: ``chat_instruction_file``'s text, else ``chat_instruction``, else the default."""
    path = config.get("chat_instruction_file")
    if path:
        from pathlib import Path

        file = Path(str(path)).expanduser()
        if not file.is_file():
            raise FileNotFoundError(
                f"chat_instruction_file={str(path)!r}: no such file (relative to {Path.cwd()})."
            )
        return file.read_text(encoding="utf-8").strip()
    return config.get("chat_instruction", DEFAULT_CHAT_INSTRUCTION)


async def chat_exec(rendered: str, call: ModelCall) -> Any:
    """Send the request plus the thread to the LLM and stage the new turns."""
    config = call.spec.config
    messages, replace_last = build_messages(
        rendered,
        call.chat_turns,
        call.chat_input,
        chat_instruction(config),
        _max_turns(config.get("max_turns")),
    )
    reply = await call.llm_messages(messages)
    text = reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)

    append: list[dict] = []
    if call.chat_input:
        append.append({"role": "user", "content": call.chat_input})
    if replace_last:
        call.stage_chat(append, text)
    else:
        call.stage_chat(append + [{"role": "assistant", "content": text}], None)
    return reply


KIND = register_model_kind(ModelKind(
    name="chat",
    exec_fn=chat_exec,
    config_keys=frozenset({"chat_instruction", "chat_instruction_file", "max_turns"}),
    accepts_global_instruction=True,
    chat=True,
))
