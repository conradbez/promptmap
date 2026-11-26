"""
Chat conversations as one prompt, for backends that take no ``messages``.

Used by :meth:`promptmap.executor.run_context.RunContext.call_llm_messages`,
the path a kind with ``chat=True`` (``model_type="chat"`` from
``promptmap[extras]``) sends its conversation through.
"""

from __future__ import annotations


def flatten_messages(messages: list[dict]) -> str:
    """The conversation as one prompt, for an ``llm_call(prompt)`` that takes
    no ``messages``::

        <system>

        <request>

        [your earlier reply]
        ...
        [/your earlier reply]

        [follow-up 1]
        ...
        [/follow-up 1]

    A conversation that is only the request flattens to the request itself
    (after the system message, if any).
    """
    parts: list[str] = []
    seen_request = False
    follow_ups = 0
    for message in messages:
        role, content = message["role"], str(message["content"])
        if role == "system":
            parts.append(content)
        elif role == "user" and not seen_request:
            parts.append(content)
            seen_request = True
        elif role == "assistant":
            parts.append(f"[your earlier reply]\n{content}\n[/your earlier reply]")
        else:
            follow_ups += 1
            parts.append(f"[follow-up {follow_ups}]\n{content}\n[/follow-up {follow_ups}]")
    return "\n\n".join(parts)
