"""
``rag_anthropic_search(query)`` — web search through Claude's ``web_search`` server tool.

    {% set hits = rag_anthropic_search(ref('topic')) %}
    {{ hits | join("\\n\\n") }}

Returns the answer first, then one ``title / url`` block per source it cited.
Pass ``instructions="..."`` to change what is asked about the query.

Credentials resolve as the Anthropic SDK resolves them (``ANTHROPIC_API_KEY``,
``ANTHROPIC_AUTH_TOKEN``, an ``ant auth login`` profile), after loading a
``.env`` from the working directory (or a parent) if the key is not set.
``ANTHROPIC_SEARCH_MODEL`` overrides the model; ``ANTHROPIC_SEARCH_EFFORT`` the
effort (default ``low``: a search wants a quick summary, not deep reasoning).

On the default model a declined request is re-run on a fallback model
(``fallbacks="default"``), so a safety-classifier false positive does not fail
the bullet.
"""

from __future__ import annotations

import os

import anthropic
import promptmap

from ._search import EMPTY, INSTRUCTIONS, clean_query, dedupe, load_env, prompt, source

DEFAULT_MODEL = "claude-opus-5-5"
TIMEOUT = 300
# A server-tool turn that hits the server's iteration limit stops with
# `pause_turn`; it resumes when re-sent. Cap the resumes.
MAX_CONTINUATIONS = 5
WEB_SEARCH_TOOL = {"type": "web_search_20260209", "name": "web_search"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"


@promptmap.rag_kind("anthropic_search")
def anthropic_search(query: str, instructions: str = INSTRUCTIONS, model: str | None = None) -> list[str]:
    """The web-grounded answer for *query*, then one `title / url` block per cited source."""
    query = clean_query(query)
    if not query:
        return [EMPTY]
    model = model or os.environ.get("ANTHROPIC_SEARCH_MODEL", DEFAULT_MODEL)
    load_env("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(timeout=TIMEOUT)
    # The fallback router is only offered for the newest models; skip it on an override.
    fallback = {"betas": [FALLBACK_BETA], "fallbacks": "default"} if model == DEFAULT_MODEL else {}
    messages: list[dict] = [{"role": "user", "content": prompt(query, instructions)}]

    blocks = []
    for _ in range(MAX_CONTINUATIONS + 1):
        try:
            response = client.beta.messages.create(
                model=model,
                max_tokens=16000,
                output_config={"effort": os.environ.get("ANTHROPIC_SEARCH_EFFORT", "low")},
                tools=[WEB_SEARCH_TOOL],
                messages=messages,
                **fallback,
            )
        except anthropic.APIStatusError as exc:
            raise RuntimeError(f"Anthropic search failed ({exc.status_code}): {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise RuntimeError(f"Anthropic search could not connect: {exc}") from exc
        except TypeError as exc:
            if "authentication" not in str(exc):
                raise
            raise RuntimeError(f"rag_anthropic_search() has no Anthropic credentials: {exc}") from exc
        blocks.extend(response.content)
        if response.stop_reason != "pause_turn":
            break
        messages.append({"role": "assistant", "content": response.content})

    if response.stop_reason == "refusal":
        category = response.stop_details.category if response.stop_details else None
        raise RuntimeError(f"Anthropic search was declined (category: {category}).")
    text_blocks = [b for b in blocks if b.type == "text"]
    answer = "".join(b.text for b in text_blocks).strip()
    if not answer:
        return [f"[No web results for: {query}]"]
    sources = [
        source(c.title, c.url)
        for b in text_blocks
        for c in (b.citations or [])
        if c.type == "web_search_result_location"
    ]
    return [answer, *dedupe(sources)]
