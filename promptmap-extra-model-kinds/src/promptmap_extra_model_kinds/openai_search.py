"""
``rag_openai_search(query)`` — web search through the OpenAI Responses API's ``web_search`` tool.

    {% set hits = rag_openai_search(ref('topic')) %}
    {{ hits | join("\\n\\n") }}

Returns the answer first, then one ``title / url`` block per source it cited.
Pass ``instructions="..."`` to change what is asked about the query.

Reads ``OPENAI_API_KEY`` from the environment, falling back to a ``.env`` in the
working directory (or a parent). ``OPENAI_SEARCH_MODEL`` overrides the model.
"""

from __future__ import annotations

import os

import openai
import promptmap

from ._search import EMPTY, INSTRUCTIONS, api_key, clean_query, dedupe, prompt, source

DEFAULT_MODEL = "gpt-5.4-mini"
TIMEOUT = 120


@promptmap.rag_kind("openai_search")
def openai_search(query: str, instructions: str = INSTRUCTIONS, model: str | None = None) -> list[str]:
    """The web-grounded answer for *query*, then one `title / url` block per cited source."""
    query = clean_query(query)
    if not query:
        return [EMPTY]
    client = openai.OpenAI(api_key=api_key("OPENAI_API_KEY", "openai_search"), timeout=TIMEOUT)
    try:
        response = client.responses.create(
            model=model or os.environ.get("OPENAI_SEARCH_MODEL", DEFAULT_MODEL),
            tools=[{"type": "web_search"}],
            input=prompt(query, instructions),
        )
    except openai.APIError as exc:
        raise RuntimeError(f"OpenAI search failed: {exc}") from exc
    answer = response.output_text.strip()
    if not answer:
        return [f"[No web results for: {query}]"]
    sources = [
        source(note.title, note.url)
        for item in response.output
        if item.type == "message"
        for part in item.content
        if part.type == "output_text"
        for note in part.annotations
        if note.type == "url_citation"
    ]
    return [answer, *dedupe(sources)]
