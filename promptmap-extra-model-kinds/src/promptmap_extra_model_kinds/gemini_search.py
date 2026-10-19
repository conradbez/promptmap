"""
``rag_gemini_search(query)`` — web search through Gemini Flash with Google Search grounding.

    {% set hits = rag_gemini_search(ref('topic')) %}
    {{ hits | join("\\n\\n") }}

Returns the grounded answer first, then one ``title / url`` block per source
Gemini cited. Pass ``instructions="..."`` to change what Gemini is asked to
report about the query.

Reads ``GEMINI_API_KEY`` from the environment, falling back to a ``.env`` in
the working directory (or a parent). ``GEMINI_SEARCH_MODEL`` overrides the
model.

promptmap renders templates synchronously, so a search holds the event loop for
as long as the request takes — hence the timeout.
"""

from __future__ import annotations

import os

import httpx
import promptmap

from ._search import EMPTY, INSTRUCTIONS, api_key, clean_query, dedupe, prompt, source

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_MODEL = "gemini-3.6-flash"
TIMEOUT = 60


@promptmap.rag_kind("gemini_search")
def gemini_search(query: str, instructions: str = INSTRUCTIONS, model: str | None = None) -> list[str]:
    """The grounded answer for *query*, then one `title / url` block per cited source."""
    query = clean_query(query)
    if not query:
        return [EMPTY]
    model = model or os.environ.get("GEMINI_SEARCH_MODEL", DEFAULT_MODEL)
    try:
        resp = httpx.post(
            GEMINI_URL.format(model=model),
            headers={"x-goog-api-key": api_key("GEMINI_API_KEY", "gemini_search")},
            json={
                "contents": [{"parts": [{"text": prompt(query, instructions)}]}],
                "tools": [{"google_search": {}}],
            },
            timeout=TIMEOUT,
        )
    except httpx.TimeoutException:
        raise RuntimeError(f"Gemini search timed out after {TIMEOUT}s.") from None
    if resp.status_code != 200:
        raise RuntimeError(f"Gemini search failed ({resp.status_code}): {resp.text[:500]}")
    candidate = (resp.json().get("candidates") or [{}])[0]
    answer = "".join(p.get("text", "") for p in candidate.get("content", {}).get("parts", [])).strip()
    if not answer:
        return [f"[No web results for: {query}]"]
    sources = [
        source(web.get("title"), web.get("uri"))
        for chunk in candidate.get("groundingMetadata", {}).get("groundingChunks", [])
        if (web := chunk.get("web"))
    ]
    return [answer, *dedupe(sources)]
