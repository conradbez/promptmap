"""
``rag_brave_search(query)`` — web search through the Brave Search API.

    {% set hits = rag_brave_search(ref('topic'), count=5) %}
    {{ hits | join("\\n\\n") }}

Returns up to ``count`` (default 8) results, one ``title / url / snippet``
block each. Needs ``BRAVE_API_KEY`` in the environment.

promptmap renders templates synchronously, so a search holds the event loop for
as long as the request takes — hence the timeout.
"""

from __future__ import annotations

import os
import re

import httpx
import promptmap

BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"
MAX_RESULTS = 8
TIMEOUT = 15
# The `[name]` / `[/name]` lines promptmap wraps each `ref()` in — labels for an
# LLM, noise in a search query.
_REF_TAG = re.compile(r"^\[/?[\w.-]+\]$", re.M)
# Brave marks the matched words in a snippet with <strong>.
_HTML_TAG = re.compile(r"</?\w+>")


@promptmap.rag_kind("brave_search")
def brave_search(query: str, count: int = MAX_RESULTS) -> list[str]:
    """The top results for *query*, one `title / url / snippet` block each."""
    query = " ".join(_REF_TAG.sub("", str(query)).split())
    if not query:
        return ["[Search skipped: the query was empty.]"]
    key = os.environ.get("BRAVE_API_KEY")
    if not key:
        raise RuntimeError("rag_brave_search() needs the BRAVE_API_KEY environment variable.")
    try:
        resp = httpx.get(
            BRAVE_URL,
            # Brave caps a query at 400 characters and count at 20.
            params={"q": query[:400], "count": min(int(count), 20)},
            headers={"Accept": "application/json", "X-Subscription-Token": key},
            timeout=TIMEOUT,
        )
    except httpx.TimeoutException:
        raise RuntimeError(f"Brave search timed out after {TIMEOUT}s.") from None
    if resp.status_code != 200:
        raise RuntimeError(f"Brave search failed ({resp.status_code}): {resp.text[:500]}")
    results = resp.json().get("web", {}).get("results", [])[: int(count)]
    if not results:
        return [f"[No web results for: {query}]"]
    return [
        "\n".join(
            filter(None, [r.get("title", ""), r.get("url", ""), _HTML_TAG.sub("", r.get("description", ""))])
        )
        for r in results
    ]
