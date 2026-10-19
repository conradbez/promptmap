"""What every LLM-backed search RAG kind shares: query cleanup, API keys, the prompt."""

from __future__ import annotations

import os
import re

INSTRUCTIONS = (
    "Search the web for the query below. Report the most relevant, current findings "
    "as concise factual bullet points, with dates, figures and names where they matter. "
    "No preamble."
)
EMPTY = "[Search skipped: the query was empty.]"

# The `[name]` / `[/name]` lines promptmap wraps each `ref()` in — labels for an
# LLM, noise in a search query.
_REF_TAG = re.compile(r"^\[/?[\w.-]+\]$", re.M)


def clean_query(query: object) -> str:
    """*query* without ref tags, on one line."""
    return " ".join(_REF_TAG.sub("", str(query)).split())


def prompt(query: str, instructions: str) -> str:
    return f"{instructions}\n\nQuery: {query}"


def load_env(name: str) -> str | None:
    """Environment variable *name*, falling back to a ``.env`` in the working directory or a parent."""
    if not os.environ.get(name):
        from dotenv import find_dotenv, load_dotenv

        load_dotenv(find_dotenv(usecwd=True))  # never overrides a variable already set
    return os.environ.get(name)


def api_key(name: str, rag: str) -> str:
    """Like :func:`load_env`, but raise naming the RAG kind when the key is missing."""
    key = load_env(name)
    if not key:
        raise RuntimeError(f"rag_{rag}() needs {name} in the environment or a .env file.")
    return key


def source(title: str | None, url: str | None) -> str:
    """One `title / url` block."""
    return "\n".join(filter(None, [title or "", url or ""]))


def dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))
