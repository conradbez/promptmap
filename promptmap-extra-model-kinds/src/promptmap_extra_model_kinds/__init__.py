"""
Extra promptmap model kinds. Importing this package registers all of them:
``python_modal`` (`modal_exec`), ``agent_modal`` (`agent_exec`) and
``agent37`` (`agent37_exec`), plus the web-search RAG kinds ``brave_search``, ``gemini_search``,
``openai_search`` and ``anthropic_search`` (called as ``rag_<name>(...)`` in templates).
"""

from . import (
    agent37_exec,
    agent_exec,
    anthropic_search,
    brave_search,
    gemini_search,
    modal_exec,
    openai_search,
)

__all__ = [
    "agent37_exec",
    "agent_exec",
    "anthropic_search",
    "brave_search",
    "gemini_search",
    "modal_exec",
    "openai_search",
]
