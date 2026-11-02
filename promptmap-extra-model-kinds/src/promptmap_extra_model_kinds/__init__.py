"""
Extra promptmap model kinds. Importing this package registers all of them:
``python_modal`` (`modal_exec`), ``agent_modal`` (`agent_exec`) and
``agent37`` (`agent37_exec`), ``local_functional_agent`` (`functional_agent_exec`), plus the web-search RAG kinds ``brave_search``, ``gemini_search``,
``openai_search`` and ``anthropic_search`` (called as ``rag_<name>(...)`` in templates).
"""

from . import (
    agent37_exec,
    agent_exec,
    anthropic_search,
    brave_search,
    functional_agent_exec,
    gemini_search,
    modal_exec,
    openai_search,
)

__all__ = [
    "agent37_exec",
    "agent_exec",
    "anthropic_search",
    "brave_search",
    "functional_agent_exec",
    "gemini_search",
    "modal_exec",
    "openai_search",
]
