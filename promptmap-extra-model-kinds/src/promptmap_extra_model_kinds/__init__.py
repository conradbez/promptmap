"""
Extra promptmap model kinds. Importing this package registers all of them:
``execute_python`` (`python_exec`), ``agent_multi_harness`` (`agent_multi_harness_exec`:
mini-swe-agent or opencode), ``python_modal`` (`modal_exec`), ``agent_modal``
(`agent_exec`) and ``agent37`` (`agent37_exec`), plus the web-search RAG kinds
``brave_search``, ``gemini_search``, ``openai_search`` and ``anthropic_search``
(called as ``rag_<name>(...)`` in templates).
"""

from . import (
    agent37_exec,
    agent_exec,
    agent_multi_harness_exec,
    anthropic_search,
    brave_search,
    gemini_search,
    modal_exec,
    openai_search,
    python_exec,
)

__all__ = [
    "agent37_exec",
    "agent_exec",
    "agent_multi_harness_exec",
    "anthropic_search",
    "brave_search",
    "gemini_search",
    "modal_exec",
    "openai_search",
    "python_exec",
]
