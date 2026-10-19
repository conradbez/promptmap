"""
Extra promptmap model kinds. Importing this package registers all of them:
``python_modal`` (`modal_exec`), ``agent_modal`` (`agent_exec`) and
``agent37`` (`agent37_exec`), plus the RAG kind ``brave_search``
(`brave_search`, called as ``rag_brave_search(...)`` in templates).
"""

from . import agent37_exec, agent_exec, brave_search, modal_exec

__all__ = ["agent37_exec", "agent_exec", "brave_search", "modal_exec"]
