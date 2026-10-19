"""
RAG backends — named retrieval functions callable from any template.

Register a function under a name and every template can call it as
``rag_<name>(...)``::

    import promptmap

    @promptmap.rag_kind("notes")
    def search_notes(query: str, k: int = 3) -> list[str]:
        ...

    # map/answer.prompt
    {% for hit in rag_notes(ref('question'), k=5) %}
    {{ hit }}
    {% endfor %}

A RAG function takes whatever the template passes and returns ``list[str]``
or a bare ``str`` (wrapped).  ``False`` / ``None`` mean "no results" and
become ``[False]`` so ``{% if hits[0] %}`` stays safe.  Rendering is
synchronous, so a RAG function is a plain ``def``, not ``async``.

Where to register
-----------------
``rag.py`` beside ``map/`` is imported before the models render, so decorators
in it register project-local RAGs; ``client.py`` works too.  A package
declares its RAGs under the ``promptmap.rag_kinds`` entry point group and
``pip install`` is then enough, exactly as for model kinds::

    [project.entry-points."promptmap.rag_kinds"]
    brave_search = "my_pkg.brave"     # a module that registers on import

Legacy ``return_list_RAG_results``
----------------------------------
``rag.py`` may still define ``do_RAG(*args)``; it backs
``return_list_RAG_results()`` as before (or ``run(rag_call=...)`` does).
"""

from __future__ import annotations

import importlib.util
import logging
import os
import re
from typing import Any, Callable

#: ``def rag_fn(*args, **kwargs) -> list[str] | str | None``
RagFn = Callable[..., Any]

#: Templates call a RAG registered as "notes" as ``rag_notes(...)``.
TEMPLATE_PREFIX = "rag_"

#: Entry point group installed packages register their RAGs under.
ENTRY_POINT_GROUP = "promptmap.rag_kinds"

_REGISTRY: dict[str, RagFn] = {}
_plugins_loaded = False

_NAME = re.compile(r"^[A-Za-z_]\w*$")
#: A ``rag_<name>(`` call in template source.
_CALL_PATTERN = re.compile(r"\brag_(\w+)\s*\(")


def load_plugins() -> None:
    """Import every installed package that registers promptmap RAGs.

    Runs once, lazily — see :func:`promptmap.model_types.load_plugins` for why.
    A plugin that fails to load is logged and skipped.
    """
    global _plugins_loaded
    if _plugins_loaded or os.environ.get("PROMPTMAP_DISABLE_PLUGINS"):
        return
    _plugins_loaded = True

    from importlib.metadata import entry_points

    for ep in entry_points(group=ENTRY_POINT_GROUP):
        try:
            obj = ep.load()  # importing runs the @promptmap.rag_kind decorators
            if callable(obj):
                obj()  # optional explicit register() hook
        except Exception:
            logging.getLogger("promptmap").exception("promptmap RAG plugin %r failed to load", ep.name)


def register_rag_kind(name: str, fn: RagFn) -> RagFn:
    """Register *fn* as ``rag_<name>()`` in templates, replacing any RAG of that name."""
    if not _NAME.match(name):
        raise ValueError(f"RAG name {name!r} must be a valid Python identifier.")
    _REGISTRY[name] = fn
    return fn


def rag_kind(name: str):
    """Decorator form of :func:`register_rag_kind`.  Returns the function unchanged."""
    def decorator(fn: RagFn) -> RagFn:
        return register_rag_kind(name, fn)

    return decorator


def get_rag_kind(name: str) -> RagFn | None:
    """Return the RAG registered under *name*, or None."""
    load_plugins()
    return _REGISTRY.get(name)


def known_rag_kinds() -> set[str]:
    """Return every registered RAG name (without the ``rag_`` prefix)."""
    load_plugins()
    return set(_REGISTRY)


def _as_list(result: Any) -> list:
    if result is False or result is None:
        return [False]
    if isinstance(result, str):
        return [result]
    return list(result)


def template_functions(template_source: str = "") -> dict[str, Callable[..., list]]:
    """Every registered RAG as its template global: ``{"rag_<name>": fn}``.

    A ``rag_<name>()`` that *template_source* calls but nobody registered gets
    a stub that raises a clear error when called — not at render start, so a
    branch that never runs (``{% if promptdata('web') %}``) costs nothing.
    """
    load_plugins()
    functions = {
        TEMPLATE_PREFIX + name: (lambda *a, _fn=fn, **kw: _as_list(_fn(*a, **kw)))
        for name, fn in _REGISTRY.items()
    }
    for name in _CALL_PATTERN.findall(template_source):
        functions.setdefault(TEMPLATE_PREFIX + name, lambda *a, _name=name, **kw: _unregistered(_name))
    return functions


def _unregistered(name: str):
    raise NameError(
        f"rag_{name}() is not a registered RAG. Known RAGs: "
        f"{', '.join(sorted(known_rag_kinds())) or '(none)'}. Register it with "
        f"@promptmap.rag_kind('{name}') in rag.py, or install a package that provides it."
    )


# ---------------------------------------------------------------------------
# Project rag.py
# ---------------------------------------------------------------------------

def resolve_rag_call(map_dir: str) -> Callable[..., list[str]]:
    """
    Import rag.py alongside map_dir (i.e. in its parent), if present.

    Importing it runs any ``@promptmap.rag_kind`` decorators.  If it defines
    ``do_RAG``, return a wrapper (backing ``return_list_RAG_results``) that
    always returns list[str]; otherwise return a stub that raises on call.
    """
    candidate = os.path.join(os.path.dirname(map_dir), "rag.py")
    if os.path.isfile(candidate):
        spec = importlib.util.spec_from_file_location("_promptmap_user_rag", candidate)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod, "do_RAG"):
            return lambda *args, _fn=mod.do_RAG: _as_list(_fn(*args))

    def _missing(*args) -> list[str]:
        raise RuntimeError(
            "return_list_RAG_results() was called but no do_RAG was found.\n"
            "Define 'do_RAG(*args) -> list[str] | str' in rag.py (alongside your map/ directory), "
            "or register a named RAG with @promptmap.rag_kind('name') and call rag_name(...)."
        )

    return _missing
