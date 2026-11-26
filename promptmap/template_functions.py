"""
Template functions — your own Jinja functions, callable from every template.

Register a function under a name and every ``.prompt`` file can call it::

    # jinja_template_functions.py  (beside map/)
    import promptmap

    @promptmap.template_function("shout")
    def shout(ctx, text):
        return text.upper()

    # map/answer.prompt
    {{ shout(ref('question')) }}

The first argument is always a :class:`TemplateContext`, the way a model kind
always receives its ``call``: the name of the model being rendered, every
model's spec, and the DAG.  The template passes the remaining arguments.
A function that needs no context ignores ``ctx``.

Where to register
-----------------
``jinja_template_functions.py`` beside ``map/`` is imported before the models
are parsed (along with ``client.py``, which works too).  A package declares its
functions under the ``promptmap.template_functions`` entry point group, and
``pip install`` is then enough, exactly as for model kinds::

    [project.entry-points."promptmap.template_functions"]
    shout = "my_pkg.shout"            # a module that registers on import
    other = "my_pkg:register"         # or a function promptmap calls once

promptmap ships one function this way: ``org_outline()`` (see
:mod:`promptmap.org_outline`).
"""

from __future__ import annotations

import importlib.util
import logging
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Mapping

if TYPE_CHECKING:
    import networkx as nx

    from promptmap.model_spec import ModelSpec

#: ``def fn(ctx: TemplateContext, *args, **kwargs) -> Any``
TemplateFn = Callable[..., Any]

#: Entry point group installed packages register their functions under.
ENTRY_POINT_GROUP = "promptmap.template_functions"

#: The project file imported for its registrations, beside ``map/``.
PROJECT_FILE = "jinja_template_functions.py"

#: Names promptmap itself puts in every template; a plugin may not shadow them.
RESERVED_NAMES = frozenset({
    "ref", "promptdata", "config", "model", "previous", "previous_of", "each", "prompt",
    "was_skipped", "skip_and_set_to_value", "skip_this_and_downstream",
    "return_list_RAG_results",
})

_REGISTRY: dict[str, TemplateFn] = {}
_plugins_loaded = False
_NAME = re.compile(r"^[A-Za-z_]\w*$")

#: A ``{{ config(...) }}`` call, possibly spanning lines.
_CONFIG_CALL = re.compile(r"\{\{-?\s*config\s*\(.*?\)\s*-?\}\}", re.DOTALL)


@dataclass(frozen=True)
class TemplateContext:
    """What a template function knows about the render it is called from."""

    #: The model being rendered ("" outside a run, e.g. in a test prompt).
    model: str = ""
    #: Every model in the project — not only the selected ones — by name.
    specs: Mapping[str, "ModelSpec"] = field(default_factory=dict)
    #: Edge A → B means B ref()s A.  Built from *specs* when not given.
    dag: "nx.DiGraph | None" = None

    def __post_init__(self) -> None:
        if self.dag is None:
            object.__setattr__(self, "dag", dag_from_specs(self.specs))

    def source(self, name: str) -> str:
        """The raw, unrendered template of model *name* ("" if unknown)."""
        spec = self.specs.get(name)
        return spec.source if spec else ""

    def body_lines(self, name: str, n: int) -> list[str]:
        """The first *n* non-blank lines of *name*'s raw template, config() calls removed."""
        if n <= 0:
            return []
        text = _CONFIG_CALL.sub("", self.source(name))
        return [line.strip() for line in text.splitlines() if line.strip()][:n]


def dag_from_specs(specs: Mapping[str, "ModelSpec"]) -> "nx.DiGraph":
    """The ref() graph of *specs*; refs to models outside *specs* are dropped."""
    import networkx as nx

    dag = nx.DiGraph()
    dag.add_nodes_from(sorted(specs))
    for name in sorted(specs):
        for dep in sorted(specs[name].depends_on):
            if dep in specs:
                dag.add_edge(dep, name)
    return dag


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def load_plugins() -> None:
    """Import every installed package that registers template functions.

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
            obj = ep.load()
            if callable(obj):
                obj()
        except Exception:
            logging.getLogger("promptmap").exception(
                "promptmap template-function plugin %r failed to load", ep.name
            )


def register_template_function(name: str, fn: TemplateFn) -> TemplateFn:
    """Register *fn* as ``name()`` in every template, replacing any function of that name."""
    if not _NAME.match(name):
        raise ValueError(f"Template function name {name!r} must be a valid Python identifier.")
    if name in RESERVED_NAMES or name.startswith("rag_"):
        raise ValueError(f"Template function name {name!r} is reserved by promptmap.")
    _REGISTRY[name] = fn
    return fn


def template_function(name: str):
    """Decorator form of :func:`register_template_function`.  Returns the function unchanged."""
    def decorator(fn: TemplateFn) -> TemplateFn:
        return register_template_function(name, fn)

    return decorator


def get_template_function(name: str) -> TemplateFn | None:
    """Return the function registered under *name*, or None."""
    load_plugins()
    return _REGISTRY.get(name)


def known_template_functions() -> set[str]:
    """Return every registered template-function name."""
    load_plugins()
    return set(_REGISTRY)


def bound_functions(ctx: TemplateContext) -> dict[str, Callable[..., Any]]:
    """Every registered function as its template global, with *ctx* bound first."""
    load_plugins()
    return {
        name: (lambda *a, _fn=fn, **kw: _fn(ctx, *a, **kw))
        for name, fn in _REGISTRY.items()
    }


def config_stubs() -> dict[str, Callable[..., str]]:
    """No-op stand-ins for config capture, so a call never cuts the capture short."""
    load_plugins()
    return {name: (lambda *a, **kw: "") for name in _REGISTRY}


# ---------------------------------------------------------------------------
# Project file
# ---------------------------------------------------------------------------

_loaded_files: set[str] = set()


def load_project_file(map_dir: str) -> None:
    """Import ``jinja_template_functions.py`` beside *map_dir*, once per path.

    A file that fails to import is logged, not raised: templates that call its
    functions then fail with an undefined-name error that points at them.
    """
    candidate = os.path.join(os.path.dirname(os.path.normpath(map_dir)), PROJECT_FILE)
    resolved = os.path.realpath(candidate)
    if resolved in _loaded_files or not os.path.isfile(resolved):
        return
    _loaded_files.add(resolved)
    try:
        spec = importlib.util.spec_from_file_location("_promptmap_user_template_functions", resolved)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception:
        logging.getLogger("promptmap").exception("Failed to import %s", resolved)


# The built-in functions register through the same decorator users call.
from promptmap import org_outline as _org_outline  # noqa: E402,F401
