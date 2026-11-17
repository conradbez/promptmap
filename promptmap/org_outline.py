"""
``org_outline()`` — where this model sits in the pipeline, as a declarative bullet tree.

    {{ org_outline() }}                 {# every model, names only #}
    {{ org_outline(up=1, lines=2) }}    {# from my consumers down, 2 lines each #}

The outline reads like a set of definitions: the top-level bullets are the
pipeline's final outputs — models nothing ref()s — and under each model are
the models it is built from.  For the pipeline ``n1 → a, b``, ``a, b → a1``,
``a, n1 → a2`` it prints, for ``a2``::

    a2 [you are here]
    - a
      - n1
    a1 (also refs a)
    - b (also refs n1)
    n2

The pipeline is a DAG, so a model several others ref() is listed once, under
the consumer closest to it — the one furthest from the final outputs.  That
leaves the fewest refs that point outside the tree: a model higher up that
also refs it usually reaches it through that consumer anyway.  A model's refs
the tree still cannot show are named in ``(also refs ...)``, leaving out the
ones another of its refs already builds on, which the outline implies.

Arguments
---------
up:
    How many levels above this model the outline starts, counting toward the
    final outputs.  ``None`` (default) shows the whole pipeline; ``0`` shows
    this model and what it is built from; ``1`` starts from the models that
    ref() it and includes everything they are built from.  A top bullet
    that is not a final output names the consumers ``up`` hides in
    ``(used by ...)``.
lines:
    How many lines of each model's raw template to show under its name —
    the first non-blank lines, with ``config()`` calls removed.  Default 0.
"""

from __future__ import annotations

from promptmap.template_functions import TemplateContext, template_function

YOU_ARE_HERE = "[you are here]"


@template_function("org_outline")
def org_outline(ctx: TemplateContext, up: int | None = None, lines: int = 0) -> str:
    import networkx as nx

    dag = ctx.dag  # edge A → B: B refs A
    me = ctx.model
    if me not in dag:
        return ""

    shown = _shown(dag, me, up)
    height = _heights(dag)
    order = {n: i for i, n in enumerate(nx.lexicographical_topological_sort(dag))}

    # Each model goes under the shown consumer furthest from the final
    # outputs; on a tie, the one earliest in the pipeline.
    parent: dict[str, str | None] = {}
    for node in shown:
        consumers = [c for c in dag.successors(node) if c in shown]
        parent[node] = (
            max(consumers, key=lambda c: (height[c], -order[c])) if consumers else None
        )

    children: dict[str | None, list[str]] = {}
    for node, p in parent.items():
        children.setdefault(p, []).append(node)
    for kids in children.values():
        kids.sort(key=lambda n: order[n])

    out: list[str] = []

    def walk(node: str, level: int) -> None:
        notes = []
        # Refs listed elsewhere in the tree — minus the ones another ref
        # already builds on, which the outline implies.
        refs = set(dag.predecessors(node))
        implied = set().union(*(nx.ancestors(dag, r) for r in refs))
        extra = sorted(refs - implied - set(children.get(node, [])), key=order.get)
        if extra:
            notes.append(f"also refs {', '.join(extra)}")
        # A top bullet `up` cut consumers off from is not a final output:
        # name them, minus those downstream of another one.
        hidden = {c for c in dag.successors(node) if c not in shown} if level == 0 else set()
        downstream = set().union(*(nx.descendants(dag, c) for c in hidden))
        used_by = sorted(hidden - downstream, key=order.get)
        if used_by:
            notes.append(f"used by {', '.join(used_by)}")

        label = node + (f" ({'; '.join(notes)})" if notes else "")
        if node == me:
            label += f" {YOU_ARE_HERE}"
        out.append(("  " * (level - 1) + "- " if level else "") + label)
        for line in ctx.body_lines(node, lines):
            out.append("  " * max(level, 1) + line)
        for child in children.get(node, []):
            walk(child, level + 1)

    for top in children.get(None, []):
        walk(top, 0)
    return "\n".join(out)


def _shown(dag, me: str, up: int | None) -> set[str]:
    """The models in the outline: all of them, or everything *up* consumer-levels above *me* is built from."""
    import networkx as nx

    if up is None:
        return set(dag.nodes)
    tops = {me}
    frontier = {me}
    for _ in range(max(up, 0)):
        frontier = {c for n in frontier for c in dag.successors(n)}
        tops |= frontier
    shown = set(tops)
    for top in tops:
        shown |= nx.ancestors(dag, top)
    return shown


def _heights(dag) -> dict[str, int]:
    """Longest path from each node down to a final output (a model nothing refs)."""
    import networkx as nx

    height: dict[str, int] = {}
    for node in reversed(list(nx.topological_sort(dag))):
        height[node] = max((height[c] + 1 for c in dag.successors(node)), default=0)
    return height
