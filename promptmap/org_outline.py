"""
``org_outline()`` — where this model sits in the pipeline, as a bullet tree.

    {{ org_outline() }}                 {# every model, names only #}
    {{ org_outline(up=1, lines=2) }}    {# from my parents down, 2 lines each #}

prints, for a model ``1a_2``::

    1
    - a
      - 1a_1 (also refs b)
      - 1a_2 [you are here]
    - b
    2

The pipeline is a DAG, so a model with several parents is listed once, under
the parent that leaves the fewest *out-of-tree* refs — refs to a model that is
not already above it in the outline.  The deepest parent is chosen: its own
ancestors cover the most of the other parents.  The refs the tree still cannot
show — including refs to models ``up`` leaves out — are named in
``(also refs ...)``.

Arguments
---------
up:
    How many levels above this model the outline starts.  ``None`` (default)
    shows the whole pipeline; ``0`` shows this model and what it feeds; ``1``
    starts from its parents and includes their other children.
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

    dag = ctx.dag
    me = ctx.model
    if me not in dag:
        return ""

    shown = _shown(dag, me, up)
    depth = _depths(dag)

    order = {n: i for i, n in enumerate(nx.lexicographical_topological_sort(dag))}

    # Deepest parent wins; on a tie, the one earliest in the pipeline.
    parent: dict[str, str | None] = {}
    for node in shown:
        preds = [p for p in dag.predecessors(node) if p in shown]
        parent[node] = max(preds, key=lambda p: (depth[p], -order[p])) if preds else None

    children: dict[str | None, list[str]] = {}
    for node, p in parent.items():
        children.setdefault(p, []).append(node)
    for kids in children.values():
        kids.sort(key=lambda n: order[n])

    out: list[str] = []

    def walk(node: str, level: int) -> None:
        # Refs the tree does not show, shown or cut by `up` — minus the ones a
        # parent already builds on, which the outline implies.
        preds = set(dag.predecessors(node))
        bases = [parent[node]] if parent[node] else preds
        implied = set().union(*(nx.ancestors(dag, b) for b in bases))
        extra = sorted(preds - implied - {parent[node]})
        label = node
        if extra:
            label += f" (also refs {', '.join(extra)})"
        if node == me:
            label += f" {YOU_ARE_HERE}"
        out.append(("  " * (level - 1) + "- " if level else "") + label)
        for line in ctx.body_lines(node, lines):
            out.append("  " * max(level, 1) + line)
        for child in children.get(node, []):
            walk(child, level + 1)

    for root in children.get(None, []):
        walk(root, 0)
    return "\n".join(out)


def _shown(dag, me: str, up: int | None) -> set[str]:
    """The models in the outline: all of them, or the subtree *up* levels above *me*."""
    import networkx as nx

    if up is None:
        return set(dag.nodes)
    tops = {me}
    frontier = {me}
    for _ in range(max(up, 0)):
        frontier = {p for n in frontier for p in dag.predecessors(n)}
        tops |= frontier
    shown = set(tops)
    for top in tops:
        shown |= nx.descendants(dag, top)
    return shown


def _depths(dag) -> dict[str, int]:
    """Longest path from a root to each node."""
    import networkx as nx

    depth: dict[str, int] = {}
    for node in nx.topological_sort(dag):
        depth[node] = max((depth[p] + 1 for p in dag.predecessors(node)), default=0)
    return depth
