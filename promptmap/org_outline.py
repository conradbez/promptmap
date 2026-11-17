"""
``org_outline()`` — where this model sits in the pipeline, as a declarative bullet tree.

    {{ org_outline() }}                       {# every model, names only #}
    {{ org_outline(up=1, down=1) }}           {# a window around me #}
    {{ org_outline(up=0, down=1, zoom=1) }}   {# window + the final outputs #}
    {{ org_outline(up=1, lines=2) }}          {# 2 template lines each #}

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
the tree does not show under it are named in ``(also refs ...)``, leaving out
the ones another of its refs already builds on, which the outline implies.

Arguments
---------
up:
    How many levels of consumers above this model to show, toward the final
    outputs.  ``None`` (default) shows them all.  A top bullet that is not a
    final output names the consumers it hides in ``(used by ...)``.
down:
    How many levels below this model to show — what it is built from.
    ``None`` (default) shows them all; with ``up`` also ``None``, the whole
    pipeline is shown, unconnected models too.  A consumer ``up`` brings in is shown
    down to the same level, so this model's siblings appear too.  Refs cut
    off are named in ``(also refs ...)``.
zoom:
    Also show the top *zoom* levels of the whole pipeline — the final
    outputs and what they are directly built from — for the big picture.
    When that and the ``up``/``down`` window do not join, the chain of models
    linking a final output down to this model is added too, each marked
    ``[on your path]``.  ``None`` (default) adds nothing.
lines:
    How many lines of each model's raw template to show under its name —
    the first non-blank lines, with ``config()`` calls removed.  Default 0.
"""

from __future__ import annotations

from promptmap.template_functions import TemplateContext, template_function

YOU_ARE_HERE = "[you are here]"
ON_YOUR_PATH = "[on your path]"


@template_function("org_outline")
def org_outline(
    ctx: TemplateContext,
    up: int | None = None,
    down: int | None = None,
    zoom: int | None = None,
    lines: int = 0,
) -> str:
    import networkx as nx

    dag = ctx.dag  # edge A → B: B refs A
    me = ctx.model
    if me not in dag:
        return ""

    height = _heights(dag)
    order = {n: i for i, n in enumerate(nx.lexicographical_topological_sort(dag))}

    def closest(consumers):
        """The consumer furthest from the final outputs; on a tie, earliest in the pipeline."""
        return max(consumers, key=lambda c: (height[c], -order[c]))

    shown = _window(dag, me, up, down)
    path: set[str] = set()
    if zoom is not None and zoom > 0:
        overview = {n for n in dag if height[n] < zoom}
        # Each step to the closest consumer brings the height down by exactly
        # one, so this chain reaches a final output — and the overview.
        node = me
        while (consumers := list(dag.successors(node))):
            node = closest(consumers)
            if node not in shown and node not in overview:
                path.add(node)
        shown |= overview | path

    parent: dict[str, str | None] = {}
    for node in shown:
        consumers = [c for c in dag.successors(node) if c in shown]
        parent[node] = closest(consumers) if consumers else None

    children: dict[str | None, list[str]] = {}
    for node, p in parent.items():
        children.setdefault(p, []).append(node)
    for kids in children.values():
        kids.sort(key=lambda n: order[n])

    out: list[str] = []

    def walk(node: str, level: int) -> None:
        notes = []
        # Refs not shown under this model — minus the ones another ref
        # already builds on, which the outline implies.
        refs = set(dag.predecessors(node))
        implied = set().union(*(nx.ancestors(dag, r) for r in refs))
        extra = sorted(refs - implied - set(children.get(node, [])), key=order.get)
        if extra:
            notes.append(f"also refs {', '.join(extra)}")
        # A top bullet with hidden consumers is not a final output: name
        # them, minus those downstream of another one.
        hidden = {c for c in dag.successors(node) if c not in shown} if level == 0 else set()
        downstream = set().union(*(nx.descendants(dag, c) for c in hidden))
        used_by = sorted(hidden - downstream, key=order.get)
        if used_by:
            notes.append(f"used by {', '.join(used_by)}")

        label = node + (f" ({'; '.join(notes)})" if notes else "")
        if node == me:
            label += f" {YOU_ARE_HERE}"
        elif node in path:
            label += f" {ON_YOUR_PATH}"
        out.append(("  " * (level - 1) + "- " if level else "") + label)
        for line in ctx.body_lines(node, lines):
            out.append("  " * max(level, 1) + line)
        for child in children.get(node, []):
            walk(child, level + 1)

    for top in children.get(None, []):
        walk(top, 0)
    return "\n".join(out)


def _window(dag, me: str, up: int | None, down: int | None) -> set[str]:
    """*me*, its consumers *up* levels above, and what they are built from *down* levels below *me*.

    With neither limit set, every model — including ones unconnected to *me*.
    """
    if up is None and down is None:
        return set(dag.nodes)
    # Consumers by distance above me (the shortest one).
    above = {me: 0}
    frontier = {me}
    k = 0
    while frontier and (up is None or k < up):
        k += 1
        frontier = {c for n in frontier for c in dag.successors(n)} - above.keys()
        above.update((c, k) for c in frontier)

    shown = set(above)
    for top, k in above.items():
        reach = None if down is None else k + down
        frontier, d = {top}, 0
        while frontier and (reach is None or d < reach):
            d += 1
            frontier = {p for n in frontier for p in dag.predecessors(n)} - shown
            shown |= frontier
    return shown


def _heights(dag) -> dict[str, int]:
    """Longest path from each node down to a final output (a model nothing refs)."""
    import networkx as nx

    height: dict[str, int] = {}
    for node in reversed(list(nx.topological_sort(dag))):
        height[node] = max((height[c] + 1 for c in dag.successors(node)), default=0)
    return height
