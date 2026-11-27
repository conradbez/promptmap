"""User Jinja template functions (@promptmap.template_function) and the built-in org_outline()."""

from __future__ import annotations

import pytest

import promptmap
from promptmap.executor.graph import build_models_from_dict
from promptmap.executor.parser_initial import parse_model_config
from promptmap.llm import try_load_client_module
from promptmap.template_functions import _REGISTRY, TemplateContext
from promptmap.org_outline import org_outline


def echo(prompt: str) -> str:
    return prompt


@pytest.fixture
def registry():
    """Remove any function a test registers."""
    before = dict(_REGISTRY)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(before)


def run(models: dict[str, str]) -> dict:
    return promptmap.run(models_from_dict=models, llm_call=echo, verbose=False)


# -- registry ---------------------------------------------------------------

def test_registered_function_renders_with_context_first(registry):
    @promptmap.template_function("shout")
    def shout(ctx, text, suffix=""):
        return f"{ctx.model}:{text.upper()}{suffix}"

    out = run({"m": "{{ shout('hi', suffix='!') }}"})
    assert out["m"] == "m:HI!"


def test_config_is_captured_when_a_function_is_called(registry):
    """An unknown name would cut config capture short; registered ones get stubs."""
    promptmap.register_template_function("boom", lambda ctx: "x")
    source = "{{ boom() }}\n{{ config(output_format='json') }}"
    assert parse_model_config(source)["output_format"] == "json"


@pytest.mark.parametrize("name", ["ref", "config", "promptdata", "rag_notes", "not valid"])
def test_reserved_and_invalid_names_are_rejected(registry, name):
    with pytest.raises(ValueError):
        promptmap.register_template_function(name, lambda ctx: "")


def test_project_file_beside_map_is_loaded(tmp_path, registry):
    (tmp_path / "map").mkdir()
    (tmp_path / "jinja_template_functions.py").write_text(
        "import promptmap\n"
        "@promptmap.template_function('from_project_file')\n"
        "def f(ctx):\n    return 'ok'\n"
    )
    try_load_client_module(str(tmp_path / "map"))
    assert "from_project_file" in promptmap.known_template_functions()


# -- org_outline ------------------------------------------------------------

PIPELINE = {
    "n1": "root one",
    "a": "{{ config(output_format='text') }}\nA body line\n\n{{ ref('n1') }}",
    "b": "{{ ref('n1') }}",
    "a1": "{{ ref('a') }} {{ ref('b') }}",
    "a2": "{{ ref('a') }} {{ ref('n1') }}",
    "n2": "two",
}


def outline(model: str, **kw) -> str:
    return org_outline(TemplateContext(model=model, specs=build_models_from_dict(PIPELINE)), **kw)


def test_outline_is_declarative_final_outputs_on_top():
    """Top bullets are models nothing refs; each model's children are what it refs."""
    assert outline("a2") == "\n".join([
        "a2 [you are here]",
        "- a",
        "  - n1",
        "a1 (also refs a)",
        "- b (also refs n1)",
        "n2",
    ])


def test_used_by_false_leaves_consumers_out():
    """up=0 still names hidden consumers; used_by=False drops them, so new consumers change nothing."""
    assert outline("a", up=0) == "a (used by a2, a1) [you are here]\n- n1"
    assert outline("a", up=0, used_by=False) == "a [you are here]\n- n1"


def test_top_bullets_have_no_dependents():
    specs = build_models_from_dict(PIPELINE)
    dag = TemplateContext(specs=specs).dag
    tops = [l.split()[0] for l in outline("a").splitlines() if not l.startswith((" ", "-"))]
    assert tops == ["a2", "a1", "n2"]
    assert all(dag.out_degree(t) == 0 for t in tops)


def test_ref_implied_by_another_ref_is_not_listed_as_extra():
    """a2 refs n1 too, but a (shown under a2) already builds on n1."""
    assert outline("a2").splitlines()[0] == "a2 [you are here]"


def test_closest_consumer_is_chosen():
    specs = build_models_from_dict({
        "top": "x", "mid": "{{ ref('top') }}", "leaf": "{{ ref('top') }} {{ ref('mid') }}",
    })
    assert org_outline(TemplateContext(model="top", specs=specs)) == (
        "leaf\n- mid\n  - top [you are here]"
    )


def test_up_limits_how_far_above_the_outline_starts():
    # up=0: this model and what it is built from; hidden consumers are named.
    assert outline("a", up=0) == "a (used by a2, a1) [you are here]\n- n1"
    assert outline("a1", up=0) == "\n".join([
        "a1 [you are here]",
        "- a",
        "  - n1",
        "- b (also refs n1)",
    ])
    # up=1 from n1: its consumers a, b, a2 and what they are built from.
    # b is not a final output — a1, which up hides, refs it.
    assert outline("n1", up=1) == "\n".join([
        "a2",
        "- a",
        "  - n1 [you are here]",
        "b (also refs n1; used by a1)",
    ])


def test_lines_shows_raw_template_without_config():
    text = outline("a", up=0, lines=2)
    assert text == "\n".join([
        "a (used by a2, a1) [you are here]",
        "  A body line",
        "  {{ ref('n1') }}",
        "- n1",
        "  root one",
    ])


def test_down_limits_how_far_below_the_outline_goes():
    # Refs cut off are named, so the model still knows what it builds on.
    assert outline("a1", up=0, down=1) == "\n".join([
        "a1 [you are here]",
        "- a (also refs n1)",
        "- b (also refs n1)",
    ])
    assert outline("a1", up=0, down=0) == "a1 (also refs a, b) [you are here]"


CHAIN = {
    "src": "x", "s1": "{{ ref('src') }}", "s2": "{{ ref('s1') }}", "s3": "{{ ref('s2') }}",
    "s4": "{{ ref('s3') }}", "report": "{{ ref('s4') }}",
    "side": "{{ ref('src') }}", "summary": "{{ ref('side') }}",
}


def chain_outline(model: str, **kw) -> str:
    return org_outline(TemplateContext(model=model, specs=build_models_from_dict(CHAIN)), **kw)


def test_zoom_adds_the_top_levels_and_marks_the_path_to_them():
    assert chain_outline("s2", up=0, down=1, zoom=1) == "\n".join([
        "report",
        "- s4 [on your path]",
        "  - s3 [on your path]",
        "    - s2 [you are here]",
        "      - s1 (also refs src)",
        "summary (also refs side)",
    ])


def test_zoom_marks_only_the_gap_between_overview_and_window():
    # zoom=2 already shows s4; up=1 already shows s3: nothing left to mark.
    assert chain_outline("s2", up=0, down=0, zoom=2) == "\n".join([
        "report",
        "- s4",
        "  - s3 [on your path]",
        "    - s2 (also refs s1) [you are here]",
        "summary",
        "- side (also refs src)",
    ])
    assert "[on your path]" not in chain_outline("s3", up=1, down=0, zoom=1)


def test_org_outline_in_a_real_run_sees_unselected_models():
    models = {**PIPELINE, "a2": "{{ ref('a') }}\n{{ org_outline() }}"}
    out = promptmap.run(models_from_dict=models, llm_call=echo, verbose=False, select=["a2"])
    assert "n2" in out["a2"] and "a2 [you are here]" in out["a2"]


def test_outside_a_run_org_outline_is_empty():
    assert org_outline(TemplateContext()) == ""
