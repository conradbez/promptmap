"""Named RAG kinds: @promptmap.rag_kind("x") → rag_x(...) in templates."""

from __future__ import annotations

import importlib.metadata
import shutil
from pathlib import Path

import pytest

import promptmap
from promptmap import rag
from promptmap.executor.parser_initial import parse_model_config
from promptmap.executor.parser_model import render_prompt

EXAMPLE = Path(__file__).parent.parent / "examples" / "rag_kinds"


@pytest.fixture(autouse=True)
def clean_registry():
    before = dict(rag._REGISTRY)
    yield
    rag._REGISTRY.clear()
    rag._REGISTRY.update(before)


def test_registered_rag_is_callable_with_args_and_kwargs():
    @promptmap.rag_kind("echo_test")
    def echo(query, k=1):
        return [f"{query}:{i}" for i in range(k)]

    rendered, _ = render_prompt("{{ rag_echo_test('q', k=2) | join(',') }}", {})
    assert rendered == "q:0,q:1"
    assert echo("x") == ["x:0"]  # decorator returns the function unchanged


@pytest.mark.parametrize("result, expected", [("one", "['one']"), (None, "[False]"), (False, "[False]")])
def test_results_are_normalised_to_a_list(result, expected):
    promptmap.register_rag_kind("norm_test", lambda *a: result)
    rendered, _ = render_prompt("{{ rag_norm_test('q') }}", {})
    assert rendered == expected


def test_unregistered_rag_raises_only_when_called():
    rendered, _ = render_prompt("{% if false %}{{ rag_nope_test('q') }}{% endif %}ok", {})
    assert rendered == "ok"
    with pytest.raises(NameError, match=r"rag_nope_test\(\) is not a registered RAG"):
        render_prompt("{{ rag_nope_test('q') }}", {})


def test_unregistered_rag_does_not_cut_config_parsing_short():
    config = parse_model_config("{{ rag_nope_test('q') }}{{ config(output_format='json') }}")
    assert config["output_format"] == "json"


def test_invalid_name_is_rejected():
    with pytest.raises(ValueError):
        promptmap.register_rag_kind("not-valid", lambda q: q)


def test_entry_point_plugins_are_loaded(monkeypatch):
    class EP:
        name = "fake"

        def load(self):
            return lambda: promptmap.register_rag_kind("plugin_rag_test", lambda q: [q])

    groups = []

    def fake_entry_points(*, group):
        groups.append(group)
        return [EP()]

    monkeypatch.delenv("PROMPTMAP_DISABLE_PLUGINS", raising=False)
    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)
    monkeypatch.setattr(rag, "_plugins_loaded", False)

    assert "plugin_rag_test" in promptmap.known_rag_kinds()
    assert groups == [rag.ENTRY_POINT_GROUP]


def test_project_rag_py_registers_without_do_rag(tmp_path):
    (tmp_path / "map").mkdir()
    (tmp_path / "rag.py").write_text(
        "import promptmap\n"
        "@promptmap.rag_kind('local_test')\n"
        "def f(q):\n"
        "    return q.upper()\n"
    )
    legacy = rag.resolve_rag_call(str(tmp_path / "map"))

    assert promptmap.get_rag_kind("local_test")("hi") == "HI"
    with pytest.raises(RuntimeError, match="no do_RAG"):
        legacy("q")


def test_example_notes_search_end_to_end(tmp_path):
    project = tmp_path / "rag_kinds"
    shutil.copytree(EXAMPLE, project)
    rag.resolve_rag_call(str(project / "map"))  # imports rag.py, registering rag_notes
    source = (project / "map" / "answer.prompt").read_text()
    rendered, _ = render_prompt(source, {"question": "How do I roll back a deploy?"}, promptdata={})

    assert "[deploys.md]" in rendered
    assert "make rollback" in rendered
    assert "Web results" not in rendered
