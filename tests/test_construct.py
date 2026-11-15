"""file_to_construct / construct_to_files, config() rewriting, and `promptmap edit`."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from promptmap.cli import main
from promptmap.construct import (
    construct_to_files,
    file_to_construct,
    read_config_call,
    write_config_call,
)
from promptmap.executor.graph import load_models


def _project(tmp_path: Path) -> Path:
    (tmp_path / "map" / "1_stage").mkdir(parents=True)
    (tmp_path / "map" / "topic.prompt").write_text(
        '{{ config(model_type="template") }}\n{{ promptdata("topic") }}\n', encoding="utf-8"
    )
    (tmp_path / "map" / "1_stage" / "1_a_article.prompt").write_text(
        "Write about {{ ref('topic') }} for {{ promptdata('audience') }}\n", encoding="utf-8"
    )
    (tmp_path / "map" / "article_ok.testprompt").write_text("{{ ref('article') }}\n", encoding="utf-8")
    (tmp_path / "promptdata.yml").write_text("# my header\ntopic: cats\naudience: ''\n", encoding="utf-8")
    return tmp_path


def test_round_trip_reads_everything_and_writes_nothing(tmp_path: Path) -> None:
    root = _project(tmp_path)
    c = file_to_construct(root)

    assert set(c.models) == {"topic", "article"}
    assert c.models["article"].path == "1_stage/1_a_article.prompt"
    assert set(c.tests) == {"article_ok"}
    assert c.promptdata == {"topic": "cats", "audience": ""}
    assert c.run_promptdata() == {"topic": "cats"}
    assert c.expected_promptdata() == ["audience", "topic"]  # file order: 1_stage/ sorts first
    assert construct_to_files(c) == []


def test_specs_match_load_models(tmp_path: Path) -> None:
    root = _project(tmp_path)
    from_construct = file_to_construct(root).specs()
    from_loader = load_models(root / "map")
    assert from_construct.keys() == from_loader.keys()
    for name in from_loader:
        assert from_construct[name].path == from_loader[name].path
        assert from_construct[name].depends_on == from_loader[name].depends_on


def test_edits_write_only_changed_files_and_keep_header(tmp_path: Path) -> None:
    root = _project(tmp_path)
    c = file_to_construct(root)
    c.promptdata["audience"] = "kids"
    c.set_config("article", model="low")

    changed = construct_to_files(c)

    assert sorted(p.name for p in changed) == ["1_a_article.prompt", "promptdata.yml"]
    text = (root / "promptdata.yml").read_text(encoding="utf-8")
    assert text.startswith("# my header\n")
    assert file_to_construct(root).promptdata == {"topic": "cats", "audience": "kids"}
    assert file_to_construct(root).get_config("article") == {"model": "low"}


def test_write_config_call_keeps_other_keys_and_order() -> None:
    src = '{{ config(model_type="template", each=ref_path) }}\nbody\n'
    out = write_config_call(src, {"model_type": None, "cache": False, "promptfiles": ["a"]})
    assert out == '{{ config(each=ref_path, cache=False, promptfiles=["a"]) }}\nbody\n'
    assert read_config_call(out) == {"each": "ref_path", "cache": False, "promptfiles": ["a"]}


def test_write_config_call_removes_empty_call_and_adds_new_one() -> None:
    assert write_config_call('{{ config(model="x") }}\nbody\n', {"model": None}) == "body\n"
    assert write_config_call("body\n", {"model": "x"}) == '{{ config(model="x") }}\nbody\n'


def test_cli_edit_flags(tmp_path: Path, monkeypatch) -> None:
    root = _project(tmp_path)
    monkeypatch.chdir(root)
    runner = CliRunner()

    r = runner.invoke(main, ["edit", "article", "--set", "model=low", "--set", "cache=false",
                             "--promptdata", "audience=kids"])
    assert r.exit_code == 0, r.output
    c = file_to_construct(root)
    assert c.get_config("article") == {"model": "low", "cache": False}
    assert c.promptdata["audience"] == "kids"

    r = runner.invoke(main, ["edit", "article", "--unset", "cache"])
    assert r.exit_code == 0, r.output
    assert file_to_construct(root).get_config("article") == {"model": "low"}


def test_cli_edit_interactive_promptdata(tmp_path: Path, monkeypatch) -> None:
    root = _project(tmp_path)
    monkeypatch.chdir(root)
    r = CliRunner().invoke(main, ["edit"], input="adults\n\n")  # audience asked first, then keep topic
    assert r.exit_code == 0, r.output
    assert file_to_construct(root).promptdata == {"topic": "cats", "audience": "adults"}


def test_cli_edit_interactive_model_config(tmp_path: Path, monkeypatch) -> None:
    root = _project(tmp_path)
    monkeypatch.chdir(root)
    # Set output_format=json, finish, decline template edit.
    r = CliRunner().invoke(main, ["edit", "article"], input="output_format\njson\n\nn\n")
    assert r.exit_code == 0, r.output
    assert "output_format" in r.output  # offered as a known key
    assert file_to_construct(root).get_config("article") == {"output_format": "json"}


def test_cli_edit_unknown_model(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(_project(tmp_path))
    r = CliRunner().invoke(main, ["edit", "nope"])
    assert r.exit_code == 1


def test_json_round_trip_in_memory_then_to_files(tmp_path: Path) -> None:
    from promptmap.construct import export_to_json, import_from_json

    src = file_to_construct(_project(tmp_path / "a"))
    text = export_to_json(src)
    assert not (tmp_path / "b").exists()

    c = import_from_json(text, root=tmp_path / "b")
    assert not (tmp_path / "b").exists()  # import writes nothing
    assert c.models == src.models and c.tests == src.tests and c.promptdata == src.promptdata
    assert set(c.specs()) == {"topic", "article"}  # usable without touching disk

    c.set_config("article", model="low")
    construct_to_files(c)
    back = file_to_construct(tmp_path / "b")
    assert back.get_config("article") == {"model": "low"}
    assert (tmp_path / "b" / "map" / "1_stage" / "1_a_article.prompt").is_file()


def test_import_from_json_shorthand_and_errors() -> None:
    import pytest

    from promptmap.construct import import_from_json

    c = import_from_json({"models": {"topic": "hi"}})
    assert c.models["topic"].path == "topic.prompt"
    with pytest.raises(ValueError):
        import_from_json({"version": 99})
    with pytest.raises(ValueError):
        import_from_json({"models": {"x": {"path": "x.prompt"}}})
