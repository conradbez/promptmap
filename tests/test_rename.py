"""`promptmap rename OLD NEW`: file, references and run history move together."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from promptmap.cli import main
from promptmap.rename import RenameError, apply_rename, plan_rename, rewrite_source, rewrite_yaml


def test_rewrite_source_renames_refs_and_config_values_only():
    text = (
        '{{ config(promptfiles=["logo", "fix.result"], each="fix.items[*]") }}\n'
        "Use {{ ref('fix') }} and {{ ref(\"fix.key\") }} but not {{ ref('fixer') }}.\n"
        "Say 'fix' in prose; model_type='fix' is untouched outside promptfiles/each.\n"
    )
    out = rewrite_source(text, "fix", "repair")
    assert '{{ config(promptfiles=["logo", "repair.result"], each="repair.items[*]") }}' in out
    assert "ref('repair')" in out and 'ref("repair.key")' in out and "ref('fixer')" in out
    assert "Say 'fix' in prose" in out


def test_rewrite_yaml_renames_given_and_expect_keys_only():
    text = (
        "baselines:\n"
        "  default:\n"
        "    promptdata:\n"
        "      fix: keep me\n"
        "cases:\n"
        "  - name: One\n"
        "    given:\n"
        "      # a comment\n"
        "      fix: {output: 1}\n"
        "      other: 2\n"
        "    expect: {fix: {ok: true}, other: 1}\n"
    )
    out = rewrite_yaml(text, "fix", "repair")
    assert "      fix: keep me\n" in out
    assert "      repair: {output: 1}\n" in out and "# a comment" in out
    assert "expect: {repair: {ok: true}, other: 1}" in out


@pytest.fixture
def project(tmp_path, monkeypatch):
    (tmp_path / "map" / "2_build").mkdir(parents=True)
    (tmp_path / "map" / "2_build" / "2_a_draft.prompt").write_text("Write a haiku")
    (tmp_path / "map" / "short.prompt").write_text("Shorten: {{ ref('draft') }}")
    (tmp_path / "map" / "draft_ok.testprompt").write_text("{{ config(gate=False) }}Is {{ ref('draft') }} a haiku?")
    (tmp_path / "validation").mkdir()
    (tmp_path / "validation" / "draft.py").write_text("def validate(p, o):\n    return o\n")
    (tmp_path / "client.py").write_text("def llm_call(prompt):\n    return 'answer: ' + prompt[:30]\n")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_plan_keeps_folder_and_prefix_and_moves_the_validator(project):
    plan = plan_rename("map", "draft", "poem")
    moves = {str(a.relative_to(project.resolve()) if a.is_absolute() else a): str(b.name) for a, b in plan.moves}
    assert moves == {"map/2_build/2_a_draft.prompt": "2_a_poem.prompt", "validation/draft.py": "poem.py"}
    assert {p.name for p in plan.edits} == {"short.prompt", "draft_ok.testprompt"}
    assert set(plan.retemplate) == {"short"}

    apply_rename(plan)
    assert (project / "map" / "2_build" / "2_a_poem.prompt").read_text() == "Write a haiku"
    assert (project / "map" / "short.prompt").read_text() == "Shorten: {{ ref('poem') }}"
    assert (project / "validation" / "poem.py").exists()


@pytest.mark.parametrize("old,new,msg", [
    ("nope", "x", "No model named 'nope'"),
    ("draft", "short", "already exists"),
    ("draft", "bad name", "not a valid model name"),
    ("draft", "draft", "same"),
])
def test_plan_refuses_bad_renames(project, old, new, msg):
    with pytest.raises(RenameError, match=msg):
        plan_rename("map", old, new)


def test_cli_rename_keeps_the_cache(project):
    run = lambda: CliRunner().invoke(main, ["run", "--no-color"])  # noqa: E731
    first = run()
    assert first.exit_code == 0, first.output

    dry = CliRunner().invoke(main, ["rename", "draft", "poem", "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert "Would move" in dry.output and (project / "map" / "2_build" / "2_a_draft.prompt").exists()

    done = CliRunner().invoke(main, ["rename", "draft", "poem"])
    assert done.exit_code == 0, done.output
    assert "Renamed draft → poem (1 stored result moved)" in done.output

    again = run()
    assert again.exit_code == 0, again.output
    assert "poem … OK" in again.output and again.output.count("· cached") == 2

    # NEW has stored results but no file (a deleted model): refuse unless --force.
    (project / "map" / "ghost.prompt").write_text("Boo")
    assert run().exit_code == 0
    (project / "map" / "ghost.prompt").unlink()
    refused = CliRunner().invoke(main, ["rename", "short", "ghost"])
    assert refused.exit_code != 0 and "--force" in refused.output
    assert CliRunner().invoke(main, ["rename", "short", "ghost", "--force"]).exit_code == 0
