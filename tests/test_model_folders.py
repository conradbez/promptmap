"""Models in stage folders: the name is the file name, the folder is ignored."""

from __future__ import annotations

from pathlib import Path

import pytest

from promptmap.executor.graph import load_models


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_name_is_the_file_name_without_suffix(tmp_path: Path) -> None:
    write(tmp_path / "1_design" / "brief.prompt", "Brief")
    write(tmp_path / "2_build" / "parts.prompt", "{{ ref('brief') }}")
    write(tmp_path / "3_qa" / "review.prompt.jinja", "{{ ref('parts') }}")
    write(tmp_path / "1_design" / "1_a_report.prompt", "kept as is")

    models = load_models(tmp_path)

    assert set(models) == {"brief", "parts", "review", "1_a_report"}
    assert models["parts"].depends_on == ["brief"]


def test_same_name_in_two_folders_is_a_duplicate(tmp_path: Path) -> None:
    write(tmp_path / "a" / "parts.prompt", "x")
    write(tmp_path / "b" / "parts.prompt", "y")
    with pytest.raises(ValueError, match="Duplicate model name 'parts'"):
        load_models(tmp_path)


# -- folder inheritance ------------------------------------------------------

def _tree(tmp_path: Path) -> None:
    write(tmp_path / "intro.prompt", "Intro")
    write(tmp_path / "node_1" / "node_1.prompt", "Summarise:")
    write(tmp_path / "node_1" / "research.prompt", "R")
    write(tmp_path / "node_1" / "critique.prompt", "C")
    write(tmp_path / "node_1" / "node_a" / "node_a.prompt", "A")
    write(tmp_path / "node_1" / "node_a" / "fact_1.prompt", "F1")
    write(tmp_path / "node_1" / "node_a" / "fact_2.prompt", "F2")
    write(tmp_path / "final.prompt", "{{ ref('node_1') }}")


def test_folder_model_refs_its_direct_children(tmp_path: Path) -> None:
    _tree(tmp_path)
    models = load_models(tmp_path)

    assert models["node_1"].depends_on == ["critique", "research", "node_a"]
    assert models["node_a"].depends_on == ["fact_1", "fact_2"]
    assert models["final"].depends_on == ["node_1"]
    assert models["intro"].depends_on == []
    assert models["node_1"].source.endswith(
        "Summarise:\n\n{{ ref('critique') }}\n{{ ref('research') }}\n{{ ref('node_a') }}\n"
    )


def test_children_already_reffed_are_not_appended_again(tmp_path: Path) -> None:
    write(tmp_path / "n" / "n.prompt", "Lead: {{ ref('a') }}")
    write(tmp_path / "n" / "a.prompt", "A")
    write(tmp_path / "n" / "b.prompt", "B")
    models = load_models(tmp_path)

    assert models["n"].depends_on == ["a", "b"]
    assert models["n"].source.count("ref('a')") == 1


def test_folder_model_may_sit_next_to_its_folder(tmp_path: Path) -> None:
    write(tmp_path / "file1.prompt", "Sum up:")
    write(tmp_path / "file1" / "filea.prompt", "A")
    write(tmp_path / "file1" / "fileb.prompt", "B")
    write(tmp_path / "file1" / "sub.prompt", "S")
    write(tmp_path / "file1" / "sub" / "x.prompt", "X")
    write(tmp_path / "top" / "top.prompt", "Top:")
    write(tmp_path / "top" / "file2.prompt", "F2")
    write(tmp_path / "top" / "file2" / "y.prompt", "Y")
    models = load_models(tmp_path)

    assert models["file1"].depends_on == ["filea", "fileb", "sub"]
    assert models["sub"].depends_on == ["x"]
    # A sibling folder model counts as its folder inside the folder above.
    assert models["top"].depends_on == ["file2"]
    assert models["file2"].depends_on == ["y"]


def test_children_that_ref_the_folder_model_are_downstream(tmp_path: Path) -> None:
    # pm-code: a chat's nodes live in a folder named after it and ref() it.
    write(tmp_path / "chat.prompt", "Plan it.")
    write(tmp_path / "chat" / "agent_a.prompt", "Do A. {{ ref('chat') }}")
    write(tmp_path / "chat" / "agent_b.prompt", "Do B after {{ ref('agent_a') }}")
    write(tmp_path / "chat" / "notes.prompt", "Notes")
    models = load_models(tmp_path)

    assert models["chat"].depends_on == ["notes"]
    assert models["agent_a"].depends_on == ["chat"]


def test_folder_without_a_same_name_file_adds_nothing(tmp_path: Path) -> None:
    write(tmp_path / "stage" / "a.prompt", "A")
    write(tmp_path / "stage" / "b.prompt", "B")
    models = load_models(tmp_path)
    assert models["a"].depends_on == [] and models["b"].depends_on == []


def test_folder_inheritance_leaves_files_on_disk_alone(tmp_path: Path) -> None:
    from promptmap.construct import construct_to_files, file_to_construct

    _tree(tmp_path / "map")
    c = file_to_construct(tmp_path)
    c.specs()
    construct_to_files(c)
    assert (tmp_path / "map" / "node_1" / "node_1.prompt").read_text() == "Summarise:"


def test_folder_model_renders_children_wrapped(tmp_path: Path) -> None:
    import promptmap
    from promptmap.storage import MemoryStorageBackend

    _tree(tmp_path)
    results = promptmap.run(str(tmp_path), llm_call=lambda prompt: prompt, verbose=False,
                            storage_backend=MemoryStorageBackend())

    assert "[node_a]\nA\n\n[fact_1]\nF1\n[/fact_1]" in results["node_1"]
    assert "[research]\nR\n[/research]" in results["node_1"]
    assert results["final"].startswith("[node_1]\nSummarise:")
