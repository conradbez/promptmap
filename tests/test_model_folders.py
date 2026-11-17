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
