"""Semantic diffs: comments and formatting are not changes; symbols are reported."""

from __future__ import annotations

import pytest

from promptmap_extra_model_kinds import _ast_diff as ast


@pytest.fixture(autouse=True)
def grammars():
    """Grammars download on first use; without them these tests mean nothing."""
    pack = ast._pack()
    try:
        pack.get_parser("python")
        pack.get_parser("yaml")
    except Exception as exc:  # offline
        pytest.skip(f"tree-sitter grammars unavailable: {exc}")


OLD = b"import os\nclass A:\n    def m(self, x):\n        return x  # hi\ndef f(a):\n    return a\n"


def status(path, old, new):
    return ast.diff_file(path, old, new)


def test_comments_and_formatting_are_format_only():
    new = b"import os\n\nclass A:\n    def m(self,x):\n        # changed comment\n        return x\ndef f(a):\n    return a\n"
    assert status("x.py", OLD, new).status == "format_only"


def test_reindenting_a_whole_block_is_format_only():
    a = b"def f(a):\n    if a:\n        b()\n    c()\n"
    b = b"def f(a):\n  if a:\n    b()\n  c()\n"
    assert status("x.py", a, b).status == "format_only"


def test_indentation_that_moves_code_is_a_change():
    a = b"def f(a):\n    if a:\n        b()\n    c()\n"
    b = b"def f(a):\n    if a:\n        b()\n        c()\n"
    change = status("x.py", a, b)
    assert change.status == "modified"
    assert change.details == ["BODY_CHANGED Function f"]


def test_directive_comments_count():
    new = OLD.replace(b"# hi", b"# type: ignore")
    assert status("x.py", OLD, new).status == "modified"


def test_string_whitespace_counts():
    assert status("x.py", b's = "a  b"\n', b's = "a b"\n').semantic


def test_symbol_changes_are_named():
    new = b"import os\nclass A:\n    def m(self, x, y):\n        return x\ndef f(a):\n    return a+1\ndef g(): pass\n"
    details = status("x.py", OLD, new).details
    assert "SIGNATURE_CHANGED A.m: def m(self, x): -> def m(self, x, y):" in details
    assert "BODY_CHANGED Function f" in details
    assert "ADDED_SYMBOL Function g: def g():" in details
    assert not any(d.startswith("BODY_CHANGED Class A") for d in details)  # explained by A.m


def test_removed_symbol_and_module_change():
    removed = status("x.py", OLD, b"import os\nclass A:\n    def m(self, x):\n        return x\n")
    assert removed.details == ["REMOVED_SYMBOL Function f: def f(a):"]
    module = status("x.py", OLD, OLD.replace(b"import os", b"import sys"))
    assert module.details[0].startswith("MODULE_CHANGED")


def test_yaml_comments_ignored_but_nesting_counts():
    assert status("c.yaml", b"a: 1 # c\nb: 2\n", b"a:   1\nb: 2\n").status == "format_only"
    assert status("c.yaml", b"a: 1\nb:\n  c: 2\n", b"a: 1\nb:\nc: 2\n").semantic


def test_fallbacks():
    assert status("notes.unknownext", b"a", b"b").details == ["TEXT_CHANGED"]
    assert status("x.py", b"def f(:\n", b"def f( :\n").details == ["TEXT_CHANGED"]  # does not parse
    assert status("img.png", b"\x89PNG\xff", b"\x89PNG\xfe").status == "binary"
    assert status("x.py", None, OLD).lines() == ["x.py: ADDED_FILE", "x.py: symbols: A, A.m, f"]
    assert status("x.py", OLD, None).status == "deleted"


def test_without_the_pack_everything_is_text(monkeypatch):
    monkeypatch.setattr(ast, "_pack", lambda: None)
    new = OLD.replace(b"# hi", b"# bye")
    assert status("x.py", OLD, new).details == ["TEXT_CHANGED"]


def test_summary_lines():
    changes = ast.diff_trees(["a.py", "b.py"], {"a.py": None, "b.py": OLD}.get, {"a.py": OLD, "b.py": OLD}.get)
    assert ast.summary(changes) == ["a.py: ADDED_FILE", "a.py: symbols: A, A.m, f"]
