"""rag_gemini_search from promptmap-extra-model-kinds, against a mocked Gemini response."""

from __future__ import annotations

import importlib

import httpx
import pytest

from promptmap import model_types, rag

RESPONSE = {
    "candidates": [
        {
            "content": {"parts": [{"text": "- Python 3.14 is the latest "}, {"text": "stable release."}]},
            "groundingMetadata": {
                "groundingChunks": [
                    {"web": {"uri": "https://example.com/a", "title": "python.org"}},
                    {"web": {"uri": "https://example.com/b", "title": "Release notes"}},
                    {"retrievedContext": {}},
                ]
            },
        }
    ]
}


@pytest.fixture
def gemini(monkeypatch):
    """The module, imported here so the extras kinds it registers never leak into other tests."""
    monkeypatch.setattr(model_types, "_REGISTRY", dict(model_types._REGISTRY))
    monkeypatch.setattr(rag, "_REGISTRY", dict(rag._REGISTRY))
    try:
        return importlib.import_module("promptmap_extra_model_kinds.gemini_search")
    except ImportError:
        pytest.skip("promptmap-extra-model-kinds is not installed")


@pytest.fixture
def sent(monkeypatch, gemini):
    calls = []

    def fake_post(url, *, headers, json, timeout):
        calls.append({"url": url, "headers": headers, "json": json})
        return httpx.Response(200, json=RESPONSE)

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_SEARCH_MODEL", raising=False)
    monkeypatch.setattr(gemini.httpx, "post", fake_post)
    return calls


def test_answer_then_sources(gemini, sent):
    result = gemini.gemini_search("[topic]\nlatest python\n[/topic]")

    assert result == [
        "- Python 3.14 is the latest stable release.",
        "python.org\nhttps://example.com/a",
        "Release notes\nhttps://example.com/b",
    ]
    [call] = sent
    assert call["headers"] == {"x-goog-api-key": "test-key"}
    assert call["json"]["tools"] == [{"google_search": {}}]
    assert call["json"]["contents"][0]["parts"][0]["text"].endswith("Query: latest python")
    assert gemini.DEFAULT_MODEL in call["url"]


def test_model_override(gemini, sent, monkeypatch):
    monkeypatch.setenv("GEMINI_SEARCH_MODEL", "env-model")
    gemini.gemini_search("q")
    gemini.gemini_search("q", model="arg-model")

    assert "/env-model:" in sent[0]["url"]
    assert "/arg-model:" in sent[1]["url"]


def test_empty_query_skips_the_call(gemini, sent):
    assert gemini.gemini_search("[topic]\n[/topic]") == ["[Search skipped: the query was empty.]"]
    assert sent == []


def test_key_is_read_from_dotenv(gemini, sent, monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY")
    (tmp_path / ".env").write_text("GEMINI_API_KEY=from-dotenv\n")
    monkeypatch.chdir(tmp_path)

    gemini.gemini_search("q")

    assert sent[0]["headers"] == {"x-goog-api-key": "from-dotenv"}
