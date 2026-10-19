"""rag_openai_search / rag_anthropic_search from promptmap-extra-model-kinds, against fake SDK clients."""

from __future__ import annotations

import importlib
from types import SimpleNamespace as NS

import pytest

from promptmap import model_types, rag


def _import(monkeypatch, name):
    """Import an extras module without leaking the kinds it registers into other tests."""
    monkeypatch.setattr(model_types, "_REGISTRY", dict(model_types._REGISTRY))
    monkeypatch.setattr(rag, "_REGISTRY", dict(rag._REGISTRY))
    try:
        return importlib.import_module(f"promptmap_extra_model_kinds.{name}")
    except ImportError:
        pytest.skip("promptmap-extra-model-kinds is not installed")


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

@pytest.fixture
def openai_search(monkeypatch):
    mod = _import(monkeypatch, "openai_search")
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        cite = lambda title, url: NS(type="url_citation", title=title, url=url)  # noqa: E731
        return NS(
            output_text=" Python 3.15 is out. ",
            output=[
                NS(type="web_search_call"),
                NS(type="message", content=[
                    NS(type="output_text", annotations=[cite("python.org", "https://a"), cite("python.org", "https://a")]),
                    NS(type="output_text", annotations=[cite("News", "https://b"), NS(type="file_citation")]),
                ]),
            ],
        )

    keys = []
    monkeypatch.setattr(
        mod.openai, "OpenAI",
        lambda api_key, timeout: keys.append(api_key) or NS(responses=NS(create=create)),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_SEARCH_MODEL", raising=False)
    return mod, calls, keys


def test_openai_answer_then_deduped_sources(openai_search):
    mod, calls, keys = openai_search

    assert mod.openai_search("[q]\nlatest python\n[/q]") == [
        "Python 3.15 is out.",
        "python.org\nhttps://a",
        "News\nhttps://b",
    ]
    [call] = calls
    assert call["tools"] == [{"type": "web_search"}]
    assert call["model"] == mod.DEFAULT_MODEL
    assert call["input"].endswith("Query: latest python")
    assert keys == ["test-key"]


def test_openai_model_override(openai_search, monkeypatch):
    mod, calls, _ = openai_search
    monkeypatch.setenv("OPENAI_SEARCH_MODEL", "env-model")
    mod.openai_search("q")
    mod.openai_search("q", model="arg-model")

    assert [c["model"] for c in calls] == ["env-model", "arg-model"]


def test_openai_missing_key_names_the_rag(openai_search, monkeypatch, tmp_path):
    mod, _, _ = openai_search
    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.chdir(tmp_path)  # no .env here or above in tmp

    with pytest.raises(RuntimeError, match=r"rag_openai_search\(\) needs OPENAI_API_KEY"):
        mod.openai_search("q")


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

def _text(text, *citations):
    return NS(type="text", text=text, citations=list(citations) or None)


def _cite(title, url):
    return NS(type="web_search_result_location", title=title, url=url)


@pytest.fixture
def anthropic_search(monkeypatch):
    mod = _import(monkeypatch, "anthropic_search")
    calls, responses = [], []

    def create(**kwargs):
        calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return responses.pop(0)

    monkeypatch.setattr(mod.anthropic, "Anthropic", lambda timeout: NS(beta=NS(messages=NS(create=create))))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("ANTHROPIC_SEARCH_MODEL", raising=False)
    return mod, calls, responses


def test_anthropic_resumes_pause_turn_and_collects_citations(anthropic_search):
    mod, calls, responses = anthropic_search
    paused = [NS(type="server_tool_use"), _text("Python 3.15 ", _cite("python.org", "https://a"))]
    responses += [
        NS(stop_reason="pause_turn", content=paused),
        NS(stop_reason="end_turn", content=[
            NS(type="web_search_tool_result"),
            _text("is out.", _cite("python.org", "https://a"), _cite("News", "https://b")),
        ]),
    ]

    assert mod.anthropic_search("latest python") == ["Python 3.15 is out.", "python.org\nhttps://a", "News\nhttps://b"]
    first, second = calls
    assert first["tools"] == [mod.WEB_SEARCH_TOOL]
    assert first["fallbacks"] == "default" and first["betas"] == [mod.FALLBACK_BETA]
    assert first["output_config"] == {"effort": "low"}
    # The paused turn is re-sent as-is, with no extra "continue" message.
    assert second["messages"] == [first["messages"][0], {"role": "assistant", "content": paused}]


def test_anthropic_override_model_skips_fallbacks(anthropic_search):
    mod, calls, responses = anthropic_search
    responses.append(NS(stop_reason="end_turn", content=[_text("ok")]))

    assert mod.anthropic_search("q", model="other-model") == ["ok"]
    assert calls[0]["model"] == "other-model"
    assert "fallbacks" not in calls[0] and "betas" not in calls[0]


def test_anthropic_refusal_raises(anthropic_search):
    mod, _, responses = anthropic_search
    responses.append(NS(stop_reason="refusal", stop_details=NS(category="cyber"), content=[]))

    with pytest.raises(RuntimeError, match="declined.*cyber"):
        mod.anthropic_search("q")
