"""litellm_llm_call(api_key=...) passes the key to LiteLLM, and only when given."""

from __future__ import annotations

from types import SimpleNamespace as NS
from unittest import mock

import promptmap.llm as llm_mod


def _call(**kwargs) -> dict:
    fake = mock.MagicMock()
    fake.completion.return_value = NS(choices=[NS(message=NS(content="hi"))], usage=None)
    with mock.patch.object(llm_mod, "quiet_litellm", return_value=fake):
        llm_mod.litellm_llm_call("openai", "gpt-4o", **kwargs)("prompt")
    return fake.completion.call_args.kwargs


def test_api_key_is_passed_to_litellm():
    assert _call(api_key="sk-user")["api_key"] == "sk-user"


def test_no_api_key_leaves_litellm_to_the_environment():
    assert "api_key" not in _call()
