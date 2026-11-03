"""``agent_multi_harness`` with ``agent_backend="pi"``."""

from __future__ import annotations

import json

from promptmap_extra_model_kinds import pi_harness
from conftest import run_models


def _model(tmp_path):
    workdir = tmp_path / "w"
    return f'{{{{ config(model_type="agent_multi_harness", agent_backend="pi", agent_dir="{workdir}") }}}}hi'


def test_missing_pi_says_how_to_install(tmp_path, monkeypatch):
    monkeypatch.setattr(pi_harness, "_pi_bin", lambda: None)
    _, _, results = run_models({"a": _model(tmp_path)})
    assert "npm i -g @mariozechner/pi-coding-agent" in results["a"].error


def test_pi_events_become_output(tmp_path, monkeypatch):
    """run_pi against a fake pi binary that prints JSON events."""
    events = [
        {"type": "session"},
        {"type": "message_update", "message": {}},
        {"type": "message_end", "message": {"role": "assistant", "content": [{"type": "text", "text": "done"}],
                                            "usage": {"cost": {"total": 0.01}}, "stopReason": "stop"}},
        {"type": "turn_end"},
        {"type": "agent_end", "messages": []},
    ]
    script = tmp_path / "pi"
    script.write_text("#!/bin/sh\ncat <<'EOF'\n" + "\n".join(json.dumps(e) for e in events) + "\nEOF\n")
    script.chmod(0o755)
    monkeypatch.setenv("PROMPTMAP_PI_BIN", str(script))
    _, _, results = run_models({"a": _model(tmp_path)})
    assert results["a"].value["output"] == "done"
    assert all(e.get("type") != "message_update" for e in results["a"].value["logs"])
