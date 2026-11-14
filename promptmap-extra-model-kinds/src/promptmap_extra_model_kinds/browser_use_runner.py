"""
Runs one browser-use task.  Executed as a script in browser-use's own Python
environment (see `browser_use_exec`), never imported by promptmap: browser-use
1.0 pins dependencies (mcp 2.x, pydantic, ...) that promptmap's litellm can't
share an environment with.

Reads a JSON request on stdin::

    {"task": "...", "workspace": "/tmp/...", "llm": "openai:gpt-4o" | null,
     "cloud": false, "headless": false, "max_steps": 100,
     "allowed_domains": ["example.com"] | null}

and prints one line ``PROMPTMAP_RESULT {"text": ..., "status": ..., "error": ...}``
on stdout; browser-use's own logging goes to stderr.  Stdlib + browser_use only.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys

MARKER = "PROMPTMAP_RESULT "


async def run(request: dict) -> dict:
    from browser_use import Agent, Browser

    browser_kwargs = {}
    if request.get("allowed_domains"):
        browser_kwargs["allowed_domains"] = request["allowed_domains"]
    if request.get("cloud"):
        browser = Browser("cloud", **browser_kwargs)
    else:
        browser = Browser("chromium", headless=bool(request.get("headless")), **browser_kwargs)
    agent_kwargs = {"browser": browser, "workspace": request["workspace"]}
    if request.get("llm"):
        agent_kwargs["llm"] = request["llm"]
    agent = Agent(request["task"], **agent_kwargs)
    try:
        history = await agent.run(max_steps=int(request.get("max_steps") or 100))
    finally:
        try:
            await agent.close()
        except Exception:
            pass
    return {
        "text": history.final_result(),
        "status": getattr(history, "status", None),
        "error": getattr(history, "error", None),
    }


def main() -> None:
    # Everything but the result line goes to stderr, so stdout stays parseable.
    logging.basicConfig(stream=sys.stderr)
    real_stdout, sys.stdout = sys.stdout, sys.stderr
    request = json.loads(sys.stdin.read())
    try:
        result = asyncio.run(run(request))
    except Exception as exc:  # reported to promptmap, which raises it there
        result = {"text": None, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
    real_stdout.write(MARKER + json.dumps(result) + "\n")
    real_stdout.flush()


if __name__ == "__main__":
    main()
