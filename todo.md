# TODO

## Is `litellm[mcp]` still required?

`pyproject.toml` depends on `litellm[mcp]==1.102.1`. The `[mcp]` extra was added
in 9206aae because an older litellm imported `mcp` on every tool-calling request
(mini-swe-agent always sends tools), so coding agents crashed with
`ModuleNotFoundError: No module named 'mcp'`.

In 1.102.1 the core `mcp` imports sit under `if TYPE_CHECKING:`, and mocked
tool-calling requests (openai, anthropic) work with `mcp` blocked. Mocks may skip
the real request path, so this is unconfirmed. The extra adds ~13 packages
(mcp, starlette, uvicorn, sse-starlette, pyjwt, pydantic-settings, ...).

Test:

1. In `pyproject.toml` change `"litellm[mcp]==1.102.1"` to `"litellm==1.102.1"`.
2. `uv lock && uv sync`, then confirm `uv run python -c "import mcp"` fails.
3. With a real key in `.env`, run one `coding_agent` model end to end, e.g.
   `cd examples/<project with a coding_agent model> && uv run promptmap run -s <model>`.
4. Passes → keep the change, bump the version. Fails with `No module named 'mcp'`
   → revert, and note the litellm version here.
