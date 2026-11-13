"""Write promptmap/provider_models.json: aisuite provider -> chat model names.

Source: LiteLLM's model price list.  Run by scripts/deploy.sh before
`uv build`, so each release ships a fixed snapshot and promptmap never fetches
it at runtime.  `promptmap init` offers these names as choices; any other
model name can still be typed in.

    python scripts/update_provider_models.py [--source PATH_OR_URL]
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

SOURCE_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/refs/heads/main/"
    "model_prices_and_context_window.json"
)
OUT = Path(__file__).resolve().parent.parent / "promptmap" / "provider_models.json"

# aisuite provider name -> the litellm_provider values naming the same API.
# aisuite providers missing here (deepgram is speech-only, edenai, requesty...)
# get no list; `promptmap init` asks for a typed model name instead.
LITELLM_PROVIDERS: dict[str, tuple[str, ...]] = {
    "anthropic": ("anthropic",),
    "aws": ("bedrock", "bedrock_converse"),
    "azure": ("azure",),
    "cerebras": ("cerebras",),
    "cohere": ("cohere_chat",),
    "crusoe": ("crusoe",),
    "deepseek": ("deepseek",),
    "featherless": ("featherless_ai",),
    "fireworks": ("fireworks_ai",),
    "gemini": ("gemini",),
    "google": ("vertex_ai-language-models",),
    "groq": ("groq",),
    "huggingface": ("huggingface",),
    "inception": ("inception",),
    "mistral": ("mistral",),
    "nebius": ("nebius",),
    "ollama": ("ollama",),
    "openai": ("openai",),
    "openrouter": ("openrouter",),
    "sambanova": ("sambanova",),
    "together": ("together_ai",),
    "tongyi": ("dashscope",),
    "watsonx": ("watsonx",),
    "xai": ("xai",),
}

CHAT_MODES = {"chat", "responses"}


def _load(source: str) -> dict:
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=60) as response:
            return json.load(response)
    return json.loads(Path(source).read_text(encoding="utf-8"))


def _strip_prefix(name: str, litellm_provider: str) -> str:
    """'groq/llama-3' -> 'llama-3': aisuite takes the bare name after 'provider:'."""
    for prefix in {litellm_provider, litellm_provider.split("_")[0], litellm_provider.split("-")[0]}:
        if name.startswith(prefix + "/"):
            return name[len(prefix) + 1:]
    return name


def extract(prices: dict) -> dict[str, list[str]]:
    wanted = {lp: ours for ours, lps in LITELLM_PROVIDERS.items() for lp in lps}
    models: dict[str, set[str]] = {ours: set() for ours in LITELLM_PROVIDERS}
    for name, info in prices.items():
        if not isinstance(info, dict) or info.get("mode") not in CHAT_MODES:
            continue
        lp = info.get("litellm_provider")
        if lp in wanted and "*" not in name:  # skip Bedrock provisioned-throughput patterns
            models[wanted[lp]].add(_strip_prefix(name, lp))
    return {provider: sorted(names) for provider, names in sorted(models.items()) if names}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default=SOURCE_URL, help="URL or local path of the LiteLLM JSON")
    args = parser.parse_args()
    mapping = extract(_load(args.source))
    if not mapping:
        print("No models extracted; refusing to overwrite.", file=sys.stderr)
        return 1
    OUT.write_text(json.dumps(mapping, indent=1) + "\n", encoding="utf-8")
    total = sum(len(v) for v in mapping.values())
    print(f"Wrote {OUT.relative_to(Path.cwd()) if OUT.is_relative_to(Path.cwd()) else OUT}: "
          f"{len(mapping)} providers, {total} models")
    return 0


if __name__ == "__main__":
    sys.exit(main())
