#!/usr/bin/env bash
# Build a release: refresh the provider -> model snapshot, then `uv build`.
# The snapshot (promptmap/provider_models.json) ships inside the wheel, so the
# model list `promptmap init` offers is fixed per release.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -I scripts/update_provider_models.py "$@"
rm -rf dist
uv build
