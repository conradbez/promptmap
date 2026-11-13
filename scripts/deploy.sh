#!/usr/bin/env bash
# Build a release.  The provider/model menu `promptmap init` shows comes from
# the model list bundled in the installed litellm package, so there is nothing
# to download first: newer litellm, newer list.
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf dist
uv build
