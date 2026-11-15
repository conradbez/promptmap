"""Find — or, with the ``[npm]`` extra, install — the npm CLIs the agents run.

Lookup order for each binary:

1. its ``PROMPTMAP_*_BIN`` env var;
2. ``PATH``;
3. with ``abxpkg`` installed (``pip install "promptmap-extra-model-kinds[npm]"``,
   Python ≥ 3.12), an ``npm install`` of a pinned version into
   ``PROMPTMAP_NPM_PREFIX`` (default ``~/.cache/promptmap/npm``).  Needs Node
   and npm on the host.

abxpkg refuses packages younger than ``MIN_RELEASE_AGE_DAYS`` and keeps npm
postinstall scripts off unless a package needs one (opencode does: its
postinstall fetches the prebuilt binary for this platform).
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

#: Pinned for the same supply-chain reason promptmap pins litellm. agent_modal
#: installs the same opencode version into its Modal image.
OPENCODE_VERSION = "1.18.32"
PI_VERSION = "0.73.1"

MIN_RELEASE_AGE_DAYS = 14


@dataclass(frozen=True)
class NpmCli:
    bin_name: str
    package: str
    version: str
    env_var: str
    postinstall: bool = False

    @property
    def install_hint(self) -> str:
        return (
            f"npm i -g {self.package}, pip install \"promptmap-extra-model-kinds[npm]\" "
            f"to install it automatically, or set {self.env_var}"
        )


OPENCODE = NpmCli("opencode", "opencode-ai", OPENCODE_VERSION, "PROMPTMAP_OPENCODE_BIN", postinstall=True)
PI = NpmCli("pi", "@mariozechner/pi-coding-agent", PI_VERSION, "PROMPTMAP_PI_BIN")


def npm_prefix() -> Path:
    return Path(os.getenv("PROMPTMAP_NPM_PREFIX") or Path.home() / ".cache" / "promptmap" / "npm").expanduser()


def find_bin(cli: NpmCli) -> str | None:
    """Path to *cli*'s binary, installing it if abxpkg is available; else None."""
    return os.getenv(cli.env_var) or shutil.which(cli.bin_name) or _abxpkg_install(cli)


def _abxpkg_install(cli: NpmCli) -> str | None:
    """npm-install *cli* into ``npm_prefix()`` via abxpkg; None without abxpkg or npm."""
    try:
        from abxpkg import Binary, NpmProvider
    except ImportError:
        return None
    if not shutil.which("npm"):
        return None
    provider = NpmProvider(
        npm_prefix=npm_prefix(),
        min_release_age=MIN_RELEASE_AGE_DAYS,
        postinstall_scripts=cli.postinstall,
    )
    binary = Binary(
        name=cli.bin_name,
        binproviders=[provider],
        overrides={"npm": {"install_args": [f"{cli.package}@{cli.version}"]}},
    )
    try:
        binary = binary.load()
    except Exception:
        binary = binary.install()
    return str(binary.loaded_abspath) if binary.loaded_abspath else None
