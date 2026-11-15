"""npm_bins: env var, then PATH, then an abxpkg install of the pinned version."""

from __future__ import annotations

import shutil
import sys
import types

import pytest

from promptmap_extra_model_kinds import npm_bins


@pytest.fixture
def no_path(monkeypatch):
    monkeypatch.delenv("PROMPTMAP_PI_BIN", raising=False)
    monkeypatch.setattr(npm_bins.shutil, "which", lambda name: "/usr/bin/npm" if name == "npm" else None)


def test_env_var_wins(monkeypatch):
    monkeypatch.setenv("PROMPTMAP_PI_BIN", "/opt/pi")
    assert npm_bins.find_bin(npm_bins.PI) == "/opt/pi"


def test_path_before_install(monkeypatch):
    monkeypatch.delenv("PROMPTMAP_PI_BIN", raising=False)
    monkeypatch.setattr(npm_bins.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(npm_bins, "_abxpkg_install", lambda cli: pytest.fail("should not install"))
    assert npm_bins.find_bin(npm_bins.PI) == "/usr/bin/pi"


def test_without_abxpkg_returns_none(no_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "abxpkg", None)  # import fails
    assert npm_bins.find_bin(npm_bins.PI) is None


def test_without_npm_returns_none(monkeypatch):
    monkeypatch.delenv("PROMPTMAP_PI_BIN", raising=False)
    monkeypatch.setattr(npm_bins.shutil, "which", lambda name: None)
    monkeypatch.setitem(sys.modules, "abxpkg", types.SimpleNamespace(Binary=None, NpmProvider=None))
    assert npm_bins.find_bin(npm_bins.PI) is None


@pytest.mark.parametrize("cli", [npm_bins.PI, npm_bins.OPENCODE])
def test_installs_pinned_version_into_prefix(cli, no_path, monkeypatch, tmp_path):
    """The pinned package, the cache prefix, min release age, postinstall only where needed."""
    seen = {}

    class FakeProvider:
        def __init__(self, **kw):
            seen["provider"] = kw

    class FakeBinary:
        loaded_abspath = None

        def __init__(self, **kw):
            seen["binary"] = kw

        def load(self):
            raise RuntimeError("not installed yet")

        def install(self):
            done = FakeBinary.__new__(FakeBinary)
            done.loaded_abspath = tmp_path / "bin" / cli.bin_name
            return done

    monkeypatch.setitem(sys.modules, "abxpkg", types.SimpleNamespace(Binary=FakeBinary, NpmProvider=FakeProvider))
    monkeypatch.setenv("PROMPTMAP_NPM_PREFIX", str(tmp_path))
    monkeypatch.delenv(cli.env_var, raising=False)

    assert npm_bins.find_bin(cli) == str(tmp_path / "bin" / cli.bin_name)
    assert seen["provider"] == {
        "npm_prefix": tmp_path,
        "min_release_age": npm_bins.MIN_RELEASE_AGE_DAYS,
        "postinstall_scripts": cli is npm_bins.OPENCODE,
    }
    assert seen["binary"]["overrides"] == {"npm": {"install_args": [f"{cli.package}@{cli.version}"]}}


@pytest.mark.skipif(
    not (shutil.which("npm") and __import__("importlib").util.find_spec("abxpkg")),
    reason="needs npm and the [npm] extra",
)
@pytest.mark.parametrize("cli", [npm_bins.PI, npm_bins.OPENCODE])
def test_real_install(cli, monkeypatch, tmp_path):
    """A real npm install of the pinned version (skips PATH: the CLI may already be there)."""
    monkeypatch.setenv("PROMPTMAP_NPM_PREFIX", str(tmp_path))
    path = npm_bins._abxpkg_install(cli)
    assert path and path.startswith(str(tmp_path))
    proc = __import__("subprocess").run([path, "--version"], capture_output=True, text=True)
    assert cli.version in proc.stdout + proc.stderr  # pi prints it on stderr
