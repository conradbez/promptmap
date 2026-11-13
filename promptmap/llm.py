"""
LLM call resolution.

The default backend is LiteLLM, calling the provider/model named in the
project's settings.yml (see :mod:`promptmap.settings`).  A client.py exposing
``llm_call(prompt: str) -> str`` replaces it, for full control over the call.

client.py is also where a project registers its own model kinds, so promptmap imports
it *before* parsing any .prompt file (see :func:`load_client_module`).  The
module is executed once per path and cached, so importing it early costs
nothing later.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType
from typing import Callable

CLIENT_FILENAME = "client.py"

_module_cache: dict[str, ModuleType] = {}


def client_path(map_dir: str) -> str:
    """Return the client.py path for *map_dir* (i.e. in its parent)."""
    return os.path.join(os.path.dirname(map_dir), CLIENT_FILENAME)


def load_client_module(map_dir: str) -> ModuleType | None:
    """Import the project's client.py, or return None when there isn't one.

    Executed once per path and cached.  Call this before loading models: any
    ``promptmap.register_model_kind(...)`` in client.py must run before a .prompt file
    naming that kind is parsed, or the kind looks unknown.
    """
    candidate = client_path(map_dir)
    if not os.path.isfile(candidate):
        return None
    return _load_client_file(candidate)


def _load_client_file(candidate: str) -> ModuleType:
    resolved = os.path.realpath(candidate)
    if resolved in _module_cache:
        return _module_cache[resolved]

    spec = importlib.util.spec_from_file_location("_promptmap_user_client", candidate)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _module_cache[resolved] = module
    return module


def client_module_for(model_path: Path) -> ModuleType | None:
    """The client.py governing the .prompt file at *model_path*, or None.

    The nearest client.py in a directory above the file — the project root
    for ``map/x.prompt`` and for models in subfolders alike.  For a kind
    that needs a project setting but is handed only its model.
    """
    if not model_path.is_file():
        return None  # an inline model has no project
    for parent in model_path.resolve().parents:
        candidate = parent / CLIENT_FILENAME
        if candidate.is_file():
            return _load_client_file(str(candidate))
    return None


def try_load_client_module(map_dir: str) -> ModuleType | None:
    """Import client.py for its model-type registrations, ignoring any failure.

    Used at discovery time, where the client is wanted only for its side
    effects.  A client.py that cannot be imported is not reported here: the
    commands that actually need a backend call :func:`resolve_llm_call` later
    and surface the error there, with the message the user expects.
    """
    try:
        return load_client_module(map_dir)
    except Exception:
        return None


def resolve_llm_call(map_dir: str) -> Callable[[str], str]:
    """Return the project's ``llm_call``.

    client.py's ``llm_call`` wins when it defines one.  Otherwise the default
    backend is LiteLLM, calling the provider/model in settings.yml.  Raises
    with a helpful message when neither is set up.
    """
    module = load_client_module(map_dir)
    if module is not None and hasattr(module, "llm_call"):
        return module.llm_call

    from promptmap.settings import SETTINGS_FILE, load_settings

    settings = load_settings(map_dir)
    if settings.get("provider") and settings.get("model"):
        return litellm_llm_call(str(settings["provider"]), str(settings["model"]))

    if module is not None:
        raise AttributeError(
            f"{client_path(map_dir)} was found but does not define an "
            f"'llm_call(prompt: str) -> str' function, and {SETTINGS_FILE} sets no "
            "provider and model."
        )
    raise FileNotFoundError(
        f"No LLM configured: {SETTINGS_FILE} (provider + model) or client.py "
        "(llm_call) is needed next to your map/ directory. Run `promptmap init "
        "--provider <name> --model <name>` to write settings.yml."
    )


def has_llm_backend(map_dir: str) -> bool:
    """True when client.py defines ``llm_call`` or settings.yml names a provider and model."""
    module = try_load_client_module(map_dir)
    if module is not None and hasattr(module, "llm_call"):
        return True
    from promptmap.settings import load_settings

    try:
        settings = load_settings(map_dir)
    except Exception:
        return True  # a broken file is reported by resolve_llm_call, not hidden by a prompt
    return bool(settings.get("provider") and settings.get("model"))


def _file_block(f) -> dict:
    """A prompt-attached file as a chat content block LiteLLM can send.

    UTF-8 text goes inline; images become ``image_url`` blocks and anything
    else (PDFs, ...) a ``file`` block, both as base64 data URLs.
    """
    import base64
    import mimetypes

    name = os.path.basename(getattr(f, "name", "file"))
    data = f.read()
    if isinstance(data, str):
        return {"type": "text", "text": f"--- {name} ---\n{data}"}
    mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
    if mime.startswith("text/") or mime in ("application/json", "application/octet-stream"):
        try:
            return {"type": "text", "text": f"--- {name} ---\n{data.decode('utf-8')}"}
        except UnicodeDecodeError:
            pass
    url = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
    if mime.startswith("image/"):
        return {"type": "image_url", "image_url": {"url": url}}
    return {"type": "file", "file": {"file_data": url, "filename": name}}


def quiet_litellm():
    """Import litellm with telemetry, message logging and debug output off.

    Clears any success/failure callbacks (Langfuse, OpenTelemetry, ...) so the
    default backend sends prompts to the provider and nowhere else.  A
    client.py that wants callbacks can set them in its own ``llm_call``.
    """
    import litellm

    litellm.telemetry = False
    litellm.turn_off_message_logging = True
    litellm.success_callback = []
    litellm.failure_callback = []
    litellm.suppress_debug_info = True
    return litellm


def litellm_model(provider: str, model: str) -> str:
    """LiteLLM's ``provider/model`` string; a model already prefixed is kept."""
    return model if model.startswith(provider + "/") else f"{provider}/{model}"


def litellm_llm_call(provider: str, model: str) -> Callable:
    """The default ``llm_call``: one user message to *provider*/*model* via LiteLLM.

    Values are passed through unchecked, so settings.yml can name any model
    LiteLLM accepts.  A .prompt file can override either for itself with
    ``{{ config(provider="openai", model="gpt-4o-mini") }}``.  Files attached
    to a prompt go in the same message.
    """
    default_target = litellm_model(provider, model)

    def llm_call(prompt: str, files: list | None = None, config: dict | None = None):
        from promptmap.tokens import LLMResult

        config = config or {}
        target = default_target
        if config.get("provider") or config.get("model"):
            target = litellm_model(str(config.get("provider") or provider), str(config.get("model") or model))
        content: str | list = prompt
        if files:
            content = [{"type": "text", "text": prompt}] + [_file_block(f) for f in files]
        response = quiet_litellm().completion(model=target, messages=[{"role": "user", "content": content}])
        usage = getattr(response, "usage", None)
        return LLMResult(
            response.choices[0].message.content,
            spent_tokens=getattr(usage, "total_tokens", None),
        )

    llm_call.__name__ = f"litellm_{provider}"
    # Part of the prompt cache key: switching model in settings.yml must not
    # serve answers cached from the old one.
    llm_call.cache_identity = default_target
    return llm_call


def resolve_blob_store(map_dir: str):
    """Return the blob store client.py declares, or None.

    client.py opts in by defining ``blob_store`` — a
    :class:`~promptmap.files.BlobStore` instance, or a zero-argument function
    returning one — to keep file outputs somewhere other than the run
    database (S3, a shared volume, …).
    """
    module = try_load_client_module(map_dir)
    provider = getattr(module, "blob_store", None) if module is not None else None
    if provider is None:
        return None
    import inspect

    from promptmap.files import BlobStore

    is_factory = inspect.isfunction(provider) or inspect.isclass(provider)
    store = provider() if is_factory else provider
    if not isinstance(store, BlobStore):
        raise TypeError(
            f"{client_path(map_dir)}: blob_store must provide put(), get() and "
            f"exists(); got {type(store).__name__}."
        )
    return store


def resolve_cache_check(map_dir: str) -> Callable[..., bool] | None:
    """Return client.py's ``cache_check(prompt, previous) -> bool``, or None.

    Asked on an exact prompt-cache miss whether a model's last result is
    still good enough for its changed prompt; see
    :meth:`~promptmap.executor.run_context.RunContext.reusable_previous`.
    """
    module = load_client_module(map_dir)
    return getattr(module, "cache_check", None) if module is not None else None


def resolve_classify_call(map_dir: str) -> Callable[[str, str], float] | None:
    """Return client.py's ``classify_call(state, question) -> float``, or None.

    Only classifier-judged tests need it; see :mod:`promptmap.classifier`.
    """
    module = load_client_module(map_dir)
    return getattr(module, "classify_call", None) if module is not None else None


def resolve_test_judge(map_dir: str) -> str:
    """Return client.py's ``test_judge`` ("llm" or "classifier"), default "llm"."""
    from promptmap.classifier import parse_judge

    module = load_client_module(map_dir)
    judge = getattr(module, "test_judge", None) if module is not None else None
    if judge is None:
        return "llm"
    return parse_judge(judge, f"{client_path(map_dir)} test_judge")
