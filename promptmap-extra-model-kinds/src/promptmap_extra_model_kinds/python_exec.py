"""
``execute_python`` — run the rendered template as Python, in-process.

Moved out of promptmap core.  Importing this module registers the kind.
"""

from __future__ import annotations

import io
import json
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from promptmap.files import Dir, File, FileOutputError, Output, contains_files
from promptmap.model_types import ModelCall, ModelKind, register_model_kind


async def run_python(rendered: str, call: ModelCall) -> Any:
    """Run the rendered template as Python source and return what it produced.

    Output is whatever the code prints; failing that, a variable named
    ``output``.  Upstream outputs are available as ``ref('name')`` and in the
    ``model_outputs`` dict.  The code runs in-process with full builtins — a
    .prompt file using this kind is trusted code.

    To produce files, either set ``output`` to a ``File``/``Dir``/``Output``
    (or a dict holding them), or write into ``out_dir`` — an empty directory
    whose contents become the output's files, with anything printed as its
    text.  ``File``, ``Dir`` and ``Output`` are in scope without an import.

    It goes through the same cache as an LLM call, so unchanged code does not
    re-execute on the next run.
    """
    # ref() in the Python code reads upstream outputs that never appear in the
    # rendered text, so they join the cache key; otherwise a changed upstream
    # would be served the old result.
    key = rendered + "\x00upstream:" + _upstream_fingerprint(call)
    return await call.compute(key, compute=lambda: _exec_python(rendered, call))


def _upstream_fingerprint(call: ModelCall) -> str:
    """A hash of every dependency's current output (files by their hashes)."""
    import hashlib

    from promptmap.files import encode_output

    seen = {
        dep: encode_output(call.outputs[dep]) if call.outputs.get(dep) is not None else None
        for dep in call.spec.depends_on
    }
    return hashlib.sha256(json.dumps(seen, sort_keys=True).encode()).hexdigest()


def _exec_python(rendered: str, call: ModelCall) -> Any:
    with tempfile.TemporaryDirectory(prefix="promptmap-out-") as tmp:
        out_dir = Path(tmp)
        namespace: dict = {
            "model_outputs": call.outputs,
            "ref": lambda name: call.outputs.get(name),
            "out_dir": out_dir,
            "File": File,
            "Dir": Dir,
            "Output": Output,
        }
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exec(compile(rendered, f"<{call.spec.name}>", "exec"), namespace)  # noqa: S102

        # Read what was written before the directory is removed.
        written = _collect_out_dir(out_dir)

    value = namespace.get("output")
    if contains_files(value):
        if written:
            raise FileOutputError(
                f"Model '{call.spec.name}' both wrote files to out_dir and set "
                "`output` to files — use one or the other."
            )
        return value

    printed = stdout.getvalue()
    if printed:
        text = printed.rstrip("\n")
    elif "output" in namespace:
        text = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
    else:
        text = ""
    return Output(text, files=written) if written else text


def _collect_out_dir(out_dir: Path) -> list:
    """The files and directories left in *out_dir*, as File/Dir objects."""
    items = []
    for path in sorted(out_dir.iterdir()):
        if path.is_symlink():
            raise FileOutputError(f"out_dir: refusing symlink '{path.name}'.")
        items.append(Dir(path) if path.is_dir() else File(path))
    return items



#: Run the rendered template as Python.  The rendered text is code, not a
#: prompt, so a global instruction is never applied.
KIND = register_model_kind(ModelKind(
    name="execute_python",
    exec_fn=run_python,
    accepts_global_instruction=False,
))
