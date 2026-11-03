"""``execute_python``: moved out of promptmap core into the extras."""

from __future__ import annotations

from promptmap.files import Output

from conftest import run_models

PNG = b"\x89PNG\r\n\x1a\n" + b"fake-image-bytes"


def image_llm(prompt: str, files: list | None = None, config: dict | None = None):
    from promptmap.files import File

    return File(PNG, name="logo.png") if "draw" in prompt else "ok"


def test_registered_with_the_extras():
    from importlib.metadata import entry_points

    eps = {ep.name: ep.value for ep in entry_points(group="promptmap.model_kinds")}
    assert eps["execute_python"] == "promptmap_extra_model_kinds.python_exec"


def test_python_out_dir_becomes_output_files():
    source = (
        '{{ config(model_type="execute_python") }}\n'
        "(out_dir / 'report.csv').write_text('a,b')\n"
        "(out_dir / 'site').mkdir()\n"
        "(out_dir / 'site' / 'index.html').write_text('<p>')\n"
        "print('made a report')\n"
    )
    _, _, results = run_models({"py": source}, llm_call=image_llm)
    value = results["py"].value
    assert isinstance(value, Output)
    assert value.text == "made a report"
    assert value.files["report.csv"].read_bytes() == b"a,b"
    assert value.files["site"]["index.html"].read_bytes() == b"<p>"


def test_python_reads_upstream_file_and_returns_one():
    source = (
        '{{ config(model_type="execute_python") }}\n'
        "src = ref('logo')\n"
        "output = File(src.read_bytes()[::-1], name='flipped.bin')\n"
    )
    _, _, results = run_models({"logo": "draw", "py": source}, llm_call=image_llm)
    assert results["py"].value.read_bytes() == PNG[::-1]


def test_execute_python_cache_hit_is_still_recorded_as_success():
    models = {"py": '{{ config(model_type="execute_python") }}\noutput = "hello"'}
    storage, _, _ = run_models(models)
    _, run_id, results = run_models(models, storage=storage)
    assert results["py"].cached is True
    assert storage.get_run_results(run_id)[0]["status"] == "success"
    assert storage.get_model_outputs_from_run(run_id, ["py"]) == {"py": "hello"}


def test_execute_python_propagates_skip_this_and_downstream():
    _, _, results = run_models({
        "py": '{{ config(model_type="execute_python") }}\n{{ skip_this_and_downstream("stop") }}',
        "after": 'Uses {{ ref("py") }}',
    })
    assert results["py"].prompt_skipped is True
    assert results["after"].status == "skipped"


def test_global_instruction_is_never_prepended():
    seen = []

    def llm(prompt, config=None):
        seen.append(prompt)
        return "mock response"

    _, _, results = run_models(
        {"topic": "Name one topic.",
         "transform": '{{ config(model_type="execute_python") }}\noutput = ref("topic").upper()'},
        llm_call=llm, global_instruction="Always answer in French.",
    )
    assert results["transform"].value == "MOCK RESPONSE"
