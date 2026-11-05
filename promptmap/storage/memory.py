"""In-memory storage backend for browser and test environments."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from promptmap.run_ids import template_hash
from promptmap.tokens import add_tokens


class MemoryStorageBackend:
    def __init__(self) -> None:
        self._runs: dict[str, dict[str, Any]] = {}
        self._results: dict[str, dict[str, dict[str, Any]]] = {}
        self._tests: dict[str, list[dict[str, Any]]] = {}
        self._cache: dict[str, str] = {}
        self._cache_rows: dict[str, dict[str, Any]] = {}
        from promptmap.files import MemoryBlobStore
        self._blob_store = MemoryBlobStore()

    def blob_store(self):
        """File outputs' bytes, kept in memory alongside everything else."""
        return self._blob_store

    def init_db(self) -> None:
        return None

    def create_run(self, model_count: int, git_sha: str | None = None) -> str:
        run_id = str(uuid.uuid4())
        now = _now()
        self._runs[run_id] = {
            "run_id": run_id,
            "run_date": _today(now),
            "created_at": now,
            "status": "running",
            "completed_at": None,
            "model_count": model_count,
            "git_sha": git_sha,
            "map_hash": None,
            "run_hash": None,
        }
        self._results[run_id] = {}
        self._tests[run_id] = []
        return run_id

    def record_run_ids(self, run_id: str, map_hash: str, run_hash: str) -> None:
        """Store the run's content hashes (see :mod:`promptmap.run_ids`)."""
        self._runs[run_id].update({"map_hash": map_hash, "run_hash": run_hash})

    def finish_run(self, run_id: str, status: str) -> None:
        self._runs[run_id]["status"] = status
        self._runs[run_id]["completed_at"] = _now()

    def get_latest_successful_run(self) -> dict[str, Any] | None:
        matches = [
            row for row in self._runs.values()
            if row["status"] in {"success", "partial"}
        ]
        if not matches:
            return None
        return max(matches, key=lambda row: row["created_at"])

    def record_test_result(self, run_id: str, result: Any) -> None:
        self._tests[run_id].append(
            {
                "run_id": run_id,
                "test_name": result.test_name,
                "status": result.status,
                "prompt_rendered": result.prompt_rendered,
                "llm_output": result.llm_output,
                "error": result.error or None,
                "completed_at": _now(),
                "execution_ms": result.execution_ms,
            }
        )

    def get_test_results(self, run_id: str) -> list[dict[str, Any]]:
        return list(self._tests.get(run_id, []))

    def get_model_outputs_from_run(self, run_id: str, model_names: list[str]) -> dict[str, str]:
        results = self._results.get(run_id, {})
        return {
            name: row.get("llm_output_validated") or row["llm_output"]
            for name, row in results.items()
            if name in model_names and row["status"] == "success"
        }

    def get_cached_llm_output(self, cache_key: str) -> str | None:
        return self._cache.get(_prompt_hash(cache_key))

    def get_previous_result(
        self, model_name: str, template_hash: str | None, exclude_run_id: str | None = None
    ) -> dict[str, Any] | None:
        """The latest successful row of *model_name* run from this same
        template (any template when *template_hash* is None), outside
        *exclude_run_id*."""
        matches = [
            row
            for run_id, results in self._results.items()
            if run_id != exclude_run_id
            for row in results.values()
            if row.get("model_name") == model_name
            and row.get("template_hash") is not None
            and (template_hash is None or row.get("template_hash") == template_hash)
            and row.get("status") == "success"
        ]
        if not matches:
            return None
        # Ties on the millisecond clock go to the row written last.
        _, row = max(enumerate(matches), key=lambda ir: (ir[1]["completed_at"], ir[0]))
        return {
            "run_id": row["run_id"],
            "prompt_rendered": row["prompt_rendered"],
            "llm_output": row.get("llm_output_validated") or row["llm_output"],
            "llm_output_raw": row["llm_output"],
        }

    def get_cached_token_usage(self, cache_key: str) -> int | None:
        row = self._cache_rows.get(_prompt_hash(cache_key))
        if row is None:
            return None
        return add_tokens(row.get("spent_tokens"), row.get("cache_spent_tokens"))

    def upsert_model_pending(
        self,
        run_id: str,
        model_name: str,
        prompt_template: str,
        depends_on: list[str],
        model_type: str = "",
        config: dict | None = None,
    ) -> None:
        self._results[run_id][model_name] = {
            "run_id": run_id,
            "model_name": model_name,
            "status": "pending",
            "prompt_template": prompt_template,
            "template_hash": template_hash(prompt_template),
            "prompt_rendered": None,
            "prompt_hash": None,
            "llm_output": None,
            "llm_output_validated": None,
            "cached": False,
            "started_at": None,
            "completed_at": None,
            "execution_ms": None,
            "error": None,
            "depends_on": json.dumps(depends_on),
            "model_type": model_type,
            "config": json.dumps(config or {}, sort_keys=True),
            "spent_tokens": None,
            "cache_spent_tokens": None,
        }

    def mark_model_running(self, run_id: str, model_name: str) -> None:
        self._results[run_id][model_name]["status"] = "running"
        self._results[run_id][model_name]["started_at"] = _now()

    def mark_model_success(
        self,
        run_id: str,
        model_name: str,
        prompt_rendered: str,
        llm_output: str,
        cache_key: str | None = None,
        cached: bool = False,
    ) -> None:
        # Tolerate a model_name that was never registered via
        # upsert_model_pending — e.g. test prompts cached by execute_tests.
        # Mirrors the SQLite backend, whose UPDATE is a no-op on a missing row.
        row = self._results.setdefault(run_id, {}).setdefault(model_name, {})
        now = _now()
        started_at = row.get("started_at")
        elapsed = 0
        if started_at:
            try:
                elapsed = int(
                    (datetime.fromisoformat(now) - datetime.fromisoformat(started_at)).total_seconds() * 1000
                )
            except ValueError:
                elapsed = 0

        row.update(
            {
                "status": "success",
                "prompt_rendered": prompt_rendered,
                "prompt_hash": _prompt_hash(cache_key or prompt_rendered),
                "llm_output": llm_output,
                "completed_at": now,
                "execution_ms": elapsed,
                "cached": cached,
            }
        )
        self._cache[_prompt_hash(cache_key or prompt_rendered)] = llm_output
        self._cache_rows[_prompt_hash(cache_key or prompt_rendered)] = row

    def record_token_usage(
        self,
        run_id: str,
        model_name: str,
        spent_tokens: int | None = None,
        cache_spent_tokens: int | None = None,
    ) -> None:
        """Store the tokens a model spent, and those its cache hits saved."""
        self._results.setdefault(run_id, {}).setdefault(model_name, {}).update(
            {
                "spent_tokens": spent_tokens,
                "cache_spent_tokens": cache_spent_tokens,
            }
        )

    def record_validated_output(self, run_id: str, model_name: str, output: str) -> None:
        """Store the post-validation output alongside the raw one.

        The raw text stays in ``llm_output`` because that is what the prompt
        cache serves — editing a validator must not force a new LLM call.
        """
        self._results.setdefault(run_id, {}).setdefault(model_name, {})[
            "llm_output_validated"
        ] = output

    def mark_model_error(self, run_id: str, model_name: str, error: str) -> None:
        self._results[run_id][model_name].update(
            {
                "status": "error",
                "completed_at": _now(),
                "error": error,
            }
        )

    def mark_model_skipped(self, run_id: str, model_name: str) -> None:
        self._results[run_id][model_name]["status"] = "skipped"

    def get_run_results(self, run_id: str) -> list[dict[str, Any]]:
        return list(self._results.get(run_id, {}).values())

    def model_has_results(self, model_name: str) -> bool:
        return any(
            name == model_name or name.startswith(model_name + "[")
            for results in self._results.values() for name in results
        )

    def rename_model(self, old: str, new: str, retemplate: dict[str, tuple[str, str]] | None = None) -> int:
        """Move *old*'s history to *new* — see :meth:`SQLiteStorageBackend.rename_model`."""
        hashes = {name: (template_hash(a), template_hash(b), b) for name, (a, b) in (retemplate or {}).items()}
        moved = 0
        for results in self._results.values():
            for name in list(results):
                if name == old or name.startswith(old + "["):
                    row = results.pop(name)
                    row["model_name"] = new + name[len(old):]
                    results[row["model_name"]] = row
                    moved += 1
            for row in results.values():
                deps = row.get("depends_on")
                if isinstance(deps, str) and json.dumps(old) in deps:
                    row["depends_on"] = json.dumps([new if d == old else d for d in json.loads(deps)])
                before_after = hashes.get(row.get("model_name"))
                if before_after and row.get("template_hash") == before_after[0]:
                    row["template_hash"], row["prompt_template"] = before_after[1], before_after[2]
        return moved

    def clear_cache(self, models: list[str] | None = None, previous: bool = False) -> int:
        """Forget cached outputs (and with *previous*, previous runs) — see
        :meth:`SQLiteStorageBackend.clear_cache`."""
        return _clear_cache_rows(self, None if models is None else set(models), previous)

    def get_latest_runs(self, limit: int = 10) -> list[dict[str, Any]]:
        return sorted(
            self._runs.values(),
            key=lambda row: row["created_at"],
            reverse=True,
        )[:limit]


def _clear_cache_rows(backend: "MemoryStorageBackend", models, previous: bool) -> int:
    changed = 0
    for results in backend._results.values():
        for row in results.values():
            if models is not None and row.get("model_name") not in models:
                continue
            live = row.get("prompt_hash") is not None or (previous and row.get("template_hash") is not None)
            if not live:
                continue
            backend._cache.pop(row.get("prompt_hash"), None)
            backend._cache_rows.pop(row.get("prompt_hash"), None)
            row["prompt_hash"] = None
            if previous:
                row["template_hash"] = None
            changed += 1
    return changed


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _today(now: str) -> str:
    return now[:10]


def _prompt_hash(cache_key: str) -> str:
    return hashlib.sha256(cache_key.encode()).hexdigest()
