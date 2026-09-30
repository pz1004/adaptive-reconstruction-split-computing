"""Atomic JSON run registry for resumable experiments."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


FINAL_STATUSES = {"complete", "failed"}


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def deterministic_run_id(spec: Mapping[str, Any]) -> str:
    prefix_fields = [
        str(spec.get("stage", "run")),
        str(spec.get("dataset", "none")),
        str(spec.get("split_point", "none")),
        str(spec.get("method", "none")),
        f"s{spec.get('seed', 'na')}",
    ]
    digest = hashlib.sha256(canonical_json(spec).encode("utf-8")).hexdigest()[:12]
    return "__".join(prefix_fields + [digest]).replace("/", "-")


def _validate_finite(value: Any, path: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite value at {path}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _validate_finite(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_finite(child, f"{path}[{index}]")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


class RunRegistry:
    def __init__(self, manifest_directory: str | Path):
        self.directory = Path(manifest_directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, run_id: str) -> Path:
        return self.directory / f"{run_id}.json"

    def register(self, spec: Mapping[str, Any], *, resume: bool = False) -> dict[str, Any]:
        normalized_spec = json.loads(canonical_json(spec))
        run_id = deterministic_run_id(normalized_spec)
        path = self.path_for(run_id)
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("spec") != normalized_spec:
                raise ValueError(f"Duplicate run ID with incompatible configuration: {run_id}")
            if not resume:
                raise ValueError(f"Duplicate run ID: {run_id}")
            return existing
        now = datetime.now(timezone.utc).isoformat()
        record = {"run_id": run_id, "status": "planned", "spec": normalized_spec, "created_at": now, "updated_at": now}
        _atomic_write_json(path, record)
        return record

    def update(self, run_id: str, status: str, **fields: Any) -> dict[str, Any]:
        path = self.path_for(run_id)
        if not path.exists():
            raise FileNotFoundError(f"Run is not registered: {run_id}")
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["status"] == "complete" and status != "complete":
            raise ValueError(f"Completed run cannot be replaced: {run_id}")
        _validate_finite(fields)
        record.update(fields)
        record["status"] = status
        record["updated_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_write_json(path, record)
        return record

    def records(self) -> list[dict[str, Any]]:
        records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(self.directory.glob("*.json"))]
        ids = [record["run_id"] for record in records]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate run IDs found in registry")
        return records
