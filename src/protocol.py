"""Portable protocol-integrity helpers shared by public tests and private readiness checks."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def registry_gate(
    directory: Path,
    expected: int,
    selection_provenance: dict[str, str] | None = None,
) -> dict[str, Any]:
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*.json"))
    ] if directory.is_dir() else []
    status_counts: dict[str, int] = {}
    for record in records:
        status = str(record.get("status", "missing"))
        status_counts[status] = status_counts.get(status, 0) + 1
    provenance_errors = [
        f"{record.get('run_id')}: missing or mismatched selection provenance"
        for record in records
        if record.get("status") == "complete"
        and (selection_provenance is None or record.get("selection_provenance") != selection_provenance)
    ]
    return {
        "pass": (
            len(records) == expected
            and status_counts.get("complete", 0) == expected
            and not provenance_errors
        ),
        "expected": expected,
        "registered": len(records),
        "status_counts": status_counts,
        "provenance_errors": provenance_errors,
    }


def primary_integrity_gate(
    directory: Path,
    expected: int,
    selection_provenance: dict[str, str] | None,
) -> dict[str, Any]:
    errors: list[str] = []
    complete = 0
    paths = sorted(directory.glob("*.json")) if directory.is_dir() else []
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "complete":
            continue
        complete += 1
        if selection_provenance is None or record.get("selection_provenance") != selection_provenance:
            errors.append(f"{record.get('run_id')}: missing or mismatched selection provenance")
            continue
        result = record.get("result", {})
        attackers = result.get("attackers", {})
        if set(attackers) != {"deconv_mse", "residual_lpips"}:
            errors.append(f"{record.get('run_id')}: incomplete attacker set")
            continue
        ssim = {
            name: value.get("test_summary", {}).get("ssim")
            for name, value in attackers.items()
        }
        if any(
            not isinstance(value, (int, float)) or not math.isfinite(float(value))
            for value in ssim.values()
        ):
            errors.append(f"{record.get('run_id')}: missing or nonfinite SSIM")
            continue
        strongest = max(ssim, key=ssim.get)
        if (
            result.get("strongest_attacker") != strongest
            or result.get("worst_case_test_ssim") != ssim[strongest]
        ):
            errors.append(f"{record.get('run_id')}: strongest-attacker mismatch")
        if result.get("evaluation_partition") != "test_after_frozen_selection":
            errors.append(f"{record.get('run_id')}: invalid test provenance")
    return {
        "pass": complete == expected and not errors,
        "complete": complete,
        "expected": expected,
        "errors": errors,
    }
