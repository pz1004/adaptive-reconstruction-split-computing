#!/usr/bin/env python3
"""Audit completed primary raw evidence and publish idempotent audit artifacts."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import load_contract
from src.result_audit import audit_primary_results


DEFAULT_ROOT = PROJECT_ROOT / "revisions" / "2026-07-22_adaptive_reconstruction_study"
DEFAULT_CONTRACT = PROJECT_ROOT / "config" / "study_contract.json"
OUTPUT_NAMES = (
    "primary_result_audit.json",
    "primary_result_audit.md",
    "raw_evidence_sha256.txt",
    "primary_seed_results.csv",
)


def _json_bytes(report: dict[str, Any]) -> bytes:
    return (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _markdown_bytes(report: dict[str, Any]) -> bytes:
    counts = report["verification_counts"]
    lines = [
        "# Completed Primary Result Audit",
        "",
        f"Audit verdict: **{'PASS' if report['passed'] else 'FAIL'}**",
        "",
        "This audit checks preregistered identity and raw-evidence integrity. Outcome direction, effect size, and p-values are not integrity gates.",
        "",
        "## Verified evidence",
        "",
        f"- Primary records: `{report['contract_matrix']['registered_records']}/{report['contract_matrix']['expected_records']}`",
        f"- Per-image metric JSON files: `{counts['metric_json_verified']}/160`",
        f"- Sample bundles: `{counts['sample_bundles_verified']}/160`",
        f"- Encoder checkpoints: `{counts['encoder_checkpoints_verified']}/40`",
        f"- Attacker checkpoints: `{counts['attacker_checkpoints_verified']}/160`",
        f"- Learned-perturbation checkpoints: `{counts['learned_perturbation_checkpoints_verified']}/20`",
        f"- Raw artifacts hashed: `{counts['raw_artifacts_hashed']}/620`",
        "",
        "## Gates",
        "",
    ]
    for name, gate in report["gates"].items():
        lines.append(f"- {'PASS' if gate['pass'] else 'FAIL'}: `{name}` ({gate['error_count']} errors)")
    utility = report["descriptive_test_utility"]
    lines.extend(
        [
            "",
            "## Utility boundary",
            "",
            f"- Frozen validation selections within the one-point constraint: `{'yes' if report['validation_utility_selection']['pass'] else 'no'}`",
            f"- Descriptive test seed losses above one percentage point: `{utility['exceedance_count']}`",
            "- Test behavior is descriptive only and was not used to reselect an operating point.",
            "",
            "## Quarantine",
            "",
            f"- Quarantined run IDs: `{len(report['quarantined_run_ids'])}`",
        ]
    )
    if report["quarantined_run_ids"]:
        lines.extend(f"  - `{run_id}`" for run_id in report["quarantined_run_ids"])
    lines.extend(["", "## Issues", ""])
    if report["issues"]:
        for entry in report["issues"]:
            run = f" `{entry['run_id']}`" if entry.get("run_id") else ""
            lines.append(f"- `{entry['gate']}`{run}: {entry['message']}")
    else:
        lines.append("- None.")
    lines.append("")
    return "\n".join(lines).encode("utf-8")


def _raw_hash_bytes(report: dict[str, Any]) -> bytes:
    lines = [f"{row['sha256']}  {row['path']}" for row in report["raw_evidence"]]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _csv_bytes(report: dict[str, Any]) -> bytes:
    fieldnames = [
        "run_id",
        "dataset",
        "split_point",
        "seed",
        "method",
        "selected_defense_value",
        "accuracy_percentage_points",
        "utility_loss_vs_standard_percentage_points",
        "test_utility_loss_exceeds_validation_limit",
        "worst_case_ssim",
        "strongest_attacker",
        "validation_selected_attacker",
        "record_passed",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(report["primary_seed_results"])
    return stream.getvalue().encode("utf-8")


def render_outputs(report: dict[str, Any]) -> dict[str, bytes]:
    return {
        "primary_result_audit.json": _json_bytes(report),
        "primary_result_audit.md": _markdown_bytes(report),
        "raw_evidence_sha256.txt": _raw_hash_bytes(report),
        "primary_seed_results.csv": _csv_bytes(report),
    }


def publish_idempotent(output_dir: Path, outputs: dict[str, bytes], *, check: bool) -> dict[str, str]:
    states: dict[str, str] = {}
    conflicts: list[str] = []
    for name, expected in outputs.items():
        path = output_dir / name
        if path.exists():
            if not path.is_file() or path.read_bytes() != expected:
                conflicts.append(str(path))
            else:
                states[name] = "verified_existing"
        elif check:
            conflicts.append(f"{path} (missing)")
        else:
            states[name] = "pending_create"
    if conflicts:
        raise RuntimeError("Audit output conflict; no artifact was overwritten: " + ", ".join(conflicts))
    if check:
        return states
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in outputs.items():
        if states[name] == "pending_create":
            with (output_dir / name).open("xb") as handle:
                handle.write(payload)
            states[name] = "created"
    return states


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--revision-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--check", action="store_true", help="Read-only: rerun the audit and verify matching existing outputs")
    args = parser.parse_args()

    contract = load_contract(args.contract)
    report = audit_primary_results(contract, args.revision_root)
    outputs = render_outputs(report)
    try:
        states = publish_idempotent(args.output_dir, outputs, check=args.check)
    except RuntimeError as error:
        print(json.dumps({"passed": False, "output_error": str(error)}, indent=2))
        raise SystemExit(3) from error
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "issues": len(report["issues"]),
                "quarantined_run_ids": len(report["quarantined_run_ids"]),
                "verification_counts": report["verification_counts"],
                "output_dir": str(args.output_dir),
                "output_states": states,
                "check_mode": args.check,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
