#!/usr/bin/env python3
"""Audit the 24 budget cells and publish canonical outputs only after a clean pass."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.budget_result_audit import audit_budget_results
from src.metrics import METRIC_FIELDS
from src.pipeline import load_contract


DEFAULT_ROOT = PROJECT_ROOT / "revisions" / "2026-07-22_adaptive_reconstruction_study"
DEFAULT_CONTRACT = PROJECT_ROOT / "config" / "study_contract.json"
OUTPUT_NAMES = (
    "budget_result_audit.json",
    "budget_result_audit.md",
    "budget_raw_evidence_sha256.txt",
    "budget_cells.csv",
    "budget_attackers.csv",
)


def _json_bytes(report: dict[str, Any]) -> bytes:
    return (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _markdown_bytes(report: dict[str, Any]) -> bytes:
    counts = report["verification_counts"]
    lines = [
        "# Completed Auxiliary-Budget Result Audit",
        "",
        f"Audit verdict: **{'PASS' if report['passed'] else 'FAIL'}**",
        "",
        "This audit checks the fixed matrix, deterministic nested subsets, primary-encoder dependencies, training histories, per-image metrics, matched samples, and all artifact hashes. Result direction is not an integrity gate.",
        "",
        "## Verified evidence",
        "",
        f"- Budget manifests: `{counts['budget_manifests_verified']}/24`",
        f"- Attacker checkpoints: `{counts['budget_attacker_checkpoints_verified']}/48`",
        f"- Training logs: `{counts['budget_training_logs_verified']}/48`",
        f"- Per-image metric files: `{counts['budget_metric_json_verified']}/48`",
        f"- Matched sample bundles: `{counts['budget_sample_bundles_verified']}/48`",
        f"- Production artifacts hashed: `{counts['budget_artifacts_hashed']}/216`",
        f"- Unique seed-42 primary encoders: `{counts['unique_primary_encoders_verified']}/8`",
        "",
        "## Gates",
        "",
    ]
    for name, gate in report["gates"].items():
        lines.append(f"- {'PASS' if gate['pass'] else 'FAIL'}: `{name}` ({gate['error_count']} errors)")
    lines.extend(
        [
            "",
            "## Statistical boundary",
            "",
            "- The 24 cells are fixed conditions at one predeclared seed, not inferential replicates.",
            "- The two attackers form an adversary family; attackers, images, and auxiliary examples are not replicates.",
            "- No confidence intervals, hypothesis tests, adjusted p-values, effect sizes, or rankings are authorized.",
            "",
            "## Quarantine",
            "",
            f"- Quarantined run IDs: `{len(report['quarantined_run_ids'])}`",
        ]
    )
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


def _csv_bytes(rows: list[dict[str, Any]], fieldnames: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def render_outputs(report: dict[str, Any]) -> dict[str, bytes]:
    cell_fields = [
        "run_id",
        "dataset",
        "split_point",
        "seed",
        "method",
        "auxiliary_fraction",
        "auxiliary_examples",
        "auxiliary_fitting_examples",
        "auxiliary_validation_examples",
        "auxiliary_index_digest",
        "auxiliary_fitting_index_digest",
        "auxiliary_validation_index_digest",
        "primary_encoder_path",
        "primary_encoder_sha256",
        "deconv_mse_ssim",
        "residual_lpips_ssim",
        "strongest_attacker",
        "worst_case_test_ssim",
        "record_passed",
    ]
    attacker_fields = [
        "run_id",
        "dataset",
        "split_point",
        "seed",
        "method",
        "auxiliary_fraction",
        "auxiliary_examples",
        "auxiliary_fitting_examples",
        "auxiliary_validation_examples",
        "auxiliary_index_digest",
        "auxiliary_fitting_index_digest",
        "auxiliary_validation_index_digest",
        "architecture",
        "selected_epoch",
        "epochs_completed",
        "best_validation_loss",
        *METRIC_FIELDS,
        "record_passed",
    ]
    hashes = "\n".join(f"{row['sha256']}  {row['path']}" for row in report["raw_evidence"]) + "\n"
    return {
        "budget_result_audit.json": _json_bytes(report),
        "budget_result_audit.md": _markdown_bytes(report),
        "budget_raw_evidence_sha256.txt": hashes.encode("utf-8"),
        "budget_cells.csv": _csv_bytes(report["budget_cell_results"], cell_fields),
        "budget_attackers.csv": _csv_bytes(report["budget_attacker_results"], attacker_fields),
    }


def publish_idempotent(
    output_dir: Path,
    outputs: dict[str, bytes],
    *,
    check: bool,
    replace_reviewed: bool = False,
) -> dict[str, str]:
    states: dict[str, str] = {}
    conflicts: list[str] = []
    for name, payload in outputs.items():
        path = output_dir / name
        if path.exists():
            if not path.is_file() or path.read_bytes() != payload:
                if replace_reviewed and path.is_file() and not check:
                    states[name] = "pending_reviewed_replacement"
                else:
                    conflicts.append(str(path))
            else:
                states[name] = "verified_existing"
        elif check:
            conflicts.append(f"{path} (missing)")
        else:
            states[name] = "pending_create"
    if conflicts:
        raise RuntimeError("Budget audit output conflict; no artifact was overwritten: " + ", ".join(conflicts))
    if check:
        return states
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in outputs.items():
        if states[name] in {"pending_create", "pending_reviewed_replacement"}:
            path = output_dir / name
            with tempfile.NamedTemporaryFile("wb", dir=output_dir, delete=False) as handle:
                handle.write(payload)
                temporary = Path(handle.name)
            try:
                if states[name] == "pending_reviewed_replacement":
                    os.replace(temporary, path)
                    states[name] = "replaced_after_review"
                else:
                    os.link(temporary, path)
                    states[name] = "created"
            finally:
                temporary.unlink(missing_ok=True)
    return states


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--replace-reviewed",
        action="store_true",
        help="Replace a preserved v2 audit only after reviewer-driven auditor changes.",
    )
    args = parser.parse_args()
    report = audit_budget_results(load_contract(args.contract), args.revision_root)
    if not report["passed"]:
        print(
            json.dumps(
                {
                    "passed": False,
                    "issues": len(report["issues"]),
                    "quarantined_run_ids": report["quarantined_run_ids"],
                    "verification_counts": report["verification_counts"],
                    "output_states": "not_published_failed_audit",
                },
                indent=2,
                sort_keys=True,
            )
        )
        raise SystemExit(2)
    try:
        states = publish_idempotent(
            args.output_dir,
            render_outputs(report),
            check=args.check,
            replace_reviewed=args.replace_reviewed,
        )
    except RuntimeError as error:
        print(json.dumps({"passed": False, "output_error": str(error)}, indent=2))
        raise SystemExit(3) from error
    print(
        json.dumps(
            {
                "passed": True,
                "verification_counts": report["verification_counts"],
                "output_dir": str(args.output_dir),
                "output_states": states,
                "check_mode": args.check,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
