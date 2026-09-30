#!/usr/bin/env python3
"""Audit completed semantic evidence and publish idempotent canonical outputs."""

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
from src.semantic_result_audit import audit_semantic_results


DEFAULT_ROOT = PROJECT_ROOT / "revisions" / "2026-07-22_adaptive_reconstruction_study"
DEFAULT_CONTRACT = PROJECT_ROOT / "config" / "study_contract.json"
OUTPUT_NAMES = (
    "semantic_result_audit.json",
    "semantic_result_audit.md",
    "semantic_raw_evidence_sha256.txt",
    "semantic_seed_results.csv",
    "semantic_attribute_results.csv",
)


def _json_bytes(report: dict[str, Any]) -> bytes:
    return (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _markdown_bytes(report: dict[str, Any]) -> bytes:
    counts = report["verification_counts"]
    lines = [
        "# Completed Semantic Result Audit",
        "",
        f"Audit verdict: **{'PASS' if report['passed'] else 'FAIL'}**",
        "",
        "This audit checks registered identity, checkpoint state, prediction/target integrity, canonical examples and attributes, draw policy, and metric reproducibility. Scientific outcomes are not integrity gates.",
        "",
        "## Verified evidence",
        "",
        f"- Semantic records: `{counts['semantic_manifests_verified']}/40`",
        f"- Probe checkpoints: `{counts['semantic_checkpoints_verified']}/40`",
        f"- Raw prediction bundles: `{counts['semantic_raw_bundles_verified']}/40`",
        f"- Seed rows: `{counts['semantic_seed_rows']}/40`",
        f"- Attribute rows: `{counts['semantic_attribute_rows']}/1560`",
        f"- Semantic production artifacts hashed: `{counts['semantic_artifacts_hashed']}/120`",
        f"- Unique primary dependencies: `{counts['unique_primary_dependencies_verified']}/30`",
        f"- Laplace five-draw bundles: `{counts['laplace_five_draw_bundles']}/10`",
        f"- Other single-draw bundles: `{counts['single_draw_bundles']}/30`",
        "",
        "## Gates",
        "",
    ]
    for name, gate in report["gates"].items():
        lines.append(f"- {'PASS' if gate['pass'] else 'FAIL'}: `{name}` ({gate['error_count']} errors)")
    lines.extend(
        [
            "",
            "## Analysis-unit boundary",
            "",
            "- The five fixed model seeds are the inferential unit.",
            "- Images, 39 attributes, and stochastic defense draws are not independent replicates.",
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


def _hash_bytes(report: dict[str, Any]) -> bytes:
    return ("\n".join(f"{row['sha256']}  {row['path']}" for row in report["raw_evidence"]) + "\n").encode(
        "utf-8"
    )


def _csv_bytes(rows: list[dict[str, Any]], fieldnames: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def render_outputs(report: dict[str, Any]) -> dict[str, bytes]:
    seed_fields = [
        "run_id",
        "dataset",
        "split_point",
        "seed",
        "method",
        "selected_defense_value",
        "macro_auroc",
        "macro_balanced_accuracy",
        "attribute_auroc_sample_sd",
        "attribute_balanced_accuracy_sample_sd",
        "semantic_draw_count",
        "semantic_draw_seeds",
        "record_passed",
    ]
    attribute_fields = [
        "run_id",
        "dataset",
        "split_point",
        "seed",
        "method",
        "attribute_index",
        "attribute_name",
        "auroc",
        "balanced_accuracy",
    ]
    return {
        "semantic_result_audit.json": _json_bytes(report),
        "semantic_result_audit.md": _markdown_bytes(report),
        "semantic_raw_evidence_sha256.txt": _hash_bytes(report),
        "semantic_seed_results.csv": _csv_bytes(report["semantic_seed_results"], seed_fields),
        "semantic_attribute_results.csv": _csv_bytes(report["semantic_attribute_results"], attribute_fields),
    }


def publish_idempotent(output_dir: Path, outputs: dict[str, bytes], *, check: bool) -> dict[str, str]:
    states: dict[str, str] = {}
    conflicts: list[str] = []
    for name, payload in outputs.items():
        path = output_dir / name
        if path.exists():
            if not path.is_file() or path.read_bytes() != payload:
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
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    report = audit_semantic_results(load_contract(args.contract), args.revision_root)
    if not report["passed"]:
        print(
            json.dumps(
                {
                    "passed": False,
                    "issues": len(report["issues"]),
                    "quarantined_run_ids": report["quarantined_run_ids"],
                    "verification_counts": report["verification_counts"],
                    "output_dir": str(args.output_dir),
                    "output_states": "not_published_failed_audit",
                    "check_mode": args.check,
                },
                indent=2,
                sort_keys=True,
            )
        )
        raise SystemExit(2)
    try:
        states = publish_idempotent(args.output_dir, render_outputs(report), check=args.check)
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


if __name__ == "__main__":
    main()
