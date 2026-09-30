#!/usr/bin/env python3
"""Publish the fixed seed-42 auxiliary-budget results as descriptive evidence only."""

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

from src.budget_result_audit import require_canonical_budget_audit
from src.metrics import METRIC_FIELDS
from src.pipeline import load_contract


DEFAULT_ROOT = PROJECT_ROOT / "revisions" / "2026-07-22_adaptive_reconstruction_study"
DEFAULT_CONTRACT = PROJECT_ROOT / "config" / "study_contract.json"
CSV_FIELDS = [
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
    "strongest_attacker",
    "worst_case_test_ssim",
    *[f"{prefix}_{metric}" for prefix in ("deconv", "residual") for metric in METRIC_FIELDS],
    "afd_minus_standard_worst_case_ssim",
]


def build_budget_analysis(audit: dict[str, Any]) -> dict[str, Any]:
    cells = audit["budget_cell_results"]
    attackers_by_run: dict[str, dict[str, dict[str, Any]]] = {}
    for row in audit["budget_attacker_results"]:
        attackers_by_run.setdefault(str(row["run_id"]), {})[str(row["architecture"])] = row
    if len(cells) != 24 or len(audit["budget_attacker_results"]) != 48:
        raise RuntimeError("Budget analysis is blocked: canonical audit does not contain 24 cells and 48 attackers")
    by_condition = {
        (row["dataset"], row["split_point"], float(row["auxiliary_fraction"]), row["method"]): row
        for row in cells
    }
    differences: list[dict[str, Any]] = []
    for dataset in ("celeba", "cifar10"):
        for split_point in ("early", "current"):
            for fraction in (0.1, 0.5, 1.0):
                standard = by_condition[(dataset, split_point, fraction, "standard")]
                afd = by_condition[(dataset, split_point, fraction, "afd")]
                differences.append(
                    {
                        "dataset": dataset,
                        "split_point": split_point,
                        "auxiliary_fraction": fraction,
                        "seed": 42,
                        "afd_worst_case_test_ssim": afd["worst_case_test_ssim"],
                        "standard_worst_case_test_ssim": standard["worst_case_test_ssim"],
                        "afd_minus_standard_worst_case_ssim": (
                            float(afd["worst_case_test_ssim"]) - float(standard["worst_case_test_ssim"])
                        ),
                    }
                )
    difference_by_condition = {
        (row["dataset"], row["split_point"], row["auxiliary_fraction"]): row[
            "afd_minus_standard_worst_case_ssim"
        ]
        for row in differences
    }
    rows: list[dict[str, Any]] = []
    for cell in cells:
        attacker_rows = attackers_by_run[str(cell["run_id"])]
        if set(attacker_rows) != {"deconv_mse", "residual_lpips"}:
            raise RuntimeError(f"Budget analysis is blocked: incomplete attacker family for {cell['run_id']}")
        row = {
            **cell,
            "attackers": {
                architecture: {metric: attacker_rows[architecture][metric] for metric in METRIC_FIELDS}
                for architecture in ("deconv_mse", "residual_lpips")
            },
            "afd_minus_standard_worst_case_ssim": (
                difference_by_condition[
                    (cell["dataset"], cell["split_point"], float(cell["auxiliary_fraction"]))
                ]
                if cell["method"] == "afd"
                else None
            ),
        }
        rows.append(row)
    rows.sort(key=lambda row: (row["dataset"], row["split_point"], row["method"], row["auxiliary_fraction"]))
    trends: list[dict[str, Any]] = []
    for dataset in ("celeba", "cifar10"):
        for split_point in ("early", "current"):
            for method in ("standard", "afd"):
                series = sorted(
                    (
                        row
                        for row in rows
                        if row["dataset"] == dataset
                        and row["split_point"] == split_point
                        and row["method"] == method
                    ),
                    key=lambda row: float(row["auxiliary_fraction"]),
                )
                values = [float(row["worst_case_test_ssim"]) for row in series]
                trends.append(
                    {
                        "dataset": dataset,
                        "split_point": split_point,
                        "method": method,
                        "worst_case_test_ssim": values,
                        "nondecreasing_with_auxiliary_cap": all(
                            left <= right for left, right in zip(values, values[1:])
                        ),
                    }
                )
    return {
        "schema_version": 2,
        "analysis": "auxiliary_budget_total_cap_v2_seed_42_descriptive_sensitivity",
        "budget_protocol_version": audit.get("budget_protocol_version", "auxiliary_budget_total_cap_v2"),
        "expected_runs": 24,
        "registered_runs": 24,
        "complete_runs": 24,
        "all_planned_runs_complete": True,
        "seed": 42,
        "unit_of_analysis": "single predeclared seed 42 sensitivity analysis; not an inferential replicate set",
        "statistical_boundary": {
            "fixed_conditions_not_replicates": True,
            "attackers_are_adversary_family_not_replicates": True,
            "images_are_not_replicates": True,
            "auxiliary_examples_are_not_replicates": True,
            "standard_deviations": False,
            "confidence_intervals": False,
            "hypothesis_tests": False,
            "multiple_testing_corrections": False,
            "effect_sizes": False,
            "method_rankings": False,
            "privacy_guarantees": False,
        },
        "rows": rows,
        "afd_minus_standard_differences": differences,
        "auxiliary_cap_trends": trends,
        "budget_audit_raw_evidence": audit["raw_evidence"],
    }


def render_outputs(analysis: dict[str, Any]) -> dict[Path, bytes]:
    json_payload = {
        key: analysis[key]
        for key in (
            "schema_version",
            "analysis",
            "budget_protocol_version",
            "expected_runs",
            "registered_runs",
            "complete_runs",
            "all_planned_runs_complete",
            "seed",
            "unit_of_analysis",
            "statistical_boundary",
            "rows",
            "afd_minus_standard_differences",
            "auxiliary_cap_trends",
        )
    }
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in analysis["rows"]:
        flat = {name: row.get(name) for name in CSV_FIELDS}
        for architecture, prefix in (("deconv_mse", "deconv"), ("residual_lpips", "residual")):
            for metric in METRIC_FIELDS:
                flat[f"{prefix}_{metric}"] = row["attackers"][architecture][metric]
        writer.writerow(flat)
    return {
        Path("metrics/consolidated_budget.json"): (
            json.dumps(json_payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8"),
        Path("metrics/consolidated_budget.csv"): stream.getvalue().encode("utf-8"),
        Path("analysis/budget_descriptive_analysis.json"): (
            json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8"),
    }


def require_matching_consolidated(path: Path, analysis: dict[str, Any]) -> dict[str, Any]:
    expected = json.loads(
        render_outputs(analysis)[Path("metrics/consolidated_budget.json")].decode("utf-8")
    )
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        actual = None
    if actual != expected:
        raise RuntimeError(
            "Figure 3 is blocked: consolidated budget payload is missing or differs from the passing audit"
        )
    return expected


def _recognized_json_placeholder(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        payload.get("expected_runs") == 24
        and payload.get("registered_runs") == 24
        and payload.get("complete_runs") == 0
        and payload.get("all_planned_runs_complete") is False
        and payload.get("records") == []
        and payload.get("rows") == []
        and payload.get("unit_of_analysis")
        == "single predeclared seed 42 sensitivity analysis; not an inferential replicate set"
    )


def _recognized_csv_placeholder(path: Path) -> bool:
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
    except OSError:
        return False
    return len(rows) == 1 and rows[0] == [
        "run_id",
        "dataset",
        "split_point",
        "seed",
        "method",
        "auxiliary_fraction",
        "auxiliary_examples",
        "worst_case_test_ssim",
        "strongest_attacker",
    ]


def publish_outputs(
    root: Path,
    outputs: dict[Path, bytes],
    *,
    check: bool,
    promote_corrected: bool = False,
) -> dict[str, str]:
    states: dict[str, str] = {}
    conflicts: list[str] = []
    for relative, payload in outputs.items():
        path = root / relative
        key = str(relative)
        if path.is_file() and path.read_bytes() == payload:
            states[key] = "verified_existing"
        elif check:
            conflicts.append(f"{path} (missing or byte-different)")
        elif relative == Path("metrics/consolidated_budget.json") and path.is_file() and _recognized_json_placeholder(path):
            states[key] = "recognized_placeholder"
        elif relative == Path("metrics/consolidated_budget.csv") and path.is_file() and _recognized_csv_placeholder(path):
            states[key] = "recognized_placeholder"
        elif promote_corrected and path.is_file():
            states[key] = "pending_corrected_replacement"
        elif not path.exists():
            states[key] = "pending_create"
        else:
            conflicts.append(str(path))
    if conflicts:
        raise RuntimeError("Budget analysis output conflict; no artifact was overwritten: " + ", ".join(conflicts))
    if check:
        return states
    for relative, payload in outputs.items():
        state = states[str(relative)]
        if state == "verified_existing":
            continue
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
            handle.write(payload)
            temporary = Path(handle.name)
        try:
            if state in {"recognized_placeholder", "pending_corrected_replacement"}:
                os.replace(temporary, path)
            else:
                os.link(temporary, path)
            states[str(relative)] = (
                "replaced_placeholder"
                if state == "recognized_placeholder"
                else "replaced_superseded_v1"
                if state == "pending_corrected_replacement"
                else "created"
            )
        finally:
            temporary.unlink(missing_ok=True)
    return states


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--budget-audit",
        type=Path,
        default=PROJECT_ROOT / "audit" / "budget_result_audit_v2_20260816" / "budget_result_audit.json",
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--promote-corrected",
        action="store_true",
        help="Replace only the canonical v1-derived budget outputs with passing v2 evidence.",
    )
    args = parser.parse_args()
    audit = require_canonical_budget_audit(load_contract(args.contract), args.revision_root, args.budget_audit)
    analysis = build_budget_analysis(audit)
    try:
        states = publish_outputs(
            args.revision_root,
            render_outputs(analysis),
            check=args.check,
            promote_corrected=args.promote_corrected,
        )
    except RuntimeError as error:
        print(json.dumps({"passed": False, "output_error": str(error)}, indent=2))
        raise SystemExit(3) from error
    print(
        json.dumps(
            {
                "passed": True,
                "rows": len(analysis["rows"]),
                "differences": len(analysis["afd_minus_standard_differences"]),
                "output_states": states,
                "check_mode": args.check,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
