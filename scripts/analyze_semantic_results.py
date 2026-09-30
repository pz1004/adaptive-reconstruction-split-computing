#!/usr/bin/env python3
"""Extend the audited primary analysis with bounded five-seed semantic results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import load_contract
from src.result_audit import sha256_file, strict_json_load
from src.semantic_analysis import SEMANTIC_COMPARISON_METHODS, build_semantic_analysis


DEFAULT_ROOT = PROJECT_ROOT / "revisions" / "2026-07-22_adaptive_reconstruction_study"
DEFAULT_CONTRACT = PROJECT_ROOT / "config" / "study_contract.json"
DEFAULT_AUDIT = PROJECT_ROOT / "audit" / "semantic_result_audit_20260809" / "semantic_result_audit.json"
EXPECTED_PRIMARY_CONTRAST_DIGEST = "c7a90157b133b3c2140175ef163a20cc83fb818c251dd6f1a5870d5ad82ce6ba"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _load_semantic_audit(path: Path) -> dict[str, Any]:
    report = strict_json_load(path)
    counts = report.get("verification_counts", {})
    expected = {
        "semantic_manifests_verified": 40,
        "semantic_checkpoints_verified": 40,
        "semantic_raw_bundles_verified": 40,
        "semantic_seed_rows": 40,
        "semantic_attribute_rows": 1560,
        "semantic_artifacts_hashed": 120,
    }
    if (
        report.get("passed") is not True
        or report.get("issues")
        or report.get("quarantined_run_ids")
        or any(counts.get(name) != value for name, value in expected.items())
    ):
        raise RuntimeError("Semantic analysis is blocked: the canonical raw audit is absent, failed, or incomplete")
    return report


def _format(value: float | None, digits: int = 4) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def _contrast_lines(semantic: dict[str, Any]) -> list[str]:
    lines = [
        "| Contrast | Mean seed difference | Sample SD | 95% t CI | Exact sign-flip p | Holm-adjusted p | Hedges g_z | Shapiro W / p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for contrast_id, contrast in semantic["contrasts"].items():
        difference = contrast["paired_difference"]
        interval = difference["mean_confidence_interval"]
        shapiro = contrast["paired_difference_shapiro_wilk"]
        lines.append(
            f"| `{contrast_id}` | {_format(difference['mean'])} | {_format(difference['sample_sd'])} | "
            f"[{_format(interval[0])}, {_format(interval[1])}] | {_format(contrast['exact_sign_flip_pvalue'])} | "
            f"{_format(contrast['holm_adjusted_sign_flip_pvalue'])} | {_format(contrast['corrected_paired_effect']['hedges_gz'])} | "
            f"{_format(shapiro['statistic'])} / {_format(shapiro['pvalue'])} |"
        )
    return lines


def _semantic_ranges(semantic: dict[str, Any], method: str) -> tuple[tuple[float, float], tuple[float, float]]:
    auroc: list[float] = []
    balanced: list[float] = []
    for contrast_id, contrast in semantic["contrasts"].items():
        if f"/{method}_vs_standard/" not in contrast_id:
            continue
        target = auroc if contrast["metric"] == "macro_auroc" else balanced
        target.append(float(contrast["paired_difference"]["mean"]))
    return (min(auroc), max(auroc)), (min(balanced), max(balanced))


def _update_claim_ledger(root: Path, semantic: dict[str, Any]) -> list[dict[str, str]]:
    path = root / "analysis" / "claim_ledger.csv"
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row["claim_id"] == "C01":
            row["forbidden_stronger_wording"] = "The full manuscript evidence package is complete."
            row["uncertainty"] = "The auxiliary-budget stage and manuscript mapping remain incomplete."
        if row["claim_id"] == "C06":
            ranges = []
            for method in SEMANTIC_COMPARISON_METHODS:
                auroc, balanced = _semantic_ranges(semantic, method)
                ranges.append(
                    f"{method}: AUROC {auroc[0]:.4f} to {auroc[1]:.4f}, balanced accuracy {balanced[0]:.4f} to {balanced[1]:.4f}"
                )
            row.update(
                {
                    "evidence_source": "audit/semantic_result_audit_20260809/semantic_result_audit.json; analysis/strict_analysis.json",
                    "allowed_wording": "Across the two interfaces, mean five-seed method-minus-standard differences were " + "; ".join(ranges) + ".",
                    "forbidden_stronger_wording": "The tested representations contain no sensitive information, or one method universally eliminates semantic leakage.",
                    "uncertainty": "The probes cover 39 CelebA attributes, two interfaces, four methods, and five fixed seeds; they do not establish information-theoretic privacy.",
                    "decision": "keep_bounded",
                }
            )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Claim Ledger", ""]
    for row in rows:
        lines.extend(
            [
                f"## {row['claim_id']}: {row['decision']}",
                "",
                f"- Claim: {row['claim']}",
                f"- Evidence: `{row['evidence_source']}`",
                f"- Allowed: {row['allowed_wording']}",
                f"- Blocked stronger wording: {row['forbidden_stronger_wording']}",
                f"- Uncertainty: {row['uncertainty']}",
                "",
            ]
        )
    (root / "analysis" / "claim_ledger.md").write_text("\n".join(lines), encoding="utf-8")
    return rows


def publish_semantic_analysis(
    contract: dict[str, Any], root: Path, audit_path: Path, audit: dict[str, Any]
) -> dict[str, Any]:
    strict_path = root / "analysis" / "strict_analysis.json"
    primary = strict_json_load(strict_path)
    primary_contrasts_before = json.dumps(primary.get("contrasts"), sort_keys=True, separators=(",", ":"))
    primary_contrast_digest = hashlib.sha256(primary_contrasts_before.encode("utf-8")).hexdigest()
    if (
        primary.get("contrast_count") != 24
        or len(primary.get("contrasts", {})) != 24
        or primary_contrast_digest != EXPECTED_PRIMARY_CONTRAST_DIGEST
    ):
        raise RuntimeError("Semantic analysis is blocked: the unchanged 24-contrast primary family is unavailable")
    semantic = build_semantic_analysis(audit["semantic_seed_results"], [int(seed) for seed in contract["seeds"]])
    primary["schema_version"] = 3
    primary["evidence_status"] = "primary_and_semantic_complete_budget_blocked"
    primary["semantic_leakage"] = {
        "expected_runs": 40,
        "registered_runs": 40,
        "complete_runs": 40,
        "all_planned_runs_complete": True,
        "raw_audit": {
            "passed": True,
            "path": str(audit_path),
            "sha256": sha256_file(audit_path),
            "semantic_artifacts_hashed": 120,
            "quarantined_run_ids": [],
        },
        **semantic,
    }
    primary["semantic_group_summary_count"] = semantic["group_summary_count"]
    primary["semantic_contrast_count"] = semantic["contrast_count"]
    primary["semantic_pvalue_role"] = "exact_two_sided_sign_flip_with_holm_across_the_separate_12_contrast_semantic_family"
    if json.dumps(primary.get("contrasts"), sort_keys=True, separators=(",", ":")) != primary_contrasts_before:
        raise RuntimeError("Primary contrast family changed while adding semantic analysis")
    _write_json(strict_path, primary)

    consolidated = {
        "schema_version": 2,
        "expected_runs": 40,
        "registered_runs": 40,
        "complete_runs": 40,
        "all_planned_runs_complete": True,
        "unit_of_analysis": "seed",
        "images_are_replicates": False,
        "attributes_are_replicates": False,
        "draws_are_replicates": False,
        "selection_provenance": audit["selection_provenance"],
        "semantic_raw_audit": primary["semantic_leakage"]["raw_audit"],
        "seed_results": audit["semantic_seed_results"],
        "group_summaries": semantic["group_summaries"],
        "contrasts": semantic["contrasts"],
    }
    _write_json(root / "metrics" / "consolidated_semantic.json", consolidated)

    claims = _update_claim_ledger(root, semantic)
    report = (root / "analysis" / "analysis_report.md").read_text(encoding="utf-8")
    report = report.replace("# Strict Primary Analysis Report", "# Strict Primary and Semantic Analysis Report", 1)
    report = report.replace(
        "The raw audit passed and the 80-record primary matrix now supports bounded primary-only reporting. Semantic leakage, auxiliary-budget sensitivity, manuscript evidence mapping, and overall submission readiness remain blocked.",
        "The primary and semantic raw audits passed. The 80-record primary matrix and 40-record semantic matrix support bounded reporting; auxiliary-budget sensitivity, manuscript evidence mapping, and overall submission readiness remain blocked.",
    )
    report = report.replace(
        "Run and audit the 40 fixed semantic-probe jobs before the 24 auxiliary-budget jobs. Do not edit manuscript result prose from this primary-only bundle.",
        "Plan a fresh preflight and monitored execution for the 24 fixed auxiliary-budget jobs. Do not launch budget work or edit manuscript result prose in this semantic execution.",
    )
    report += "\n## Semantic observations\n\n"
    for label, group in semantic["group_summaries"].items():
        auroc = group["metrics"]["macro_auroc"]
        balanced = group["metrics"]["macro_balanced_accuracy"]
        report += (
            f"- `{label}`: macro AUROC {_format(auroc['mean'])} (SD {_format(auroc['sample_sd'])}, "
            f"95% CI [{_format(auroc['mean_confidence_interval'][0])}, {_format(auroc['mean_confidence_interval'][1])}]); "
            f"macro balanced accuracy {_format(balanced['mean'])} (SD {_format(balanced['sample_sd'])}, "
            f"95% CI [{_format(balanced['mean_confidence_interval'][0])}, {_format(balanced['mean_confidence_interval'][1])}]).\n"
        )
    report += (
        "\nAll 12 method-versus-standard semantic contrasts are retained. Exact sign-flip p-values and Holm adjustments belong to a separate semantic family; the existing 24 primary contrasts are unchanged. Images, attributes, and draws are not replicates.\n"
    )
    (root / "analysis" / "analysis_report.md").write_text(report, encoding="utf-8")

    appendix_path = root / "analysis" / "statistics_appendix.md"
    appendix = appendix_path.read_text(encoding="utf-8")
    appendix += "\n## Semantic five-seed contrasts\n\n"
    appendix += (
        "Macro AUROC and macro balanced accuracy use the same five fixed model seeds as paired units. The 12 rows below form their own Holm family. Shapiro-Wilk is retained only as an n=5 low-power diagnostic; images, attributes, and draws are not replicates.\n\n"
    )
    appendix += "\n".join(_contrast_lines(semantic)) + "\n"
    appendix_path.write_text(appendix, encoding="utf-8")

    catalog_path = root / "analysis" / "figure_catalog.md"
    catalog = catalog_path.read_text(encoding="utf-8")
    catalog = catalog.replace(
        "- Decision consequence: use the grid only to illustrate reconstruction appearance under the fixed primary protocol; semantic claims await the 40 probes.",
        "- Decision consequence: use the grid only to illustrate reconstruction appearance; semantic estimates come from the separate audited 40-probe evidence base.",
    )
    catalog_path.write_text(catalog, encoding="utf-8")
    return {"strict_analysis": primary, "semantic_analysis": semantic, "claims": claims}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--semantic-audit", type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()
    audit = _load_semantic_audit(args.semantic_audit)
    result = publish_semantic_analysis(
        load_contract(args.contract), args.revision_root, args.semantic_audit, audit
    )
    print(
        json.dumps(
            {
                "status": result["strict_analysis"]["evidence_status"],
                "semantic_groups": result["semantic_analysis"]["group_summary_count"],
                "semantic_contrasts": result["semantic_analysis"]["contrast_count"],
                "primary_contrasts_unchanged": len(result["strict_analysis"]["contrasts"]),
                "budget_complete": result["strict_analysis"]["auxiliary_budget"]["complete_runs"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
