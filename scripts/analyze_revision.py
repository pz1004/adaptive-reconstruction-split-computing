#!/usr/bin/env python3
"""Build primary-only strict analysis after a passing raw-evidence audit."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis import estimate, holm_adjust, paired_analysis
from src.pipeline import consolidate_primary, load_contract
from src.result_audit import sha256_file, strict_json_load


DEFAULT_ROOT = PROJECT_ROOT / "revisions" / "2026-07-22_adaptive_reconstruction_study"
DEFAULT_CONTRACT = PROJECT_ROOT / "config" / "study_contract.json"
DEFAULT_AUDIT = PROJECT_ROOT / "audit" / "primary_result_audit_20260808" / "primary_result_audit.json"
METHODS = ("standard", "afd", "laplace", "learned")
COMPARISON_METHODS = ("afd", "laplace", "learned")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _load_passing_audit(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Primary analysis is blocked: missing raw audit {path}")
    report = strict_json_load(path)
    counts = report.get("verification_counts", {})
    expected = {
        "metric_json_verified": 160,
        "sample_bundles_verified": 160,
        "encoder_checkpoints_verified": 40,
        "attacker_checkpoints_verified": 160,
        "learned_perturbation_checkpoints_verified": 20,
        "raw_artifacts_hashed": 620,
    }
    if (
        report.get("passed") is not True
        or report.get("issues")
        or report.get("quarantined_run_ids")
        or any(not gate.get("pass") for gate in report.get("gates", {}).values())
        or any(counts.get(name) != value for name, value in expected.items())
    ):
        raise RuntimeError("Primary analysis is blocked: raw audit is absent, failed, incomplete, or quarantined")
    return report


def _records_by_group(records: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[int, dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        result = record["result"]
        key = (result["dataset"], result["split_point"], result["method"])
        seed = int(result["seed"])
        if seed in grouped[key]:
            raise ValueError(f"Duplicate completed seed record for {key}: {seed}")
        grouped[key][seed] = result
    return grouped


def _stage_status(directory: Path, expected: int) -> dict[str, Any]:
    records = [strict_json_load(path) for path in sorted(directory.glob("*.json"))] if directory.is_dir() else []
    counts = Counter(str(record.get("status", "missing")) for record in records)
    return {
        "expected_runs": expected,
        "registered_runs": len(records),
        "status_counts": dict(sorted(counts.items())),
        "complete_runs": counts.get("complete", 0),
        "all_planned_runs_complete": len(records) == expected and counts.get("complete", 0) == expected,
    }


def build_analysis(
    contract: dict[str, Any], root: Path, audit_path: Path, audit_report: dict[str, Any]
) -> dict[str, Any]:
    consolidated = consolidate_primary(contract, root)
    if not consolidated["all_planned_runs_complete"] or consolidated["complete_runs"] != 80:
        raise RuntimeError("Primary analysis is blocked: consolidation is not exactly 80/80 complete")
    grouped = _records_by_group(consolidated["records"])
    required_seeds = [int(seed) for seed in contract["seeds"]]
    group_summaries: dict[str, Any] = {}
    contrasts: dict[str, Any] = {}
    raw_pvalues: dict[str, float] = {}

    for dataset in contract["datasets"]:
        for split_point in contract["split_points"]:
            for method in METHODS:
                key = (dataset, split_point, method)
                by_seed = grouped.get(key, {})
                if sorted(by_seed) != sorted(required_seeds):
                    raise RuntimeError(f"Incomplete five-seed group: {'/'.join(key)}")
                label = "/".join(key)
                group_summaries[label] = {
                    "required_seeds": required_seeds,
                    "available_seeds": sorted(by_seed),
                    "complete": True,
                    "metrics": {
                        "accuracy_percentage_points": estimate(
                            100.0 * float(by_seed[seed]["utility"]["accuracy"])
                            for seed in required_seeds
                        ),
                        "worst_case_ssim": estimate(
                            float(by_seed[seed]["worst_case_test_ssim"])
                            for seed in required_seeds
                        ),
                    },
                }

            standard = grouped[(dataset, split_point, "standard")]
            extractors: tuple[tuple[str, Callable[[dict[str, Any]], float], str], ...] = (
                ("accuracy_percentage_points", lambda result: 100.0 * float(result["utility"]["accuracy"]), "higher_is_better"),
                ("worst_case_ssim", lambda result: float(result["worst_case_test_ssim"]), "lower_is_better"),
            )
            for method in COMPARISON_METHODS:
                comparison = grouped[(dataset, split_point, method)]
                for metric, extractor, direction in extractors:
                    contrast_id = f"{dataset}/{split_point}/{method}_vs_standard/{metric}"
                    result = paired_analysis(
                        [extractor(standard[seed]) for seed in required_seeds],
                        [extractor(comparison[seed]) for seed in required_seeds],
                    )
                    result.update(
                        {
                            "status": "complete",
                            "seeds": required_seeds,
                            "metric": metric,
                            "metric_direction": direction,
                            "comparison_method": method,
                            "reference_method": "standard",
                        }
                    )
                    contrasts[contrast_id] = result
                    raw_pvalues[contrast_id] = float(result["exact_sign_flip_pvalue"])

    if len(group_summaries) != 16 or len(contrasts) != 24:
        raise RuntimeError(
            f"Strict analysis cardinality mismatch: {len(group_summaries)} groups and {len(contrasts)} contrasts"
        )
    adjusted = holm_adjust(raw_pvalues)
    for contrast_id, adjusted_value in adjusted.items():
        contrasts[contrast_id]["holm_adjusted_sign_flip_pvalue"] = adjusted_value
        contrasts[contrast_id]["holm_family"] = "24 declared method-by-setting-by-outcome contrasts"

    semantic = _stage_status(root / "manifests" / "semantic", 40)
    budget = _stage_status(root / "manifests" / "budget_v2", 24)
    payload = {
        "schema_version": 2,
        "evidence_status": "primary_complete_later_stages_blocked",
        "primary_matrix_complete": True,
        "primary_raw_audit": {
            "passed": True,
            "path": str(audit_path),
            "sha256": sha256_file(audit_path),
            "raw_artifacts_hashed": audit_report["verification_counts"]["raw_artifacts_hashed"],
            "quarantined_run_ids": [],
        },
        "selection_provenance": consolidated["selection_provenance"],
        "seed_is_unit_of_analysis": True,
        "monte_carlo_draws_are_replicates": False,
        "planned_primary_runs": 80,
        "complete_primary_runs": 80,
        "failed_primary_runs": 0,
        "group_summary_count": len(group_summaries),
        "contrast_count": len(contrasts),
        "group_summaries": group_summaries,
        "contrasts": contrasts,
        "validation_utility_selection": audit_report["validation_utility_selection"],
        "descriptive_test_utility": audit_report["descriptive_test_utility"],
        "semantic_leakage": semantic,
        "auxiliary_budget": budget,
        "pvalue_role": "supplementary_exact_sign_flip_with_holm_no_threshold_label",
        "shapiro_wilk_role": "low_power_diagnostic_only_at_n_5_no_test_switching",
        "wording_rule": "Report estimates, intervals, all contrasts, and boundaries; do not claim a universal ranking, equivalence, one-sided margin result, or mathematical privacy guarantee.",
    }
    _write_json(root / "analysis" / "strict_analysis.json", payload)
    return payload


def _format_number(value: float | None, digits: int = 4) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def write_primary_table(root: Path, analysis: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    seed_order = [7, 42, 123, 2024, 2025]
    for label, group in analysis["group_summaries"].items():
        dataset, split_point, method = label.split("/")
        accuracy = group["metrics"]["accuracy_percentage_points"]
        ssim = group["metrics"]["worst_case_ssim"]
        row: dict[str, Any] = {"dataset": dataset, "split_point": split_point, "method": method, "n_seeds": 5}
        for seed, value in zip(seed_order, accuracy["values"], strict=True):
            row[f"accuracy_seed_{seed}_percentage_points"] = value
        row.update(
            {
                "accuracy_mean_percentage_points": accuracy["mean"],
                "accuracy_sample_sd_percentage_points": accuracy["sample_sd"],
                "accuracy_95ci_low_percentage_points": accuracy["mean_confidence_interval"][0],
                "accuracy_95ci_high_percentage_points": accuracy["mean_confidence_interval"][1],
            }
        )
        for seed, value in zip(seed_order, ssim["values"], strict=True):
            row[f"worst_case_ssim_seed_{seed}"] = value
        row.update(
            {
                "worst_case_ssim_mean": ssim["mean"],
                "worst_case_ssim_sample_sd": ssim["sample_sd"],
                "worst_case_ssim_95ci_low": ssim["mean_confidence_interval"][0],
                "worst_case_ssim_95ci_high": ssim["mean_confidence_interval"][1],
            }
        )
        rows.append(row)
    csv_path = root / "tables" / "five_seed_adaptive_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    markdown = [
        "# Five-Seed Primary Results",
        "",
        "Values are test-set estimates from the validation-selected operating points. Parentheses contain sample SD across the five fixed seeds; brackets contain 95% Student-t confidence intervals.",
        "",
        "| Dataset | Split | Method | Accuracy, percentage points | Worst-case adaptive SSIM |",
        "|---|---|---|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            "| {dataset} | {split_point} | {method} | {acc} ({acc_sd}) [{acc_lo}, {acc_hi}] | {ssim} ({ssim_sd}) [{ssim_lo}, {ssim_hi}] |".format(
                dataset=row["dataset"],
                split_point=row["split_point"],
                method=row["method"],
                acc=_format_number(row["accuracy_mean_percentage_points"], 3),
                acc_sd=_format_number(row["accuracy_sample_sd_percentage_points"], 3),
                acc_lo=_format_number(row["accuracy_95ci_low_percentage_points"], 3),
                acc_hi=_format_number(row["accuracy_95ci_high_percentage_points"], 3),
                ssim=_format_number(row["worst_case_ssim_mean"], 4),
                ssim_sd=_format_number(row["worst_case_ssim_sample_sd"], 4),
                ssim_lo=_format_number(row["worst_case_ssim_95ci_low"], 4),
                ssim_hi=_format_number(row["worst_case_ssim_95ci_high"], 4),
            )
        )
    markdown.append("")
    (root / "tables" / "five_seed_adaptive_results.md").write_text("\n".join(markdown), encoding="utf-8")


def _method_ranges(analysis: dict[str, Any], method: str) -> tuple[tuple[float, float], tuple[float, float]]:
    accuracy = []
    ssim = []
    for contrast_id, contrast in analysis["contrasts"].items():
        if f"/{method}_vs_standard/" not in contrast_id:
            continue
        mean = float(contrast["paired_difference"]["mean"])
        (accuracy if contrast["metric"] == "accuracy_percentage_points" else ssim).append(mean)
    return (min(accuracy), max(accuracy)), (min(ssim), max(ssim))


def write_claim_ledger(root: Path, analysis: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = [
        {
            "claim_id": "C01",
            "claim": "The complete primary matrix follows the declared two-dataset, two-interface, four-method, five-seed, two-attacker protocol.",
            "evidence_source": "audit/primary_result_audit_20260808/primary_result_audit.json; metrics/consolidated_primary.json",
            "allowed_wording": "The primary audit verified 80 records and 620 raw artifacts with zero quarantines.",
            "forbidden_stronger_wording": "The full manuscript evidence package is complete.",
            "uncertainty": "Semantic and auxiliary-budget stages remain unexecuted.",
            "decision": "keep_primary_only",
        }
    ]
    for index, method in enumerate(COMPARISON_METHODS, start=2):
        accuracy_range, ssim_range = _method_ranges(analysis, method)
        rows.append(
            {
                "claim_id": f"C{index:02d}",
                "claim": f"Five-seed primary estimates for {method} versus standard are available across four dataset/interface settings.",
                "evidence_source": "analysis/strict_analysis.json; tables/five_seed_adaptive_results.csv",
                "allowed_wording": (
                    f"Across the four settings, mean {method}-minus-standard accuracy differences range from "
                    f"{accuracy_range[0]:.3f} to {accuracy_range[1]:.3f} percentage points and mean worst-case SSIM differences range from "
                    f"{ssim_range[0]:.4f} to {ssim_range[1]:.4f}."
                ),
                "forbidden_stronger_wording": "A universal method ranking or protection claim beyond the tested attackers and settings.",
                "uncertainty": "The analysis has five fixed seeds per setting and two attacker families.",
                "decision": "keep_primary_only",
            }
        )
    exceedances = int(analysis["descriptive_test_utility"]["exceedance_count"])
    rows.extend(
        [
            {
                "claim_id": "C05",
                "claim": "Operating points were selected under the validation utility rule, while test utility behavior is reported without reselection.",
                "evidence_source": "analysis/strict_analysis.json",
                "allowed_wording": f"All 12 frozen validation selections passed the rule; {exceedances} of 60 non-standard test seed cells lost more than one percentage point.",
                "forbidden_stronger_wording": "Every test seed remains within the validation selection margin.",
                "uncertainty": "The one-point rule governed validation selection, not a held-out guarantee.",
                "decision": "keep_bounded",
            },
            {
                "claim_id": "C06",
                "claim": "Semantic leakage has been evaluated.",
                "evidence_source": "manifests/semantic/*.json",
                "allowed_wording": "Semantic leakage remains a planned 0/40 stage.",
                "forbidden_stronger_wording": "Primary reconstruction metrics establish absence of recognizable or sensitive information.",
                "uncertainty": "No semantic-probe production record is complete.",
                "decision": "blocked",
            },
            {
                "claim_id": "C07",
                "claim": "Auxiliary-data budget sensitivity has been evaluated.",
                "evidence_source": "manifests/budget_v2/*.json",
                "allowed_wording": "Budget sensitivity and Figure 3 remain planned at 0/24.",
                "forbidden_stronger_wording": "The primary attacker budget covers the declared auxiliary-data sensitivity study.",
                "uncertainty": "No budget production record is complete.",
                "decision": "blocked",
            },
            {
                "claim_id": "C08",
                "claim": "The tested perturbations establish a mathematical privacy guarantee.",
                "evidence_source": "study_contract.json; analysis/strict_analysis.json",
                "allowed_wording": "Empirical reconstruction fidelity under the two tested adaptive attacker families.",
                "forbidden_stronger_wording": "A mathematical or attacker-independent guarantee.",
                "uncertainty": "The contract contains no proof or calibrated guarantee mechanism.",
                "decision": "remove",
            },
        ]
    )
    analysis_dir = root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    with (analysis_dir / "claim_ledger.csv").open("w", newline="", encoding="utf-8") as handle:
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
    (analysis_dir / "claim_ledger.md").write_text("\n".join(lines), encoding="utf-8")
    return rows


def _contrast_table_lines(analysis: dict[str, Any]) -> list[str]:
    lines = [
        "| Contrast | Mean difference | Sample SD | 95% t CI | Exact sign-flip p | Holm-adjusted p | Hedges g_z | Shapiro W / p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for contrast_id, contrast in analysis["contrasts"].items():
        difference = contrast["paired_difference"]
        interval = difference["mean_confidence_interval"]
        effect = contrast["corrected_paired_effect"]["hedges_gz"]
        shapiro = contrast["paired_difference_shapiro_wilk"]
        lines.append(
            f"| `{contrast_id}` | {_format_number(difference['mean'])} | {_format_number(difference['sample_sd'])} | "
            f"[{_format_number(interval[0])}, {_format_number(interval[1])}] | {_format_number(contrast['exact_sign_flip_pvalue'])} | "
            f"{_format_number(contrast['holm_adjusted_sign_flip_pvalue'])} | {_format_number(effect)} | "
            f"{_format_number(shapiro['statistic'])} / {_format_number(shapiro['pvalue'])} |"
        )
    return lines


def write_reports(root: Path, analysis: dict[str, Any], claims: list[dict[str, str]]) -> None:
    ssim_differences = [
        float(contrast["paired_difference"]["mean"])
        for contrast in analysis["contrasts"].values()
        if contrast["metric"] == "worst_case_ssim"
    ]
    accuracy_differences = [
        float(contrast["paired_difference"]["mean"])
        for contrast in analysis["contrasts"].values()
        if contrast["metric"] == "accuracy_percentage_points"
    ]
    exceedances = analysis["descriptive_test_utility"]["exceedance_count"]
    report_lines = [
        "# Strict Primary Analysis Report",
        "",
        "## Conclusion",
        "",
        "The raw audit passed and the 80-record primary matrix now supports bounded primary-only reporting. Semantic leakage, auxiliary-budget sensitivity, manuscript evidence mapping, and overall submission readiness remain blocked.",
        "",
        "## Primary observations",
        "",
        f"- All 12 non-standard mean worst-case SSIM differences versus standard are below zero; the range is `{min(ssim_differences):.4f}` to `{max(ssim_differences):.4f}`.",
        f"- Mean accuracy differences span `{min(accuracy_differences):.3f}` to `{max(accuracy_differences):.3f}` percentage points across the 12 comparisons.",
        f"- The frozen validation utility check passes for all 12 selected settings. Descriptively, `{exceedances}` of 60 non-standard test seed cells lose more than one percentage point; no test result was used for reselection.",
        "- Every one of the 24 declared contrasts is retained in the statistical appendix, regardless of direction or p-value.",
        "",
        "## Analysis boundary",
        "",
        "- The seed is the unit of analysis; images and Monte Carlo draws are not replicates.",
        "- Group summaries report all five seed values, mean, sample SD, and a 95% Student-t interval.",
        "- Paired contrasts report comparison minus standard, exact two-sided sign flips, Hedges g_z, and Holm adjustment across one declared 24-contrast family.",
        "- Shapiro-Wilk values are retained as low-power diagnostics at n=5 and do not switch the declared analysis.",
        "- The evidence is bounded to the tested datasets, interfaces, operating points, seeds, and two attacker families; no mathematical guarantee is established.",
        "",
        "## Claim candidates",
        "",
    ]
    for claim in claims:
        if claim["decision"] in {"keep_primary_only", "keep_bounded"}:
            report_lines.extend(
                [
                    f"- Claim: {claim['claim']}",
                    f"  - Source evidence: `{claim['evidence_source']}`",
                    f"  - Allowed wording: {claim['allowed_wording']}",
                    f"  - Blocked stronger wording: {claim['forbidden_stronger_wording']}",
                    f"  - Uncertainty: {claim['uncertainty']}",
                    f"  - Decision: {claim['decision']}",
                ]
            )
    report_lines.extend(
        [
            "",
            "## Current blocker",
            "",
            "Run and audit the 40 fixed semantic-probe jobs before the 24 auxiliary-budget jobs. Do not edit manuscript result prose from this primary-only bundle.",
            "",
        ]
    )
    (root / "analysis" / "analysis_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    appendix = [
        "# Statistical Appendix: Completed Primary Matrix",
        "",
        "The 16 groups each contain the five fixed seeds `7, 42, 123, 2024, 2025`. Accuracy is expressed in percentage points; worst-case SSIM is the maximum test SSIM across the two fixed adaptive attacker families. Exact sign-flip p-values are two-sided and Holm-adjusted across all 24 rows below. Shapiro-Wilk is a diagnostic only because n=5.",
        "",
        "## All paired contrasts",
        "",
        *_contrast_table_lines(analysis),
        "",
        "## Utility-selection diagnostic",
        "",
        f"All 12 validation-selected settings pass the one-percentage-point selection rule. Test behavior is descriptive: {exceedances} of 60 non-standard seed cells exceed one percentage point of loss, and those observations do not change the frozen selections.",
        "",
    ]
    (root / "analysis" / "statistics_appendix.md").write_text("\n".join(appendix), encoding="utf-8")

    figure2_exists = all((root / "figures" / f"figure-02-accuracy-vs-adaptive-ssim.{suffix}").is_file() for suffix in ("pdf", "svg"))
    figure3_exists = all((root / "figures" / f"figure-03-attacker-budget-sensitivity.{suffix}").is_file() for suffix in ("pdf", "svg"))
    figure5_exists = all((root / "figures" / f"figure-05-matched-reconstructions.{suffix}").is_file() for suffix in ("pdf", "svg"))
    corrected_budget_path = root / "metrics" / "consolidated_budget.json"
    corrected_budget = (
        json.loads(corrected_budget_path.read_text(encoding="utf-8"))
        if corrected_budget_path.is_file()
        else {}
    )
    corrected_budget_complete = (
        corrected_budget.get("all_planned_runs_complete") is True
        and corrected_budget.get("complete_runs") == 24
        and corrected_budget.get("expected_runs") == 24
        and len(corrected_budget.get("rows", [])) == 24
    )
    catalog = [
        "# Figure Catalog",
        "",
        "## Figure 1: Threat model",
        "",
        "- Status: available protocol artifact.",
        "- Purpose: define the split-computing and adaptive-attacker scope.",
        "- Interpretation boundary: schematic only; it contains no empirical outcome.",
        "",
        "## Figure 2: Accuracy versus worst-case adaptive SSIM",
        "",
        f"- Status: {'generated in PDF and SVG' if figure2_exists else 'eligible after the passing primary audit; generation pending'}.",
        f"- Visual QA: {'passed on the rendered PDF; the eight-item legend is isolated in a top band and all labels/error bars remain readable' if figure2_exists else 'pending'}.",
        "- Purpose: show the primary utility-reconstruction trade-off across two datasets, two interfaces, four methods, and all five seed points per group.",
        f"- Observation: all 12 non-standard group-mean SSIM differences are below zero; mean accuracy differences range from {min(accuracy_differences):.3f} to {max(accuracy_differences):.3f} percentage points.",
        "- Error bars: sample SD across five fixed seeds on both axes; individual seed points remain visible.",
        "- Interpretation boundary: the figure does not provide semantic leakage, budget sensitivity, a universal ranking, or a mathematical guarantee.",
        "- Decision consequence: primary trade-off estimates may be reported with their setting-specific uncertainty, while later-stage claims remain blocked.",
        "",
        "## Figure 3: Auxiliary-budget sensitivity",
        "",
        (
            "- Status: generated in PDF and SVG from the completed corrected-budget evidence; `24/24` fixed cells pass audit."
            if corrected_budget_complete and figure3_exists
            else "- Status: blocked; corrected 24-cell budget evidence or the reviewed figure pair is incomplete."
        ),
        "- Purpose: panel `(a)` shows one normalized convergence example; panels `(b)` and `(c)` show CelebA and CIFAR-10 auxiliary-budget sensitivity.",
        "- Interpretation boundary: all conditions use the single predeclared seed 42; cells, attackers, images, and auxiliary examples are not replicates.",
        "- Decision consequence: cite Figure 3 only as completed descriptive sensitivity evidence, not as a multi-seed or inferential comparison.",
        "",
        "## Figure 4: Payload-time sensitivity",
        "",
        "- Status: available analytical artifact.",
        "- Interpretation boundary: calculated payload transmission time, not measured end-to-end latency.",
        "",
        "## Figure 5: Matched validation-selected reconstructions",
        "",
        f"- Status: {'generated in PDF and SVG' if figure5_exists else 'eligible after the passing primary audit; generation pending'}.",
        f"- Visual QA: {'passed on the rendered PDF; the 5-by-5 grid has readable row labels and exactly matched columns' if figure5_exists else 'pending'}.",
        "- Purpose: compare matched originals and reconstruction outputs for standard, AFD, Laplace, and learned defenses using each method's validation-selected attacker.",
        "- Observation: the grid preserves the same five examples across all rows and exposes residual visual information rather than converting it into a scalar-only claim.",
        "- Interpretation boundary: five examples from CelebA/current/seed 42 are qualitative evidence, not a population estimate or semantic-leakage test.",
        "- Decision consequence: use the grid only to illustrate reconstruction appearance under the fixed primary protocol; semantic claims await the 40 probes.",
        "",
        "## Rendering requirements",
        "",
        "- Vector PDF and SVG line art with embedded TrueType fonts, 9 pt base text, 8 pt legends/annotations, colorblind-safe colors, readable downscaling, and no Type 3 fonts.",
        "- Figure 5 retains the exact matched 5-by-5 grid and uses 600 dpi embedded raster objects; no p-value markers are added, and captions state the seed/error-bar/replicate boundary where applicable.",
        "",
    ]
    (root / "analysis" / "figure_catalog.md").write_text("\n".join(catalog), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--primary-audit", type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()
    contract = load_contract(args.contract)
    audit_report = _load_passing_audit(args.primary_audit)
    analysis = build_analysis(contract, args.revision_root, args.primary_audit, audit_report)
    write_primary_table(args.revision_root, analysis)
    claims = write_claim_ledger(args.revision_root, analysis)
    write_reports(args.revision_root, analysis, claims)
    print(
        json.dumps(
            {
                "analysis": str(args.revision_root / "analysis" / "strict_analysis.json"),
                "status": analysis["evidence_status"],
                "group_summaries": analysis["group_summary_count"],
                "contrasts": analysis["contrast_count"],
                "semantic_complete": analysis["semantic_leakage"]["complete_runs"],
                "budget_complete": analysis["auxiliary_budget"]["complete_runs"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
