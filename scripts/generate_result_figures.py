#!/usr/bin/env python3
"""Generate audited Figure 3 from the canonical budget audit and consolidation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_budget_results import build_budget_analysis, require_matching_consolidated
from src.budget_result_audit import require_canonical_budget_audit
from src.pipeline import load_contract
from src.result_audit import strict_json_load


DEFAULT_ROOT = PROJECT_ROOT / "revisions" / "2026-07-22_adaptive_reconstruction_study"
DEFAULT_CONTRACT = PROJECT_ROOT / "config" / "study_contract.json"
CHECKPOINT_ROOT = PROJECT_ROOT / "audit" / "figure_revision_20260822" / "pre_change"
CHECKPOINT_REVISION = (
    CHECKPOINT_ROOT / "copies" / "revisions" / "2026-07-22_adaptive_reconstruction_study"
)
CANONICAL_FIGURE_ROOT = DEFAULT_ROOT / "figures"
COLORS = {"standard": "#000000", "afd": "#0072B2"}
LINESTYLES = {"early": "--", "current": "-"}
MARKERS = {"early": "o", "current": "s"}
ATTACKER_COLORS = {"deconv_mse": "#D55E00", "residual_lpips": "#009E73"}
ATTACKER_STYLES = {"deconv_mse": "-", "residual_lpips": "--"}
BOUNDARY_TEXT = "Single predeclared seed 42; descriptive sensitivity; no error bars."


def _require_checkpoint_match(live: Path, preserved: Path, label: str) -> None:
    if not live.is_file() or not preserved.is_file():
        raise RuntimeError(f"Figure 3 is blocked: missing checkpointed {label}")
    if live.read_bytes() != preserved.read_bytes():
        raise RuntimeError(f"Figure 3 is blocked: {label} differs from the pre-change checkpoint")


def _require_corrected_budget_raw_artifact(path: Path) -> None:
    raw_manifest = (
        CHECKPOINT_ROOT
        / "copies"
        / "audit"
        / "budget_result_audit_v2_20260816"
        / "budget_raw_evidence_sha256.txt"
    )
    if not raw_manifest.is_file():
        raise RuntimeError("Figure 3 is blocked: checkpointed corrected-budget raw hash list is missing")
    try:
        relative = path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError as error:
        raise RuntimeError(f"Figure 3 is blocked: raw input escapes the project root: {path}") from error
    rows: dict[str, str] = {}
    for line in raw_manifest.read_text(encoding="utf-8").splitlines():
        digest, separator, artifact = line.partition("  ")
        if separator != "  " or len(digest) != 64 or artifact in rows:
            raise RuntimeError("Figure 3 is blocked: malformed corrected-budget raw hash list")
        rows[artifact] = digest
    expected = rows.get(relative)
    if expected is None:
        raise RuntimeError(f"Figure 3 is blocked: input is absent from the corrected-budget hash list: {relative}")
    if not path.is_file():
        raise RuntimeError(f"Figure 3 is blocked: corrected-budget raw input is missing: {relative}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise RuntimeError(f"Figure 3 is blocked: corrected-budget raw input hash differs: {relative}")


def build_figure_data(audit: dict[str, Any], revision_root: Path) -> dict[str, Any]:
    analysis = build_budget_analysis(audit)
    representative = next(
        row
        for row in analysis["rows"]
        if row["dataset"] == "cifar10"
        and row["split_point"] == "current"
        and row["method"] == "standard"
        and float(row["auxiliary_fraction"]) == 1.0
    )
    manifest_path = revision_root / "manifests" / "budget_v2" / f"{representative['run_id']}.json"
    _require_corrected_budget_raw_artifact(manifest_path)
    manifest = strict_json_load(manifest_path)
    convergence = []
    for architecture in ("deconv_mse", "residual_lpips"):
        history = manifest["result"]["attackers"][architecture]["training"]["history"]
        convergence.append(
            {
                "architecture": architecture,
                "epochs": [int(row["epoch"]) for row in history],
                "validation_loss": [float(row["validation_loss"]) for row in history],
                "normalized_validation_loss": [
                    float(row["validation_loss"]) / float(history[0]["validation_loss"]) for row in history
                ],
            }
        )
    panels = []
    for dataset in ("celeba", "cifar10"):
        series = []
        for method in ("standard", "afd"):
            for split_point in ("early", "current"):
                selected = sorted(
                    (
                        row
                        for row in analysis["rows"]
                        if row["dataset"] == dataset
                        and row["method"] == method
                        and row["split_point"] == split_point
                    ),
                    key=lambda row: float(row["auxiliary_fraction"]),
                )
                if len(selected) != 3:
                    raise RuntimeError(f"Figure 3 is blocked: incomplete series {dataset}/{method}/{split_point}")
                series.append(
                    {
                        "method": method,
                        "split_point": split_point,
                        "auxiliary_fraction": [float(row["auxiliary_fraction"]) for row in selected],
                        "auxiliary_examples": [int(row["auxiliary_examples"]) for row in selected],
                        "worst_case_test_ssim": [float(row["worst_case_test_ssim"]) for row in selected],
                    }
                )
        panels.append({"dataset": dataset, "series": series})
    return {
        "schema_version": 2,
        "figure": "figure-03-attacker-budget-sensitivity",
        "source_audit": "audit/budget_result_audit_v2_20260816/budget_result_audit.json",
        "source_consolidation": "metrics/consolidated_budget.json",
        "seed": 42,
        "boundary_text": BOUNDARY_TEXT,
        "fixed_conditions_are_replicates": False,
        "attackers_are_replicates": False,
        "images_are_replicates": False,
        "auxiliary_examples_are_replicates": False,
        "error_bars": False,
        "inferential_annotations": False,
        "convergence_example": {
            "dataset": "cifar10",
            "split_point": "current",
            "method": "standard",
            "auxiliary_fraction": 1.0,
            "run_id": representative["run_id"],
            "attackers": convergence,
        },
        "sensitivity_panels": panels,
        "style": {
            "method_colors": COLORS,
            "split_line_styles": LINESTYLES,
            "split_markers": MARKERS,
            "attacker_colors": ATTACKER_COLORS,
            "attacker_line_styles": ATTACKER_STYLES,
        },
    }


def render_figure(data: dict[str, Any], output_dir: Path) -> dict[str, bytes]:
    data_bytes = (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    data_sha256 = hashlib.sha256(data_bytes).hexdigest()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "ml2-budget-figure-03-reviewed-v1",
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(8.4, 3.35))
    convergence_axis = axes[0]
    for series in data["convergence_example"]["attackers"]:
        architecture = series["architecture"]
        convergence_axis.plot(
            np.asarray(series["epochs"]),
            np.asarray(series["normalized_validation_loss"], dtype=float),
            color=ATTACKER_COLORS[architecture],
            linestyle=ATTACKER_STYLES[architecture],
            marker="o" if architecture == "deconv_mse" else "s",
            markevery=max(1, len(series["epochs"]) // 5),
            markersize=2.8,
            linewidth=1.2,
            label={"deconv_mse": "Deconvolutional MSE", "residual_lpips": "Residual LPIPS"}[architecture],
        )
    convergence_axis.set_xlabel("Attacker epoch")
    convergence_axis.set_ylabel("Validation loss / epoch-1 loss")
    convergence_axis.text(
        0.0,
        1.03,
        "(a)",
        transform=convergence_axis.transAxes,
        va="bottom",
        fontweight="bold",
        fontsize=9,
    )
    convergence_axis.grid(axis="y", color="#D0D0D0", linestyle=":", linewidth=0.6)
    convergence_axis.legend(frameon=False, fontsize=8, loc="best")

    panel_by_dataset = {panel["dataset"]: panel for panel in data["sensitivity_panels"]}
    for axis, dataset, panel_label in zip(
        axes[1:], ("celeba", "cifar10"), ("(b) CelebA", "(c) CIFAR-10"), strict=True
    ):
        for series in panel_by_dataset[dataset]["series"]:
            method = series["method"]
            split_point = series["split_point"]
            axis.plot(
                100.0 * np.asarray(series["auxiliary_fraction"], dtype=float),
                np.asarray(series["worst_case_test_ssim"], dtype=float),
                color=COLORS[method],
                linestyle=LINESTYLES[split_point],
                marker=MARKERS[split_point],
                markersize=4.0,
                linewidth=1.2,
                label=f"{'AFD' if method == 'afd' else 'Standard'}, {split_point}",
            )
        axis.set_xticks((10, 50, 100))
        axis.set_xlabel("Auxiliary training data (%)")
        axis.text(
            0.0,
            1.03,
            panel_label,
            transform=axis.transAxes,
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )
        axis.grid(axis="y", color="#D0D0D0", linestyle=":", linewidth=0.6)
    axes[1].set_ylabel("Maximum observed SSIM\n(two configurations)")
    handles, labels = axes[2].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.69, 0.99), ncol=2, frameon=False, fontsize=8)
    figure.text(0.5, 0.035, BOUNDARY_TEXT, ha="center", va="bottom", fontsize=8)
    figure.subplots_adjust(left=0.075, right=0.99, bottom=0.24, top=0.79, wspace=0.34)

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / "figure.pdf"
    svg = output_dir / "figure.svg"
    subject = f"data_sha256={data_sha256}; {BOUNDARY_TEXT}"
    figure.savefig(
        pdf,
        metadata={
            "Title": "Auxiliary attacker-budget sensitivity",
            "Author": "",
            "Creator": "ml2 audited figure generator",
            "Producer": "matplotlib",
            "CreationDate": None,
            "ModDate": None,
            "Subject": subject,
        },
        dpi=600,
    )
    figure.savefig(
        svg,
        metadata={
            "Title": "Auxiliary attacker-budget sensitivity",
            "Description": subject,
            "Date": None,
        },
        dpi=600,
    )
    plt.close(figure)
    return {
        "figure-03-attacker-budget-sensitivity.data.json": data_bytes,
        "figure-03-attacker-budget-sensitivity.pdf": pdf.read_bytes(),
        "figure-03-attacker-budget-sensitivity.svg": svg.read_bytes(),
    }


def publish_idempotent(
    directory: Path,
    outputs: dict[str, bytes],
    *,
    check: bool,
    replace_reviewed_style: bool = False,
) -> dict[str, str]:
    if replace_reviewed_style and directory.resolve() != CANONICAL_FIGURE_ROOT.resolve():
        raise RuntimeError("Figure 3 reviewed-style replacement is restricted to the canonical figure directory")
    states: dict[str, str] = {}
    conflicts: list[str] = []
    for name, payload in outputs.items():
        path = directory / name
        if path.is_file() and path.read_bytes() == payload:
            states[name] = "verified_existing"
        elif check:
            conflicts.append(f"{path} (missing or byte-different)")
        elif not path.exists():
            states[name] = "pending_create"
        elif replace_reviewed_style and path.is_file():
            states[name] = "pending_reviewed_style_replacement"
        else:
            conflicts.append(str(path))
    if conflicts:
        raise RuntimeError("Figure 3 output conflict; no artifact was overwritten: " + ", ".join(conflicts))
    if check:
        return states
    directory.mkdir(parents=True, exist_ok=True)
    for name, payload in outputs.items():
        if states[name] not in {"pending_create", "pending_reviewed_style_replacement"}:
            continue
        path = directory / name
        with tempfile.NamedTemporaryFile("wb", dir=directory, delete=False) as handle:
            handle.write(payload)
            temporary = Path(handle.name)
        try:
            if states[name] == "pending_reviewed_style_replacement":
                os.replace(temporary, path)
                states[name] = "replaced_reviewed_style"
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
    parser.add_argument(
        "--budget-audit",
        type=Path,
        default=PROJECT_ROOT / "audit" / "budget_result_audit_v2_20260816" / "budget_result_audit.json",
    )
    parser.add_argument("--output-dir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument(
        "--replace-reviewed-style",
        "--promote-corrected",
        dest="replace_reviewed_style",
        action="store_true",
        help="Replace canonical Figure 3 only after the checkpointed scientific inputs match.",
    )
    args = parser.parse_args()
    _require_checkpoint_match(
        args.contract.resolve(),
        CHECKPOINT_ROOT / "copies" / "config" / "study_contract.json",
        "study contract",
    )
    _require_checkpoint_match(
        args.budget_audit.resolve(),
        CHECKPOINT_ROOT / "copies" / "audit" / "budget_result_audit_v2_20260816" / "budget_result_audit.json",
        "corrected budget audit",
    )
    _require_checkpoint_match(
        args.revision_root.resolve() / "metrics" / "consolidated_budget.json",
        CHECKPOINT_REVISION / "metrics" / "consolidated_budget.json",
        "consolidated corrected-budget results",
    )
    audit = require_canonical_budget_audit(load_contract(args.contract), args.revision_root, args.budget_audit)
    analysis = build_budget_analysis(audit)
    consolidated_path = args.revision_root / "metrics" / "consolidated_budget.json"
    try:
        require_matching_consolidated(consolidated_path, analysis)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    data = build_figure_data(audit, args.revision_root)
    with tempfile.TemporaryDirectory(prefix="ml2-budget-figure-") as temporary:
        outputs = render_figure(data, Path(temporary))
    try:
        states = publish_idempotent(
            (args.output_dir or args.revision_root / "figures").resolve(),
            outputs,
            check=args.check,
            replace_reviewed_style=args.replace_reviewed_style,
        )
    except RuntimeError as error:
        print(json.dumps({"passed": False, "output_error": str(error)}, indent=2))
        raise SystemExit(3) from error
    data_sha256 = hashlib.sha256(outputs["figure-03-attacker-budget-sensitivity.data.json"]).hexdigest()
    print(
        json.dumps(
            {
                "passed": True,
                "seed": 42,
                "data_sha256": data_sha256,
                "boundary": BOUNDARY_TEXT,
                "output_states": states,
                "check_mode": args.check,
                "replace_reviewed_style": args.replace_reviewed_style,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
