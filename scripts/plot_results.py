#!/usr/bin/env python3
"""Generate deterministic reviewed-style Figure 2 from audited primary results."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "revisions" / "2026-07-22_adaptive_reconstruction_study"
DEFAULT_AUDIT = PROJECT_ROOT / "audit" / "primary_result_audit_20260808" / "primary_result_audit.json"
CHECKPOINT_ROOT = PROJECT_ROOT / "audit" / "figure_revision_20260822" / "pre_change"
CHECKPOINT_REVISION = (
    CHECKPOINT_ROOT / "copies" / "revisions" / "2026-07-22_adaptive_reconstruction_study"
)
CANONICAL_FIGURE_ROOT = DEFAULT_ROOT / "figures"
COLORS = {"standard": "#000000", "afd": "#0072B2", "laplace": "#D55E00", "learned": "#009E73"}
MARKERS = {"early": "o", "current": "s"}
SEEDS = (7, 42, 123, 2024, 2025)
BASE_STYLE: dict[str, Any] = {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "svg.hashsalt": "ml2-figure-02-reviewed-v1",
}


def _require_checkpoint_match(live: Path, preserved: Path, label: str) -> None:
    if not live.is_file() or not preserved.is_file():
        raise RuntimeError(f"Figure 2 is blocked: missing checkpointed {label}")
    if live.read_bytes() != preserved.read_bytes():
        raise RuntimeError(f"Figure 2 is blocked: {label} differs from the pre-change checkpoint")


def _require_passing_audit(path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"Figure 2 is blocked: missing primary audit {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("passed") is not True or report.get("issues") or report.get("quarantined_run_ids"):
        raise RuntimeError("Figure 2 is blocked: primary raw audit did not pass cleanly")


def _figure_bytes(figure: Any) -> dict[str, bytes]:
    stem = "figure-02-accuracy-vs-adaptive-ssim"
    title = "Accuracy versus maximum observed reconstruction similarity"
    subject = "All five model seeds per group with sample-SD error bars on both axes."
    with tempfile.TemporaryDirectory(prefix="ml2-figure-02-") as temporary:
        root = Path(temporary)
        pdf = root / f"{stem}.pdf"
        svg = root / f"{stem}.svg"
        figure.savefig(
            pdf,
            dpi=600,
            bbox_inches="tight",
            metadata={
                "Title": title,
                "Author": "",
                "Subject": subject,
                "Creator": "ml2 reviewed figure generator",
                "Producer": "matplotlib",
                "CreationDate": None,
                "ModDate": None,
            },
        )
        figure.savefig(
            svg,
            dpi=600,
            bbox_inches="tight",
            metadata={"Title": title, "Description": subject, "Date": None},
        )
        return {pdf.name: pdf.read_bytes(), svg.name: svg.read_bytes()}


def _publish(
    output_dir: Path,
    outputs: dict[str, bytes],
    *,
    check: bool,
    replace_reviewed_style: bool,
) -> dict[str, str]:
    if replace_reviewed_style and output_dir.resolve() != CANONICAL_FIGURE_ROOT.resolve():
        raise RuntimeError("Figure 2 reviewed-style replacement is restricted to the canonical figure directory")
    states: dict[str, str] = {}
    conflicts: list[str] = []
    for name, payload in outputs.items():
        path = output_dir / name
        if path.is_file() and path.read_bytes() == payload:
            states[name] = "verified_existing"
        elif check:
            conflicts.append(f"{path} (missing or byte-different)")
        elif path.exists() and not replace_reviewed_style:
            conflicts.append(f"{path} (replacement requires --replace-reviewed-style)")
        else:
            states[name] = "pending_replacement" if path.exists() else "pending_create"
    if conflicts:
        raise RuntimeError("Figure 2 output conflict; no artifact was written: " + ", ".join(conflicts))
    if check:
        return states
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in outputs.items():
        if not states[name].startswith("pending_"):
            continue
        path = output_dir / name
        with tempfile.NamedTemporaryFile("wb", dir=output_dir, delete=False) as handle:
            handle.write(payload)
            temporary = Path(handle.name)
        try:
            if states[name] == "pending_replacement":
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
    parser.add_argument("--primary-audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output-dir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument(
        "--replace-reviewed-style",
        action="store_true",
        help="Atomically replace canonical Figure 2 after checkpointed scientific inputs match.",
    )
    args = parser.parse_args()
    revision_root = args.revision_root.resolve()
    primary_audit = args.primary_audit.resolve()
    _require_passing_audit(primary_audit)
    _require_checkpoint_match(
        primary_audit,
        CHECKPOINT_ROOT / "copies" / "audit" / "primary_result_audit_20260808" / "primary_result_audit.json",
        "primary audit",
    )
    consolidated_path = revision_root / "metrics" / "consolidated_primary.json"
    _require_checkpoint_match(
        consolidated_path,
        CHECKPOINT_REVISION / "metrics" / "consolidated_primary.json",
        "consolidated primary results",
    )
    consolidated = json.loads(consolidated_path.read_text(encoding="utf-8"))
    if not consolidated.get("all_planned_runs_complete") or consolidated.get("complete_runs") != 80:
        raise RuntimeError(
            f"Figure 2 is blocked: {consolidated.get('complete_runs', 0)}/{consolidated.get('expected_runs', 80)} primary jobs complete"
        )

    with plt.rc_context(BASE_STYLE):
        figure, axes = plt.subplots(1, 2, figsize=(7.5, 3.45), sharey=False)
        verified_groups = 0
        for axis, dataset, panel in zip(axes, ("celeba", "cifar10"), ("a", "b"), strict=True):
            for method, color in COLORS.items():
                for split_point, marker in MARKERS.items():
                    rows = sorted(
                        [
                            record["result"]
                            for record in consolidated["records"]
                            if record["result"]["dataset"] == dataset
                            and record["result"]["split_point"] == split_point
                            and record["result"]["method"] == method
                        ],
                        key=lambda row: int(row["seed"]),
                    )
                    if tuple(int(row["seed"]) for row in rows) != SEEDS:
                        raise RuntimeError(
                            f"Figure 2 group {dataset}/{split_point}/{method} does not contain the five fixed seeds"
                        )
                    verified_groups += 1
                    x = np.asarray([row["worst_case_test_ssim"] for row in rows], dtype=float)
                    y = np.asarray([row["utility"]["accuracy"] * 100.0 for row in rows], dtype=float)
                    display_method = "AFD" if method == "afd" else method.capitalize()
                    label = f"{display_method}, {split_point}"
                    axis.scatter(
                        x,
                        y,
                        facecolors="none" if method == "standard" else color,
                        edgecolors=color,
                        marker=marker,
                        s=28,
                        linewidths=0.9,
                        alpha=0.85,
                        label=label,
                        zorder=2,
                    )
                    axis.errorbar(
                        x.mean(),
                        y.mean(),
                        xerr=x.std(ddof=1),
                        yerr=y.std(ddof=1),
                        color=color,
                        marker=marker,
                        markerfacecolor="white" if method == "standard" else color,
                        markeredgecolor=color,
                        markersize=6.0,
                        capsize=2.5,
                        linewidth=1.1,
                        zorder=3,
                    )
            axis.text(0.02, 0.97, f"({panel})", transform=axis.transAxes, va="top", fontweight="bold", fontsize=9)
            axis.text(
                0.98,
                0.97,
                "CelebA Smiling" if dataset == "celeba" else "CIFAR-10",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontweight="bold",
                fontsize=9,
            )
            axis.set_xlabel("Maximum observed SSIM\n(lower is better)")
            axis.set_ylabel("Test accuracy (%)")
            axis.grid(axis="both", color="#D0D0D0", linestyle="--", linewidth=0.55, alpha=0.8)
        if verified_groups != 16:
            raise RuntimeError(f"Figure 2 expected 16 groups, found {verified_groups}")
        handles, labels = axes[1].get_legend_handles_labels()
        figure.subplots_adjust(left=0.08, right=0.99, bottom=0.16, top=0.75, wspace=0.23)
        legend_axis = figure.add_axes((0.04, 0.79, 0.92, 0.18))
        legend_axis.axis("off")
        legend_axis.legend(handles, labels, loc="center", ncol=4, frameon=False, fontsize=8)
        outputs = _figure_bytes(figure)
        plt.close(figure)

    output_dir = (args.output_dir or revision_root / "figures").resolve()
    states = _publish(
        output_dir,
        outputs,
        check=args.check,
        replace_reviewed_style=args.replace_reviewed_style,
    )
    print(
        json.dumps(
            {
                "passed": True,
                "check_mode": args.check,
                "replace_reviewed_style": args.replace_reviewed_style,
                "groups": verified_groups,
                "seed_points_per_group": 5,
                "error_bars": "sample SD across five fixed seeds on both axes",
                "pvalue_markers": False,
                "palette": "Okabe-Ito",
                "output_states": states,
                "output_sha256": {name: hashlib.sha256(payload).hexdigest() for name, payload in outputs.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
