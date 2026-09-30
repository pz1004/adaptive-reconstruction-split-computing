from __future__ import annotations
from typing import Any
from pathlib import Path
import json,hashlib
import matplotlib.pyplot as plt
import numpy as np
COLORS = {"standard": "#000000", "afd": "#0072B2"}

MARKERS = {"standard": "o", "afd": "s"}

LABELS = {"standard": "Standard", "afd": "AFD"}

BOUNDARY = (
    "Five coupled runs; model and attacker-training seeds paired; unadjusted 95% "
    "Student-t intervals; no confirmatory tests."
)

def render(data: dict[str, Any], output_dir: Path) -> dict[str, bytes]:
    data_bytes = (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    data_hash = hashlib.sha256(data_bytes).hexdigest()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "ml2-benchmark-figure-03-v1",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.15, 4.8), sharex=True)
    panel_labels = ("(a)", "(b)", "(c)", "(d)")
    for axis, panel, panel_label in zip(axes.flat, data["panels"], panel_labels, strict=True):
        for method in ("standard", "afd"):
            offset = -2.2 if method == "standard" else 2.2
            method_conditions = [row for row in panel["conditions"] if row["method"] == method]
            method_summaries = sorted(
                (row for row in panel["summaries"] if row["method"] == method),
                key=lambda row: float(row["auxiliary_fraction"]),
            )
            for index, fraction in enumerate((0.1, 0.5, 1.0)):
                selected = sorted(
                    (row for row in method_conditions if float(row["auxiliary_fraction"]) == fraction),
                    key=lambda row: int(row["model_seed"]),
                )
                if len(selected) != 5:
                    raise RuntimeError("Figure 3 is blocked: a method/fraction cell does not contain five seeds")
                jitter = np.linspace(-0.9, 0.9, 5)
                axis.scatter(
                    100.0 * fraction + offset + jitter,
                    [float(row["worst_case_ssim"]) for row in selected],
                    s=16,
                    marker=MARKERS[method],
                    facecolors="none" if method == "standard" else COLORS[method],
                    edgecolors=COLORS[method],
                    linewidths=0.8,
                    alpha=0.82,
                    zorder=2,
                )
            x = np.asarray([100.0 * float(row["auxiliary_fraction"]) + offset for row in method_summaries])
            mean = np.asarray([float(row["mean"]) for row in method_summaries])
            lower = np.asarray([float(row["ci95_lower"]) for row in method_summaries])
            upper = np.asarray([float(row["ci95_upper"]) for row in method_summaries])
            axis.errorbar(
                x,
                mean,
                yerr=np.vstack((mean - lower, upper - mean)),
                color=COLORS[method],
                marker=MARKERS[method],
                markersize=4.8,
                linewidth=1.3,
                capsize=2.8,
                label=LABELS[method],
                zorder=3,
            )
        dataset_label = "CelebA" if panel["dataset"] == "celeba" else "CIFAR-10"
        axis.set_title(f"{panel_label} {dataset_label}, {panel['interface']} interface", loc="left", fontweight="bold")
        axis.set_xticks((10, 50, 100))
        axis.grid(axis="y", color="#D0D0D0", linestyle=":", linewidth=0.6)
    axes[0, 0].set_ylabel("Maximum observed SSIM\n(two configurations)")
    axes[1, 0].set_ylabel("Maximum observed SSIM\n(two configurations)")
    axes[1, 0].set_xlabel("Auxiliary training data (%)")
    axes[1, 1].set_xlabel("Auxiliary training data (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=2, frameon=False)
    figure.text(0.5, 0.018, BOUNDARY, ha="center", va="bottom", fontsize=8)
    figure.subplots_adjust(left=0.10, right=0.985, bottom=0.15, top=0.88, wspace=0.20, hspace=0.36)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / "figure.pdf"
    svg = output_dir / "figure.svg"
    subject = f"data_sha256={data_hash}; {BOUNDARY}"
    figure.savefig(
        pdf,
        dpi=600,
        metadata={
            "Title": "Five-seed auxiliary attacker-budget sensitivity",
            "Author": "",
            "Creator": "ml2 benchmark-v1 figure generator",
            "Producer": "matplotlib",
            "CreationDate": None,
            "ModDate": None,
            "Subject": subject,
        },
    )
    figure.savefig(svg, dpi=600, metadata={"Title": "Five-seed auxiliary attacker-budget sensitivity", "Description": subject, "Date": None})
    plt.close(figure)
    return {
        "figure-03-attacker-budget-sensitivity.data.json": data_bytes,
        "figure-03-attacker-budget-sensitivity.pdf": pdf.read_bytes(),
        "figure-03-attacker-budget-sensitivity.svg": svg.read_bytes(),
    }
