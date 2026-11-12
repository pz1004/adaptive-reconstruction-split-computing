from __future__ import annotations
from typing import Any
from pathlib import Path
import tempfile
import matplotlib.pyplot as plt
import numpy as np
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

def render(consolidated):
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

    return outputs
