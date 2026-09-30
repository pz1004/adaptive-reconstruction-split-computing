#!/usr/bin/env python3
"""Generate deterministic reviewed-style Figure 5 from audited matched examples."""

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
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "revisions" / "2026-07-22_adaptive_reconstruction_study"
DEFAULT_AUDIT = PROJECT_ROOT / "audit" / "primary_result_audit_20260808" / "primary_result_audit.json"
CHECKPOINT_ROOT = PROJECT_ROOT / "audit" / "figure_revision_20260822" / "pre_change"
CHECKPOINT_REVISION = (
    CHECKPOINT_ROOT / "copies" / "revisions" / "2026-07-22_adaptive_reconstruction_study"
)
CANONICAL_FIGURE_ROOT = DEFAULT_ROOT / "figures"
METHODS = ("standard", "afd", "laplace", "learned")
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
    "svg.hashsalt": "ml2-figure-05-reviewed-v1",
}


def _require_checkpoint_match(live: Path, preserved: Path, label: str) -> None:
    if not live.is_file() or not preserved.is_file():
        raise RuntimeError(f"Figure 5 is blocked: missing checkpointed {label}")
    if live.read_bytes() != preserved.read_bytes():
        raise RuntimeError(f"Figure 5 is blocked: {label} differs from the pre-change checkpoint")


def _require_passing_audit(path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"Figure 5 is blocked: missing primary audit {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("passed") is not True or report.get("issues") or report.get("quarantined_run_ids"):
        raise RuntimeError("Figure 5 is blocked: primary raw audit did not pass cleanly")


def _artifact_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _sealed_raw_rows() -> dict[str, dict[str, Any]]:
    path = CHECKPOINT_ROOT / "raw_artifacts_manifest.json"
    if not path.is_file():
        raise RuntimeError("Figure 5 is blocked: sealed 956-artifact manifest is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("artifact_count") != 956 or len(payload.get("artifacts", [])) != 956:
        raise RuntimeError("Figure 5 is blocked: sealed raw-artifact inventory is not exactly 956")
    return {
        row["path"]: row
        for row in payload["artifacts"]
        if isinstance(row, dict) and isinstance(row.get("path"), str)
    }


def _require_sealed_raw_artifact(path: Path, rows: dict[str, dict[str, Any]]) -> None:
    try:
        relative = path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError as error:
        raise RuntimeError(f"Figure 5 is blocked: sample bundle escapes the project root: {path}") from error
    expected = rows.get(relative)
    if expected is None:
        raise RuntimeError(f"Figure 5 is blocked: sample bundle is absent from the sealed raw manifest: {relative}")
    if not path.is_file() or path.stat().st_size != expected.get("size_bytes"):
        raise RuntimeError(f"Figure 5 is blocked: sample bundle is missing or size-different: {relative}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected.get("sha256"):
        raise RuntimeError(f"Figure 5 is blocked: sample bundle hash differs: {relative}")


def _figure_bytes(figure: Any) -> dict[str, bytes]:
    stem = "figure-05-matched-reconstructions"
    title = "Matched 32 by 32 input and reconstruction grid"
    subject = "Exact five-by-five grid using the same five 32 by 32 inputs across rows."
    with tempfile.TemporaryDirectory(prefix="ml2-figure-05-") as temporary:
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
        raise RuntimeError("Figure 5 reviewed-style replacement is restricted to the canonical figure directory")
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
        raise RuntimeError("Figure 5 output conflict; no artifact was written: " + ", ".join(conflicts))
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
    parser.add_argument("--dataset", choices=("celeba", "cifar10"), default="celeba")
    parser.add_argument("--split-point", choices=("early", "current"), default="current")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument(
        "--replace-reviewed-style",
        action="store_true",
        help="Atomically replace canonical Figure 5 after checkpointed scientific inputs match.",
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
            f"Figure 5 is blocked: {consolidated.get('complete_runs', 0)}/{consolidated.get('expected_runs', 80)} primary jobs complete"
        )

    results: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        matches = [
            record["result"]
            for record in consolidated["records"]
            if record["result"]["dataset"] == args.dataset
            and record["result"]["split_point"] == args.split_point
            and record["result"]["method"] == method
            and int(record["result"]["seed"]) == args.seed
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one primary record for {method}, found {len(matches)}")
        results[method] = matches[0]

    raw_rows = _sealed_raw_rows()
    original_reference: torch.Tensor | None = None
    reconstructions: dict[str, np.ndarray] = {}
    selected_attackers: dict[str, str] = {}
    draw_seeds: dict[str, int] = {}
    for method in METHODS:
        result = results[method]
        architecture = result["validation_selected_attacker"]
        sample_bundle = _artifact_path(result["attackers"][architecture]["sample_bundle_path"])
        _require_sealed_raw_artifact(sample_bundle, raw_rows)
        bundle = torch.load(sample_bundle, map_location="cpu", weights_only=True)
        original = bundle["original"]
        reconstruction = bundle["reconstruction"]
        if tuple(original.shape) != (5, 3, 32, 32) or tuple(reconstruction.shape) != (5, 3, 32, 32):
            raise RuntimeError(f"Figure 5 bundle for {method} has an incompatible tensor shape")
        if not torch.isfinite(original).all() or not torch.isfinite(reconstruction).all():
            raise RuntimeError(f"Figure 5 bundle for {method} contains non-finite values")
        if original_reference is None:
            original_reference = original.clone()
        elif not torch.equal(original_reference, original):
            raise RuntimeError("Figure 5 originals are not exactly matched across methods")
        reconstructions[method] = np.clip(reconstruction.numpy().transpose(0, 2, 3, 1), 0.0, 1.0)
        selected_attackers[method] = architecture
        draw_seeds[method] = int(bundle["draw_seed"])
    assert original_reference is not None
    originals = np.clip(original_reference.numpy().transpose(0, 2, 3, 1), 0.0, 1.0)

    display_methods = {"standard": "Standard", "afd": "AFD", "laplace": "Laplace", "learned": "Learned"}
    display_attackers = {"deconv_mse": "deconv MSE", "residual_lpips": "residual + LPIPS"}
    row_labels = ["Original\n(32 × 32 input)"] + [
        f"{display_methods[method]}\n({display_attackers[selected_attackers[method]]})"
        for method in METHODS
    ]
    image_rows = [originals] + [reconstructions[method] for method in METHODS]
    with plt.rc_context(BASE_STYLE):
        figure, axes = plt.subplots(5, 5, figsize=(7.5, 6.25))
        for row_index, images in enumerate(image_rows):
            for column_index in range(5):
                axes[row_index, column_index].imshow(
                    images[column_index], interpolation="nearest", rasterized=True
                )
                axes[row_index, column_index].set_xticks([])
                axes[row_index, column_index].set_yticks([])
                for spine in axes[row_index, column_index].spines.values():
                    spine.set_visible(False)
            axes[row_index, 0].set_ylabel(
                row_labels[row_index],
                rotation=0,
                ha="right",
                va="center",
                labelpad=11,
                fontsize=8,
            )
        figure.tight_layout(pad=0.35, h_pad=0.2, w_pad=0.2)
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
                "dataset": args.dataset,
                "split_point": args.split_point,
                "seed": args.seed,
                "examples": 5,
                "grid": [5, 5],
                "input_shape": [3, 32, 32],
                "raster_export_dpi": 600,
                "methods": list(METHODS),
                "attackers_selected_on_validation": selected_attackers,
                "stochastic_draw_seed_if_applicable": draw_seeds,
                "matched_originals": True,
                "output_states": states,
                "output_sha256": {name: hashlib.sha256(payload).hexdigest() for name, payload in outputs.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
