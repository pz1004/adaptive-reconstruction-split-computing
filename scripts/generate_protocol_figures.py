#!/usr/bin/env python3
"""Generate deterministic reviewed-style protocol and payload figures."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "revisions" / "2026-07-22_adaptive_reconstruction_study"
CHECKPOINT_ROOT = PROJECT_ROOT / "audit" / "figure_revision_20260822" / "pre_change"
CHECKPOINT_REVISION = (
    CHECKPOINT_ROOT / "copies" / "revisions" / "2026-07-22_adaptive_reconstruction_study"
)
CANONICAL_FIGURE_ROOT = DEFAULT_ROOT / "figures"
OKABE_ITO_REVIEWED = ("#000000", "#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00")
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
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_checkpoint_match(live: Path, preserved: Path, label: str) -> None:
    if not live.is_file() or not preserved.is_file():
        raise RuntimeError(f"Reviewed-style replacement is blocked: missing {label}")
    if live.read_bytes() != preserved.read_bytes():
        raise RuntimeError(
            f"Reviewed-style replacement is blocked: {label} differs from the pre-change checkpoint"
        )


def payload_rows() -> list[dict[str, float | int | str | bool]]:
    bandwidths = (0.5, 1, 2, 5, 10, 20, 50)
    interfaces = {"early": 6 * 16 * 16, "current": 6 * 8 * 8}
    rows: list[dict[str, float | int | str | bool]] = []
    for interface, value_count in interfaces.items():
        for precision in (32, 16, 8):
            payload_bits = value_count * precision
            for bandwidth in bandwidths:
                rows.append(
                    {
                        "interface": interface,
                        "precision_bits_per_value": precision,
                        "bandwidth_mbps": bandwidth,
                        "value_count": value_count,
                        "payload_bits": payload_bits,
                        "payload_bytes": payload_bits / 8,
                        "payload_transmission_time_ms": payload_bits / (bandwidth * 1e6) * 1e3,
                        "analytical_not_measured": True,
                    }
                )
    raw_value_count = 3 * 32 * 32
    raw_payload_bits = raw_value_count * 8
    for bandwidth in bandwidths:
        rows.append(
            {
                "interface": "raw_image",
                "precision_bits_per_value": 8,
                "bandwidth_mbps": bandwidth,
                "value_count": raw_value_count,
                "payload_bits": raw_payload_bits,
                "payload_bytes": raw_payload_bits / 8,
                "payload_transmission_time_ms": raw_payload_bits / (bandwidth * 1e6) * 1e3,
                "analytical_not_measured": True,
            }
        )
    if len(rows) != 49:
        raise RuntimeError(f"Figure 4 is blocked: expected 49 analytical rows, found {len(rows)}")
    return rows


def payload_plot_series(
    rows: list[dict[str, float | int | str | bool]],
) -> list[dict[str, str | list[float]]]:
    def values(interface: str, precision: int) -> list[float]:
        return [
            float(row["payload_transmission_time_ms"])
            for row in rows
            if row["interface"] == interface and row["precision_bits_per_value"] == precision
        ]

    early_8 = values("early", 8)
    current_32 = values("current", 32)
    if early_8 != current_32:
        raise RuntimeError("Figure 4 is blocked: the reviewed shared-curve identity no longer holds")
    series: list[dict[str, str | list[float]]] = [
        {"label": "early, 32-bit", "values": values("early", 32)},
        {"label": "early, 16-bit", "values": values("early", 16)},
        {"label": "early, 8-bit = current, 32-bit", "values": early_8},
        {"label": "current, 16-bit", "values": values("current", 16)},
        {"label": "current, 8-bit", "values": values("current", 8)},
        {"label": "raw image, 8-bit", "values": values("raw_image", 8)},
    ]
    if len(series) != 6 or any(len(row["values"]) != 7 for row in series):
        raise RuntimeError("Figure 4 is blocked: expected six displayed seven-point series")
    return series


def _require_reviewed_inputs(revision_root: Path) -> list[dict[str, float | int | str | bool]]:
    for relative in (
        Path("study_contract.json"),
        Path("tables/payload_time_sensitivity.csv"),
        Path("tables/payload_time_sensitivity.json"),
    ):
        _require_checkpoint_match(
            revision_root / relative,
            CHECKPOINT_REVISION / relative,
            relative.as_posix(),
        )
    _require_checkpoint_match(
        PROJECT_ROOT / "config" / "study_contract.json",
        CHECKPOINT_ROOT / "copies" / "config" / "study_contract.json",
        "config/study_contract.json",
    )
    rows = payload_rows()
    live_payload = json.loads(
        (revision_root / "tables" / "payload_time_sensitivity.json").read_text(encoding="utf-8")
    )
    if live_payload.get("measurement_type") != "analytical payload transmission-time estimate":
        raise RuntimeError("Figure 4 is blocked: analytical measurement boundary changed")
    if live_payload.get("rows") != rows:
        raise RuntimeError("Figure 4 is blocked: the 49 analytical payload rows changed")
    return rows


def _box(
    axis: Any,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    color: str,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02",
        linewidth=1.25,
        edgecolor="#222222",
        facecolor=color,
    )
    axis.add_patch(patch)
    axis.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=9)


def _arrow(
    axis: Any,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    dashed: bool = False,
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            linestyle="--" if dashed else "-",
            color="#333333",
        )
    )


def _render_threat_model() -> tuple[Any, str, str]:
    figure, axis = plt.subplots(figsize=(7.5, 3.5))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 6)
    axis.axis("off")
    _box(axis, 0.3, 3.7, 1.7, 1.0, "Input image\n3 × 32 × 32", "#E8E8E8")
    _box(axis, 2.5, 3.7, 1.8, 1.0, "Edge encoder", "#56B4E9")
    _box(axis, 4.9, 3.7, 2.1, 1.0, "Defended feature\nearly or current", "#F0E442")
    _box(axis, 7.7, 3.7, 1.8, 1.0, "Cloud classifier", "#009E73")
    _box(axis, 9.9, 3.7, 1.7, 1.0, "Task output", "#E8E8E8")
    for start, end in (
        ((2.0, 4.2), (2.5, 4.2)),
        ((4.3, 4.2), (4.9, 4.2)),
        ((7.0, 4.2), (7.7, 4.2)),
        ((9.5, 4.2), (9.9, 4.2)),
    ):
        _arrow(axis, start, end)

    _box(axis, 2.8, 1.3, 2.25, 1.0, "Co-trained decoder\ntraining diagnostic", "#CC79A7")
    _box(axis, 5.55, 1.3, 2.3, 1.0, "Adaptive decoders\nMSE and L₁ + LPIPS", "#E69F00")
    _box(axis, 8.35, 1.3, 2.2, 1.0, "Attribute probe\n39 CelebA labels", "#D55E00")
    _arrow(axis, (5.7, 3.7), (4.0, 2.3), dashed=True)
    _arrow(axis, (6.0, 3.7), (6.7, 2.3), dashed=True)
    _arrow(axis, (6.35, 3.7), (9.4, 2.3), dashed=True)
    axis.text(
        3.45,
        0.5,
        "Training: GRL updates the encoder",
        ha="center",
        va="center",
        fontsize=8,
        color="#333333",
    )
    axis.text(
        8.55,
        0.5,
        "Evaluation: encoder frozen; attackers adapt",
        ha="center",
        va="center",
        fontsize=8,
        color="#333333",
    )
    figure.tight_layout(pad=0.35)
    return (
        figure,
        "Split-computing threat model and evaluated paths",
        "Solid arrows show inference; dashed arrows show training-diagnostic and frozen-encoder evaluation paths.",
    )


def _render_payload(
    rows: list[dict[str, float | int | str | bool]],
) -> tuple[Any, str, str]:
    bandwidths = np.asarray((0.5, 1, 2, 5, 10, 20, 50), dtype=float)
    figure, axis = plt.subplots(figsize=(7.5, 4.35))
    markers = ("s", "o", "D", "^", "v", "P")
    linestyles = ("-", "--", "-.", ":", "-", "--")
    for index, item in enumerate(payload_plot_series(rows)):
        axis.plot(
            bandwidths,
            np.asarray(item["values"], dtype=float),
            color=OKABE_ITO_REVIEWED[index],
            marker=markers[index],
            linestyle=linestyles[index],
            linewidth=1.45,
            markersize=4.8,
            markeredgewidth=0.55,
            markeredgecolor="#222222",
            markerfacecolor="white" if index in (1, 5) else OKABE_ITO_REVIEWED[index],
            label=str(item["label"]),
            zorder=2 + index,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xticks(bandwidths)
    axis.set_xticklabels([str(value).rstrip("0").rstrip(".") for value in bandwidths])
    axis.set_xlabel("Link bandwidth (Mbit/s)")
    axis.set_ylabel("Payload transmission-time estimate (ms)")
    axis.grid(axis="both", which="major", color="#D0D0D0", linestyle="--", linewidth=0.6)
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False, fontsize=8)
    figure.tight_layout(pad=0.55)
    return (
        figure,
        "Analytical payload transmission-time sensitivity",
        "Six displayed series from 49 unchanged analytical rows; early 8-bit and current 32-bit coincide.",
    )


def _figure_pair_bytes(figure: Any, stem: str, title: str, subject: str) -> dict[str, bytes]:
    with tempfile.TemporaryDirectory(prefix=f"ml2-{stem}-") as temporary:
        root = Path(temporary)
        pdf = root / f"{stem}.pdf"
        svg = root / f"{stem}.svg"
        with plt.rc_context({**BASE_STYLE, "svg.hashsalt": f"ml2-{stem}-reviewed-v1"}):
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
        raise RuntimeError("Reviewed-style replacement is restricted to the canonical figure directory")
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
        raise RuntimeError("Figure output conflict; no artifact was written: " + ", ".join(conflicts))
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
    parser.add_argument("--output-dir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument(
        "--replace-reviewed-style",
        action="store_true",
        help="Atomically replace canonical Figure 1/4 pairs after checkpointed inputs match.",
    )
    args = parser.parse_args()
    revision_root = args.revision_root.resolve()
    rows = _require_reviewed_inputs(revision_root)
    output_dir = (args.output_dir or revision_root / "figures").resolve()
    outputs: dict[str, bytes] = {}
    with plt.rc_context(BASE_STYLE):
        threat, threat_title, threat_subject = _render_threat_model()
        try:
            outputs.update(
                _figure_pair_bytes(
                    threat,
                    "figure-01-threat-model",
                    threat_title,
                    threat_subject,
                )
            )
        finally:
            plt.close(threat)
        payload, payload_title, payload_subject = _render_payload(rows)
        try:
            outputs.update(
                _figure_pair_bytes(
                    payload,
                    "figure-04-payload-time-sensitivity",
                    payload_title,
                    payload_subject,
                )
            )
        finally:
            plt.close(payload)
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
                "input_checkpoint": str(CHECKPOINT_ROOT),
                "payload_row_count": len(rows),
                "displayed_payload_series": len(payload_plot_series(rows)),
                "output_states": states,
                "output_sha256": {name: hashlib.sha256(payload).hexdigest() for name, payload in outputs.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
