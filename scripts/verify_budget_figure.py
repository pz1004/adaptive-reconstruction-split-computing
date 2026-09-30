#!/usr/bin/env python3
"""Verify Figure 3 data identity, labels, vector structure, and grayscale encodings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "revisions" / "2026-07-22_adaptive_reconstruction_study"
DEFAULT_OUTPUT = PROJECT_ROOT / "audit" / "budget_result_audit_v2_20260816" / "figure_3_verification.json"
BOUNDARY = "Single predeclared seed 42; descriptive sensitivity; no error bars."


def verify(revision_root: Path) -> dict[str, Any]:
    revision_root = revision_root.resolve()
    figure_root = revision_root / "figures"
    data_path = figure_root / "figure-03-attacker-budget-sensitivity.data.json"
    pdf_path = figure_root / "figure-03-attacker-budget-sensitivity.pdf"
    svg_path = figure_root / "figure-03-attacker-budget-sensitivity.svg"
    errors: list[str] = []
    try:
        data_bytes = data_path.read_bytes()
        data = json.loads(data_bytes)
    except (OSError, ValueError) as error:
        return {"schema_version": 1, "passed": False, "errors": [f"data payload: {error}"]}
    digest = hashlib.sha256(data_bytes).hexdigest()
    if data.get("seed") != 42 or data.get("boundary_text") != BOUNDARY:
        errors.append("data payload does not state the fixed seed-42 descriptive boundary")
    for field in (
        "fixed_conditions_are_replicates",
        "attackers_are_replicates",
        "images_are_replicates",
        "auxiliary_examples_are_replicates",
        "error_bars",
        "inferential_annotations",
    ):
        if data.get(field) is not False:
            errors.append(f"data payload violates the non-inferential boundary: {field}")
    panels = data.get("sensitivity_panels", [])
    if len(panels) != 2 or any(len(panel.get("series", [])) != 4 for panel in panels):
        errors.append("data payload is missing one of the two four-series sensitivity panels")
    style = data.get("style", {})
    if style.get("split_line_styles", {}).get("early") == style.get("split_line_styles", {}).get("current"):
        errors.append("early/current series are not distinguished by line style")
    if style.get("split_markers", {}).get("early") == style.get("split_markers", {}).get("current"):
        errors.append("early/current series are not distinguished by marker")
    if style.get("method_colors", {}).get("standard") == style.get("method_colors", {}).get("afd"):
        errors.append("standard/AFD series are not distinguished by color")

    svg_text = ""
    try:
        ET.parse(svg_path)
        svg_text = svg_path.read_text(encoding="utf-8")
    except (OSError, ET.ParseError) as error:
        errors.append(f"SVG is missing or malformed: {error}")
    for required in (digest, BOUNDARY, "CelebA", "CIFAR-10", "Worst-case adaptive SSIM"):
        if required not in svg_text:
            errors.append(f"SVG is missing required data/label text: {required}")

    pdf_text = ""
    pdf_info = ""
    if not pdf_path.is_file():
        errors.append("PDF is missing")
    else:
        pdf_bytes = pdf_path.read_bytes()
        if digest.encode("ascii") not in pdf_bytes:
            errors.append("PDF metadata does not bind the Figure 3 data digest")
        with tempfile.TemporaryDirectory(prefix="ml2-budget-pdf-check-") as temporary:
            text_path = Path(temporary) / "figure.txt"
            text_result = subprocess.run(
                ["pdftotext", str(pdf_path), str(text_path)], text=True, capture_output=True, check=False
            )
            if text_result.returncode == 0 and text_path.is_file():
                pdf_text = text_path.read_text(encoding="utf-8", errors="replace")
            else:
                errors.append(f"pdftotext failed: {text_result.stderr.strip()}")
        info_result = subprocess.run(["pdfinfo", str(pdf_path)], text=True, capture_output=True, check=False)
        if info_result.returncode == 0:
            pdf_info = info_result.stdout
            if "Pages:           1" not in pdf_info:
                errors.append("Figure 3 PDF is not exactly one page")
        else:
            errors.append(f"pdfinfo failed: {info_result.stderr.strip()}")
    for required in (BOUNDARY, "CelebA", "CIFAR-10", "Worst-case adaptive SSIM"):
        if required not in pdf_text:
            errors.append(f"PDF text is missing required label: {required}")

    return {
        "schema_version": 1,
        "passed": not errors,
        "errors": errors,
        "data_sha256": digest,
        "paths": {"data": str(data_path), "pdf": str(pdf_path), "svg": str(svg_path)},
        "checks": {
            "same_data_digest_embedded_in_pdf_svg": digest.encode("ascii") in pdf_path.read_bytes()
            if pdf_path.is_file()
            else False,
            "required_labels_present": not any("missing required" in error for error in errors),
            "one_page_vector_pdf": "Pages:           1" in pdf_info,
            "svg_parses": bool(svg_text),
            "grayscale_distinguishable_line_styles_and_markers": not any(
                "not distinguished" in error for error in errors
            ),
            "boundary_text_present": BOUNDARY in svg_text and BOUNDARY in pdf_text,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    report = verify(args.revision_root)
    payload = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    if args.output.is_file() and args.output.read_bytes() == payload:
        state = "verified_existing"
    elif args.check:
        raise SystemExit(f"Figure verification output is missing or byte-different: {args.output}")
    elif args.output.exists():
        raise SystemExit(f"Refusing to overwrite conflicting figure verification: {args.output}")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("wb", dir=args.output.parent, delete=False) as handle:
            handle.write(payload)
            temporary = Path(handle.name)
        try:
            os.link(temporary, args.output)
            state = "created"
        finally:
            temporary.unlink(missing_ok=True)
    print(json.dumps({"passed": report["passed"], "errors": report["errors"], "output_state": state}, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
