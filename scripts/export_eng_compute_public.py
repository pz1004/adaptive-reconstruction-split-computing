#!/usr/bin/env python3
"""Verify or export the consolidated public-safe v1.3.0 evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_RELEASE = PROJECT_ROOT / "analysis-output" / "public-release-v1.1.0"
COMPUTE_ROOT = PROJECT_ROOT / "revisions" / "2026-08-27_eng_compute_extension" / "eng_compute_v1"
PRIVATE_CANDIDATE = PROJECT_ROOT / "analysis-output" / "public-release-v1.2.0"
DEFAULT_OUTPUT = PROJECT_ROOT / "release" / "v1.3.0"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_public_repository import V13_CHECKSUMS  # noqa: E402
COMPUTE_FILES = {
    "analysis/compute_analysis.json": COMPUTE_ROOT / "analysis" / "compute_analysis.json",
    "analysis/compute_claim_audit.json": COMPUTE_ROOT / "analysis" / "compute_claim_audit.json",
    "results/compute_group_summary.csv": COMPUTE_ROOT / "analysis" / "compute_group_summary.csv",
    "results/compute_paired_effects.csv": COMPUTE_ROOT / "analysis" / "compute_paired_effects.csv",
    "results/compute_interface_effects.csv": COMPUTE_ROOT / "analysis" / "compute_interface_effects.csv",
}

CSV_SPECS = {
    "results/attacker_restart_rows.csv": (
        96,
        (
            "run_id", "dataset", "interface", "method", "architecture", "model_seed",
            "attacker_training_seed", "mse", "psnr", "ssim", "lpips", "origin",
        ),
    ),
    "results/attacker_restart_summaries.csv": (
        32,
        (
            "dataset", "interface", "method", "architecture", "model_seed",
            "attacker_training_seeds", "ssim_values", "median_ssim", "minimum_ssim",
            "maximum_ssim", "range_ssim",
        ),
    ),
    "results/benchmark_budget_attacker_rows.csv": (
        240,
        (
            "run_id", "dataset", "interface", "method", "model_seed",
            "attacker_training_seed", "auxiliary_fraction", "auxiliary_examples",
            "architecture", "mse", "psnr", "ssim", "lpips", "origin",
        ),
    ),
    "results/benchmark_budget_conditions.csv": (
        120,
        (
            "run_id", "dataset", "interface", "method", "model_seed",
            "attacker_training_seed", "auxiliary_fraction", "auxiliary_examples",
            "strongest_attacker", "worst_case_ssim", "origin",
        ),
    ),
    "results/benchmark_budget_paired_effects.csv": (
        12,
        (
            "dataset", "interface", "auxiliary_fraction", "contrast", "n_model_seeds",
            "model_seeds", "values", "mean", "sample_sd", "ci95_lower", "ci95_upper",
            "ci95_method",
        ),
    ),
    "results/compute_group_summary.csv": (
        32,
        (
            "dataset", "interface", "method", "batch_size",
            "cpu_encoder_plus_defense_latency_seconds__n",
            "cpu_encoder_plus_defense_latency_seconds__mean",
            "cpu_encoder_plus_defense_latency_seconds__std",
            "cpu_encoder_plus_defense_latency_seconds__ci95_low",
            "cpu_encoder_plus_defense_latency_seconds__ci95_high",
            "cpu_encoder_plus_defense_throughput_per_second__n",
            "cpu_encoder_plus_defense_throughput_per_second__mean",
            "cpu_encoder_plus_defense_throughput_per_second__std",
            "cpu_encoder_plus_defense_throughput_per_second__ci95_low",
            "cpu_encoder_plus_defense_throughput_per_second__ci95_high",
            "gpu_classifier_latency_seconds__n", "gpu_classifier_latency_seconds__mean",
            "gpu_classifier_latency_seconds__std", "gpu_classifier_latency_seconds__ci95_low",
            "gpu_classifier_latency_seconds__ci95_high",
            "gpu_classifier_throughput_per_second__n",
            "gpu_classifier_throughput_per_second__mean",
            "gpu_classifier_throughput_per_second__std",
            "gpu_classifier_throughput_per_second__ci95_low",
            "gpu_classifier_throughput_per_second__ci95_high",
            "summed_component_time_seconds__n", "summed_component_time_seconds__mean",
            "summed_component_time_seconds__std", "summed_component_time_seconds__ci95_low",
            "summed_component_time_seconds__ci95_high", "isolated_cpu_rss_delta_bytes__n",
            "isolated_cpu_rss_delta_bytes__mean", "isolated_cpu_rss_delta_bytes__std",
            "isolated_cpu_rss_delta_bytes__ci95_low",
            "isolated_cpu_rss_delta_bytes__ci95_high", "cuda_peak_allocation_bytes__n",
            "cuda_peak_allocation_bytes__mean", "cuda_peak_allocation_bytes__std",
            "cuda_peak_allocation_bytes__ci95_low", "cuda_peak_allocation_bytes__ci95_high",
            "static_total_bytes__n", "static_total_bytes__mean", "static_total_bytes__std",
            "static_total_bytes__ci95_low", "static_total_bytes__ci95_high",
            "static_defense_bytes__n", "static_defense_bytes__mean",
            "static_defense_bytes__std", "static_defense_bytes__ci95_low",
            "static_defense_bytes__ci95_high", "defense_only_latency_seconds__n",
            "defense_only_latency_seconds__mean", "defense_only_latency_seconds__std",
            "defense_only_latency_seconds__ci95_low",
            "defense_only_latency_seconds__ci95_high",
            "defense_only_throughput_per_second__n",
            "defense_only_throughput_per_second__mean",
            "defense_only_throughput_per_second__std",
            "defense_only_throughput_per_second__ci95_low",
            "defense_only_throughput_per_second__ci95_high",
        ),
    ),
    "results/compute_interface_effects.csv": (
        16,
        (
            "dataset", "method", "batch_size", "cpu_n", "cpu_mean", "cpu_std",
            "cpu_ci95_low", "cpu_ci95_high", "cpu_seed_differences_seconds", "gpu_n",
            "gpu_mean", "gpu_std", "gpu_ci95_low", "gpu_ci95_high",
            "gpu_seed_differences_seconds",
        ),
    ),
    "results/compute_paired_effects.csv": (
        8,
        (
            "dataset", "interface", "batch_size", "cpu_n", "cpu_mean", "cpu_std",
            "cpu_ci95_low", "cpu_ci95_high", "cpu_seed_differences_seconds", "gpu_n",
            "gpu_mean", "gpu_std", "gpu_ci95_low", "gpu_ci95_high",
            "gpu_seed_differences_seconds", "summed_n", "summed_mean", "summed_std",
            "summed_ci95_low", "summed_ci95_high", "summed_seed_differences_seconds",
        ),
    ),
}

# Historical schema file identifiers and private-candidate paths are immutable.
LEGACY_LOCATION_MAP = {'analysis/compute_analysis.json': 'analysis/compute_analysis.json',
 'analysis/compute_claim_audit.json': 'analysis/compute_claim_audit.json',
 'results/attacker_restart_rows.csv': 'inputs/attacker_restart_rows.csv',
 'results/attacker_restart_summaries.csv': 'expected/attacker_restart_summaries.csv',
 'results/benchmark_budget_attacker_rows.csv': 'evidence/benchmark_budget_attacker_rows.csv',
 'results/benchmark_budget_conditions.csv': 'inputs/benchmark_budget_conditions.csv',
 'results/benchmark_budget_paired_effects.csv': 'expected/benchmark_budget_paired_effects.csv',
 'results/compute_group_summary.csv': 'expected/compute_group_summary.csv',
 'results/compute_interface_effects.csv': 'expected/compute_interface_effects.csv',
 'results/compute_paired_effects.csv': 'expected/compute_paired_effects.csv',
 'schemas/attacker_restart_rows.schema.json': 'schemas/attacker_restart_rows.schema.json',
 'schemas/attacker_restart_summaries.schema.json': 'schemas/attacker_restart_summaries.schema.json',
 'schemas/benchmark_budget_attacker_rows.schema.json': 'schemas/benchmark_budget_attacker_rows.schema.json',
 'schemas/benchmark_budget_conditions.schema.json': 'schemas/benchmark_budget_conditions.schema.json',
 'schemas/benchmark_budget_paired_effects.schema.json': 'schemas/benchmark_budget_paired_effects.schema.json',
 'schemas/benchmark_v1_tables.schema.json': 'schemas/benchmark_v1_tables.schema.json',
 'schemas/eng_compute_v1.schema.json': 'schemas/eng_compute_v1.schema.json'}
CSV_SPECS = {LEGACY_LOCATION_MAP[path]: spec for path, spec in CSV_SPECS.items()}

EXPECTED_CHECKSUMS = {
    path.removeprefix("release/v1.3.0/"): digest for path, digest in V13_CHECKSUMS.items()
}
EXPECTED_FILES = frozenset(EXPECTED_CHECKSUMS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _schema() -> bytes:
    payload = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "eng_compute_v1 public aggregate package",
        "schema_version": 1,
        "protocol_id": "eng_compute_v1",
        "scope": "aggregate numeric component-compute summaries only",
        "inferential_unit": "model_seed",
        "technical_blocks_are_inferential_units": False,
        "uncertainty": "descriptive two-sided Student-t 95% intervals",
        "p_values": False,
        "multiplicity_tests": False,
        "equivalence_claims": False,
        "measurement_exclusions": [
            "dataset_loading",
            "serialization",
            "host_to_device_transfer",
            "networking",
            "queueing",
            "energy",
        ],
        "excluded_from_release": [
            "raw technical-block timing records",
            "model weights and checkpoints",
            "raw or reconstructed images",
            "dataset-derived per-example records",
            "private run artifacts",
            "manuscript sources and PDFs",
        ],
    }
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")



def _release_files(root: Path) -> set[str]:
    """Inventory delivery files, excluding only generated Python bytecode."""
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"Release must be a regular directory: {root}")
    files = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"Release contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise RuntimeError(f"Release contains a non-regular file: {path}")
        if path.parent.name == "__pycache__" and path.suffix == ".pyc":
            continue
        files.add(path.relative_to(root).as_posix())
    return files


def build_private_payloads() -> dict[str, bytes]:
    """Regenerate 17 scientific/schema files at their consolidated locations."""
    payloads = {}
    for historical, current in LEGACY_LOCATION_MAP.items():
        if historical == "schemas/eng_compute_v1.schema.json":
            payload = _schema()
        else:
            source = COMPUTE_FILES.get(historical, SOURCE_RELEASE / historical)
            payload = source.read_bytes()
        payloads[current] = payload
    retained = mapped_candidate_payloads()
    if payloads != retained:
        raise RuntimeError("Retained private evidence differs from canonical regeneration")
    return payloads


def build_public_payloads(source: Path) -> dict[str, bytes]:
    """Combine canonical evidence with pinned reproduction files before writing."""
    scientific = build_private_payloads()
    if set(scientific) != set(LEGACY_LOCATION_MAP.values()):
        raise RuntimeError("Private evidence inventory differs from the mapped file policy")
    actual = _release_files(source)
    required = EXPECTED_FILES - set(scientific)
    if actual - EXPECTED_FILES or required - actual:
        raise RuntimeError(
            "v1.3.0 inventory differs: "
            f"missing={sorted(required - actual)}, unexpected={sorted(actual - EXPECTED_FILES)}"
        )
    payloads = {}
    for relative, digest in EXPECTED_CHECKSUMS.items():
        path = source / relative
        payload = scientific[relative] if relative in scientific else path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != digest:
            raise RuntimeError(f"v1.3.0 payload checksum differs: {relative}")
        if relative in actual and path.read_bytes() != payload:
            raise RuntimeError(f"Scientific bytes differ from private regeneration: {relative}")
        payloads[relative] = payload
    return payloads


def mapped_candidate_payloads(candidate: Path = PRIVATE_CANDIDATE) -> dict[str, bytes]:
    """Read retained evidence without requiring obsolete public release paths."""
    return {current: (candidate / historical).read_bytes()
            for historical, current in LEGACY_LOCATION_MAP.items()}


def verify_private_candidate(root: Path, candidate: Path = PRIVATE_CANDIDATE) -> dict[str, object]:
    report = validate_public_release(root)
    for current, payload in mapped_candidate_payloads(candidate).items():
        if (root / current).read_bytes() != payload:
            raise RuntimeError(f"Consolidated evidence differs from retained private candidate: {current}")
    return {**report, "candidate_byte_identity": True,
            "mapped_scientific_files_verified": len(LEGACY_LOCATION_MAP)}


def preflight_payloads(
    root: Path,
    payloads: dict[str, bytes],
    *,
    replaceable: frozenset[str] = frozenset(),
) -> None:
    """Reject path escapes, symlinks and conflicts across the entire write set."""
    for relative, payload in payloads.items():
        name = Path(relative)
        if name.is_absolute() or ".." in name.parts or not name.parts:
            raise RuntimeError(f"Invalid export path: {relative}")
        path = root / name
        for directory in (root, *(root / parent for parent in reversed(name.parents[:-1]))):
            if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
                raise RuntimeError(f"Export parent must be a regular directory, not a symlink: {directory}")
        if path.is_symlink() or (path.exists() and (
            not path.is_file() or (path.read_bytes() != payload and relative not in replaceable)
        )):
            raise RuntimeError(f"Refusing to overwrite byte-different public output: {path}")


def export_payloads(root: Path, payloads: dict[str, bytes]) -> None:
    """Preflight every conflict before creating files; preserve unrelated files."""
    preflight_payloads(root, payloads)
    for relative, payload in payloads.items():
        path = root / relative
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(payload)


def validate_public_release(root: Path) -> dict[str, object]:
    """Validate the self-contained public v1.3.0 inventory without private inputs."""
    actual = _release_files(root)
    if actual != EXPECTED_FILES:
        raise RuntimeError(
            "v1.3.0 inventory differs: "
            f"missing={sorted(EXPECTED_FILES - actual)}, "
            f"unexpected={sorted(actual - EXPECTED_FILES)}"
        )

    checksum_path = root / "SHA256SUMS"
    checksum_rows: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as error:
            raise RuntimeError(f"Malformed v1.3.0 checksum row: {line!r}") from error
        if relative in checksum_rows or len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise RuntimeError(f"Invalid or duplicate v1.3.0 checksum row: {line!r}")
        checksum_rows[relative] = digest
    expected_checksum_paths = EXPECTED_FILES - {"SHA256SUMS"}
    if set(checksum_rows) != expected_checksum_paths:
        raise RuntimeError("v1.3.0 checksum inventory differs from the fixed 39-file policy")
    for relative, digest in checksum_rows.items():
        if _sha256(root / relative) != digest:
            raise RuntimeError(f"v1.3.0 checksum mismatch: {relative}")

    for relative, digest in EXPECTED_CHECKSUMS.items():
        if (root / relative).is_symlink() or _sha256(root / relative) != digest:
            raise RuntimeError(f"v1.3.0 payload checksum differs: {relative}")

    result_rows = 0
    for relative, (expected_rows, expected_fields) in CSV_SPECS.items():
        with (root / relative).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        if tuple(reader.fieldnames or ()) != expected_fields:
            raise RuntimeError(f"Unexpected v1.3.0 CSV schema: {relative}")
        if len(rows) != expected_rows or any(None in row.values() for row in rows):
            raise RuntimeError(f"Unexpected v1.3.0 CSV row count/content: {relative}")
        result_rows += len(rows)

    schema_pairs = {
        "results/attacker_restart_rows.csv": "schemas/attacker_restart_rows.schema.json",
        "results/attacker_restart_summaries.csv": "schemas/attacker_restart_summaries.schema.json",
        "results/benchmark_budget_attacker_rows.csv": "schemas/benchmark_budget_attacker_rows.schema.json",
        "results/benchmark_budget_conditions.csv": "schemas/benchmark_budget_conditions.schema.json",
        "results/benchmark_budget_paired_effects.csv": "schemas/benchmark_budget_paired_effects.schema.json",
    }
    for historical, schema_relative in schema_pairs.items():
        csv_relative = LEGACY_LOCATION_MAP[historical]
        schema = json.loads((root / schema_relative).read_text(encoding="utf-8"))
        columns = tuple(column["name"] for column in schema.get("columns", []))
        if (
            schema.get("schema_version") != 2
            or schema.get("file") != historical
            or schema.get("rows") != CSV_SPECS[csv_relative][0]
            or columns != CSV_SPECS[csv_relative][1]
        ):
            raise RuntimeError(f"v1.3.0 schema metadata differs: {schema_relative}")
    benchmark_schema = json.loads(
        (root / "schemas/benchmark_v1_tables.schema.json").read_text(encoding="utf-8")
    )
    if benchmark_schema.get("schema_version") != 2 or set(
        benchmark_schema.get("tables", {})
    ) != {Path(path).name for path in schema_pairs}:
        raise RuntimeError("v1.3.0 benchmark table schema differs")
    compute_schema = json.loads(
        (root / "schemas/eng_compute_v1.schema.json").read_text(encoding="utf-8")
    )
    if (
        compute_schema.get("schema_version") != 1
        or compute_schema.get("protocol_id") != "eng_compute_v1"
        or compute_schema.get("inferential_unit") != "model_seed"
        or compute_schema.get("technical_blocks_are_inferential_units") is not False
        or compute_schema.get("p_values") is not False
        or compute_schema.get("equivalence_claims") is not False
    ):
        raise RuntimeError("v1.3.0 compute schema boundary differs")

    analysis = json.loads((root / "analysis/compute_analysis.json").read_text(encoding="utf-8"))
    claims = json.loads((root / "analysis/compute_claim_audit.json").read_text(encoding="utf-8"))
    if (
        analysis.get("protocol_id") != "eng_compute_v1"
        or analysis.get("condition_count") != 160
        or analysis.get("quarantine_count") != 0
        or analysis.get("inferential_unit") != "model_seed"
        or analysis.get("technical_blocks_are_inferential_units") is not False
    ):
        raise RuntimeError("v1.3.0 compute analysis boundary differs")
    if claims.get("claim_count") != 3 or [
        (claim.get("claim_id"), claim.get("status")) for claim in claims.get("claims", [])
    ] != [("E01", "supported"), ("E02", "supported"), ("E03", "supported")]:
        raise RuntimeError("v1.3.0 compute claim audit differs")
    return {
        "passed": True,
        "file_count": len(actual),
        "checksum_entries": len(checksum_rows),
        "csv_table_count": len(CSV_SPECS),
        "aggregate_rows": result_rows,
        "status": "repository_files",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--public-only", action="store_true",
                        help="Validate the consolidated release without private evidence")
    args = parser.parse_args()
    if args.public_only and not args.check:
        parser.error("--public-only requires --check")
    if args.public_only:
        report = validate_public_release(args.output)
        state = "verified_existing_public_only"
    else:
        expected = build_public_payloads(DEFAULT_OUTPUT if not args.check else args.output)
        if not args.check:
            export_payloads(args.output, expected)
        report = verify_private_candidate(args.output)
        state = "verified_existing_private_regeneration" if args.check else "written"
    print(json.dumps({**report, "state": state, "output": str(args.output),
                      "private_source_regeneration_verified": not args.public_only,
                      "retained_private_candidate_verified": not args.public_only,
                      "external_action": False}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
