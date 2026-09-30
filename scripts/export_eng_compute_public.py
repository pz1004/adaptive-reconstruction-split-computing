#!/usr/bin/env python3
"""Stage the public-safe aggregate v1.2.0 package locally without publishing."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_RELEASE = PROJECT_ROOT / "analysis-output" / "public-release-v1.1.0"
COMPUTE_ROOT = PROJECT_ROOT / "revisions" / "2026-08-27_eng_compute_extension" / "eng_compute_v1"
PRIVATE_CANDIDATE = PROJECT_ROOT / "analysis-output" / "public-release-v1.2.0"
DEFAULT_OUTPUT = PROJECT_ROOT / "release" / "v1.2.0"
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

EXPECTED_FILES = frozenset(
    {
        "RELEASE_NOTES.md",
        "SHA256SUMS",
        "analysis/compute_analysis.json",
        "analysis/compute_claim_audit.json",
        "schemas/attacker_restart_rows.schema.json",
        "schemas/attacker_restart_summaries.schema.json",
        "schemas/benchmark_budget_attacker_rows.schema.json",
        "schemas/benchmark_budget_conditions.schema.json",
        "schemas/benchmark_budget_paired_effects.schema.json",
        "schemas/benchmark_v1_tables.schema.json",
        "schemas/eng_compute_v1.schema.json",
    }
    | set(CSV_SPECS)
)


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


def _release_notes() -> bytes:
    return b"""# Local public-safe aggregate release candidate v1.2.0

This local candidate extends the unchanged v1.1.0 benchmark aggregates with five public-safe `eng_compute_v1` aggregate files: grouped component summaries, paired AFD-minus-standard effects, interface effects, the bounded analysis contract, and the three-claim audit.

The compute extension covers 160 declared conditions on the bounded one-thread CPU and RTX 2080 Ti testbed. Technical timing blocks quantify measurement noise; model seed is the inferential unit. Intervals are descriptive Student-t 95% intervals. No p-values, multiplicity tests, equivalence claims, deployed end-to-end latency, networking, queueing, serialization, data-loading, energy, or hardware-general claims are included.

This directory is a local staging artifact, not a published GitHub release or live URL. Publication remains blocked by `V1.2.0_RELEASE_URL_REQUIRED`. No tag, release, upload, DOI, or external service action has occurred.
"""


def _build(root: Path) -> None:
    if not SOURCE_RELEASE.is_dir():
        raise RuntimeError(f"Missing v1.1.0 source release: {SOURCE_RELEASE}")
    shutil.copytree(SOURCE_RELEASE, root)
    for relative, source in COMPUTE_FILES.items():
        if not source.is_file():
            raise RuntimeError(f"Missing accepted compute aggregate: {source}")
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    (root / "schemas" / "eng_compute_v1.schema.json").write_bytes(_schema())
    (root / "RELEASE_NOTES.md").write_bytes(_release_notes())
    checksum = root / "SHA256SUMS"
    rows = []
    for path in sorted(path for path in root.rglob("*") if path.is_file() and path != checksum):
        rows.append(f"{_sha256(path)}  {path.relative_to(root).as_posix()}")
    checksum.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _payloads(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(path for path in root.rglob("*") if path.is_file())
    }


def build_private_payloads() -> dict[str, bytes]:
    """Regenerate the complete package from private canonical evidence."""
    with tempfile.TemporaryDirectory(prefix="public-release-v1.2.0-") as directory:
        temporary = Path(directory) / "release"
        _build(temporary)
        validate_public_release(temporary)
        return _payloads(temporary)


def validate_public_release(root: Path) -> dict[str, object]:
    """Validate the self-contained public v1.2.0 inventory without private inputs."""
    if not root.is_dir():
        raise RuntimeError(f"Missing public v1.2.0 release: {root}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual != EXPECTED_FILES:
        raise RuntimeError(
            "v1.2.0 inventory differs: "
            f"missing={sorted(EXPECTED_FILES - actual)}, "
            f"unexpected={sorted(actual - EXPECTED_FILES)}"
        )

    checksum_path = root / "SHA256SUMS"
    checksum_rows: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as error:
            raise RuntimeError(f"Malformed v1.2.0 checksum row: {line!r}") from error
        if relative in checksum_rows or len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise RuntimeError(f"Invalid or duplicate v1.2.0 checksum row: {line!r}")
        checksum_rows[relative] = digest
    expected_checksum_paths = EXPECTED_FILES - {"SHA256SUMS"}
    if set(checksum_rows) != expected_checksum_paths:
        raise RuntimeError("v1.2.0 checksum inventory differs from the fixed 18-file policy")
    for relative, digest in checksum_rows.items():
        if _sha256(root / relative) != digest:
            raise RuntimeError(f"v1.2.0 checksum mismatch: {relative}")

    result_rows = 0
    for relative, (expected_rows, expected_fields) in CSV_SPECS.items():
        with (root / relative).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        if tuple(reader.fieldnames or ()) != expected_fields:
            raise RuntimeError(f"Unexpected v1.2.0 CSV schema: {relative}")
        if len(rows) != expected_rows or any(None in row.values() for row in rows):
            raise RuntimeError(f"Unexpected v1.2.0 CSV row count/content: {relative}")
        result_rows += len(rows)

    schema_pairs = {
        "results/attacker_restart_rows.csv": "schemas/attacker_restart_rows.schema.json",
        "results/attacker_restart_summaries.csv": "schemas/attacker_restart_summaries.schema.json",
        "results/benchmark_budget_attacker_rows.csv": "schemas/benchmark_budget_attacker_rows.schema.json",
        "results/benchmark_budget_conditions.csv": "schemas/benchmark_budget_conditions.schema.json",
        "results/benchmark_budget_paired_effects.csv": "schemas/benchmark_budget_paired_effects.schema.json",
    }
    for csv_relative, schema_relative in schema_pairs.items():
        schema = json.loads((root / schema_relative).read_text(encoding="utf-8"))
        columns = tuple(column["name"] for column in schema.get("columns", []))
        if (
            schema.get("schema_version") != 2
            or schema.get("file") != csv_relative
            or schema.get("rows") != CSV_SPECS[csv_relative][0]
            or columns != CSV_SPECS[csv_relative][1]
        ):
            raise RuntimeError(f"v1.2.0 schema metadata differs: {schema_relative}")
    benchmark_schema = json.loads(
        (root / "schemas/benchmark_v1_tables.schema.json").read_text(encoding="utf-8")
    )
    if benchmark_schema.get("schema_version") != 2 or set(
        benchmark_schema.get("tables", {})
    ) != {Path(path).name for path in schema_pairs}:
        raise RuntimeError("v1.2.0 benchmark table schema differs")
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
        raise RuntimeError("v1.2.0 compute schema boundary differs")

    analysis = json.loads((root / "analysis/compute_analysis.json").read_text(encoding="utf-8"))
    claims = json.loads((root / "analysis/compute_claim_audit.json").read_text(encoding="utf-8"))
    if (
        analysis.get("protocol_id") != "eng_compute_v1"
        or analysis.get("condition_count") != 160
        or analysis.get("quarantine_count") != 0
        or analysis.get("inferential_unit") != "model_seed"
        or analysis.get("technical_blocks_are_inferential_units") is not False
    ):
        raise RuntimeError("v1.2.0 compute analysis boundary differs")
    if claims.get("claim_count") != 3 or [
        (claim.get("claim_id"), claim.get("status")) for claim in claims.get("claims", [])
    ] != [("E01", "supported"), ("E02", "supported"), ("E03", "supported")]:
        raise RuntimeError("v1.2.0 compute claim audit differs")
    notes = (root / "RELEASE_NOTES.md").read_text(encoding="utf-8")
    for fragment in (
        "local staging artifact",
        "not a published GitHub release or live URL",
        "V1.2.0_RELEASE_URL_REQUIRED",
        "No tag, release, upload, DOI, or external service action has occurred.",
    ):
        if fragment not in notes:
            raise RuntimeError(f"v1.2.0 release notes missing boundary: {fragment}")
    return {
        "passed": True,
        "file_count": len(actual),
        "checksum_entries": len(checksum_rows),
        "csv_table_count": len(CSV_SPECS),
        "aggregate_rows": result_rows,
        "status": "staged_not_published",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--public-only",
        action="store_true",
        help="Validate the staged release without private canonical evidence",
    )
    args = parser.parse_args()
    if args.public_only and not args.check:
        parser.error("--public-only requires --check")
    if args.check and args.public_only:
        release_report = validate_public_release(args.output)
        private_regeneration = False
        private_candidate_match = False
        state = "verified_existing_public_only"
    else:
        expected = build_private_payloads()
        if args.check:
            if _payloads(PRIVATE_CANDIDATE) != expected:
                raise RuntimeError(
                    f"Retained private v1.2.0 candidate differs: {PRIVATE_CANDIDATE}"
                )
            if not args.output.is_dir() or _payloads(args.output) != expected:
                raise RuntimeError(f"Existing v1.2.0 package differs: {args.output}")
            state = "verified_existing_private_regeneration"
        else:
            if args.output.exists():
                raise RuntimeError(f"Refusing to overwrite local release: {args.output}")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="public-release-v1.2.0-write-") as directory:
                temporary = Path(directory) / "release"
                _build(temporary)
                shutil.copytree(temporary, args.output)
            state = "written"
        release_report = validate_public_release(args.output)
        private_regeneration = True
        private_candidate_match = True
    print(
        json.dumps(
            {
                **release_report,
                "state": state,
                "output": str(args.output),
                "private_source_regeneration_verified": private_regeneration,
                "retained_private_candidate_match": private_candidate_match,
                "release_url_blocker": "V1.2.0_RELEASE_URL_REQUIRED",
                "external_action": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
