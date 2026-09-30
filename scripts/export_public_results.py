#!/usr/bin/env python3
"""Export and independently verify the public-safe numeric disclosure bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
import tempfile
from itertools import product
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.export_eng_compute_public import (
    CSV_SPECS as RELEASE_CSV_SPECS,
    EXPECTED_FILES as EXPECTED_RELEASE_FILES,
    build_private_payloads as build_private_release_payloads,
    validate_public_release,
)


PRIVATE_ROOT = PROJECT_ROOT / "revisions" / "2026-07-22_adaptive_reconstruction_study"
PROTOCOL_ROOT = PROJECT_ROOT / "results" / "protocol"
MANIFEST_PATH = PROTOCOL_ROOT / "disclosure_manifest.json"

PRIVATE_SOURCES = {
    "study_contract": PRIVATE_ROOT / "study_contract.json",
    "environment": PRIVATE_ROOT / "environment.json",
    "frozen_selection_lock": PRIVATE_ROOT / "selection" / "frozen_operating_points.lock.json",
    "primary_seed_results": PROJECT_ROOT / "audit" / "primary_result_audit_20260808" / "primary_seed_results.csv",
    "semantic_seed_results": PROJECT_ROOT / "audit" / "semantic_result_audit_20260809" / "semantic_seed_results.csv",
    "semantic_attribute_results": PROJECT_ROOT / "audit" / "semantic_result_audit_20260809" / "semantic_attribute_results.csv",
    "budget_result_audit": PROJECT_ROOT / "audit" / "budget_result_audit_v2_20260816" / "budget_result_audit.json",
    "implementation_protocol": PROJECT_ROOT / "config" / "implementation_protocol_v2.json",
}

PRIMARY_FIELDS = (
    "run_id", "dataset", "split_point", "seed", "method", "selected_defense_value",
    "accuracy_percentage_points", "utility_loss_vs_standard_percentage_points",
    "test_utility_loss_exceeds_validation_limit", "worst_case_ssim", "strongest_attacker",
    "validation_selected_attacker", "record_passed",
)
SEMANTIC_SEED_FIELDS = (
    "run_id", "dataset", "split_point", "seed", "method", "selected_defense_value",
    "macro_auroc", "macro_balanced_accuracy", "attribute_auroc_sample_sd",
    "attribute_balanced_accuracy_sample_sd", "semantic_draw_count", "semantic_draw_seeds",
    "record_passed",
)
SEMANTIC_ATTRIBUTE_SOURCE_FIELDS = (
    "run_id", "dataset", "split_point", "seed", "method", "attribute_index",
    "attribute_name", "auroc", "balanced_accuracy",
)
SEMANTIC_ATTRIBUTE_FIELDS = tuple(
    field for field in SEMANTIC_ATTRIBUTE_SOURCE_FIELDS if field != "attribute_name"
)
BUDGET_CELL_FIELDS = (
    "run_id", "dataset", "split_point", "seed", "method", "auxiliary_fraction",
    "auxiliary_examples", "auxiliary_fitting_examples", "auxiliary_validation_examples",
    "deconv_mse_ssim", "residual_lpips_ssim", "strongest_attacker",
    "worst_case_test_ssim", "record_passed",
)
BUDGET_ATTACKER_FIELDS = (
    "run_id", "dataset", "split_point", "seed", "method", "auxiliary_fraction",
    "auxiliary_examples", "auxiliary_fitting_examples", "auxiliary_validation_examples",
    "architecture", "selected_epoch", "epochs_completed",
    "best_validation_loss", "mse", "psnr", "ssim", "lpips", "record_passed",
)

BASE_RESULT_SPECS = {
    "results/raw/primary_seed_results.csv": (80, PRIMARY_FIELDS),
    "results/raw/semantic_seed_results.csv": (40, SEMANTIC_SEED_FIELDS),
    "results/raw/semantic_attribute_results.csv": (1560, SEMANTIC_ATTRIBUTE_FIELDS),
    "results/raw/budget_cell_results.csv": (24, BUDGET_CELL_FIELDS),
    "results/raw/budget_attacker_results.csv": (48, BUDGET_ATTACKER_FIELDS),
}
PUBLIC_RELEASE_ROOT = "release/v1.3.0"
RELEASE_RESULT_SPECS = {
    f"{PUBLIC_RELEASE_ROOT}/{relative}": spec
    for relative, spec in RELEASE_CSV_SPECS.items()
}
RESULT_SPECS = {**BASE_RESULT_SPECS, **RELEASE_RESULT_SPECS}
PUBLIC_RELEASE_FILES = tuple(
    sorted(f"{PUBLIC_RELEASE_ROOT}/{relative}" for relative in EXPECTED_RELEASE_FILES)
)

PUBLIC_SOURCE_MODULES = (
    "src/analysis.py",
    "src/budget_result_audit.py",
    "src/checkpointing.py",
    "src/dataset.py",
    "src/defenses.py",
    "src/eval.py",
    "src/experiment.py",
    "src/implementation_provenance.py",
    "src/manifest.py",
    "src/metrics.py",
    "src/models.py",
    "src/pipeline.py",
    "src/posthoc_attacker_eval.py",
    "src/protocol.py",
    "src/result_audit.py",
    "src/selection_provenance.py",
    "src/semantic_analysis.py",
    "src/semantic_result_audit.py",
    "src/train.py",
    "src/watcher_state.py",
)

PUBLIC_SCRIPTS = (
    "scripts/analyze_budget_results.py",
    "scripts/analyze_revision.py",
    "scripts/analyze_semantic_results.py",
    "scripts/audit_budget_results.py",
    "scripts/audit_primary_results.py",
    "scripts/audit_public_repository.py",
    "scripts/audit_semantic_results.py",
    "scripts/export_public_results.py",
    "scripts/export_eng_compute_public.py",
    "scripts/generate_images.py",
    "scripts/generate_protocol_figures.py",
    "scripts/generate_result_figures.py",
    "scripts/plot_results.py",
    "scripts/run_ablation.sh",
    "scripts/run_eval.sh",
    "scripts/run_revision_pipeline.py",
    "scripts/run_seed_robustness.sh",
    "scripts/run_train.sh",
    "scripts/summarize_seed_robustness.py",
    "scripts/train_shredder.py",
    "scripts/verify_budget_figure.py",
)
PUBLIC_TESTS = (
    "tests/test_aggregate_reproduction.py",
    "tests/__init__.py",
    "tests/test_budget_result_audit.py",
    "tests/test_code_scripts.py",
    "tests/test_dataset_remedy.py",
    "tests/test_public_repository.py",
    "tests/test_public_protocol.py",
    "tests/test_result_audit.py",
    "tests/test_revision_protocol.py",
    "tests/test_selection_provenance.py",
    "tests/test_semantic_result_audit.py",
)
PUBLIC_WORKFLOWS = (
    ".github/workflows/public-checks.yml",
)
ROOT_METADATA = (
    ".gitattributes",
    ".gitignore",
    "CITATION.cff",
    "LICENSE",
    "LICENSE-RESULTS",
    "README.md",
    "THIRD_PARTY_DATA.md",
    "pytest.ini",
    "requirements-figures.txt",
    "requirements.txt",
    "run_public_checks.sh",
)
PROTOCOL_OUTPUTS = (
    "config/implementation_protocol_v2.json",
    "config/study_contract.json",
    "results/protocol/environment.json",
    "results/protocol/frozen_selection_lock.json",
    "results/protocol/disclosure_manifest.json",
)
CORRECTED_PROMOTION_REPLACEABLE_OUTPUTS = frozenset(
    {
        "results/raw/budget_attacker_results.csv",
        "results/raw/budget_cell_results.csv",
        "results/protocol/environment.json",
        "results/protocol/disclosure_manifest.json",
    }
)
V13_FILES = ('release/v1.3.0/DATA_DICTIONARY.md',
 'release/v1.3.0/README.md',
 'release/v1.3.0/SHA256SUMS',
 'release/v1.3.0/analysis/compute_analysis.json',
 'release/v1.3.0/analysis/compute_claim_audit.json',
 'release/v1.3.0/analysis_contract.json',
 'release/v1.3.0/budget_figure.py',
 'release/v1.3.0/compute_core.py',
 'release/v1.3.0/diagnostics/known-offset-diagnostic.csv',
 'release/v1.3.0/diagnostics/known-offset-diagnostic.json',
 'release/v1.3.0/evidence/benchmark_budget_attacker_rows.csv',
 'release/v1.3.0/expected/attacker_restart_summaries.csv',
 'release/v1.3.0/expected/benchmark_budget_paired_effects.csv',
 'release/v1.3.0/expected/budget_method_summaries.csv',
 'release/v1.3.0/expected/compute_group_summary.csv',
 'release/v1.3.0/expected/compute_interface_effects.csv',
 'release/v1.3.0/expected/compute_paired_effects.csv',
 'release/v1.3.0/expected/figure_signatures.json',
 'release/v1.3.0/expected/figure_sources.json',
 'release/v1.3.0/expected/laplace-single-release-statistics.csv',
 'release/v1.3.0/expected/supplementary-statistics.csv',
 'release/v1.3.0/inputs/attacker_restart_rows.csv',
 'release/v1.3.0/inputs/benchmark_budget_conditions.csv',
 'release/v1.3.0/inputs/compute_condition_metrics.csv',
 'release/v1.3.0/inputs/laplace_draw_metrics.csv',
 'release/v1.3.0/inputs/laplace_run_metrics.csv',
 'release/v1.3.0/inputs/primary_seed_results.csv',
 'release/v1.3.0/inputs/semantic_seed_results.csv',
 'release/v1.3.0/primary_figure.py',
 'release/v1.3.0/provenance.json',
 'release/v1.3.0/reproduce.py',
 'release/v1.3.0/requirements.txt',
 'release/v1.3.0/schemas/attacker_restart_rows.schema.json',
 'release/v1.3.0/schemas/attacker_restart_summaries.schema.json',
 'release/v1.3.0/schemas/benchmark_budget_attacker_rows.schema.json',
 'release/v1.3.0/schemas/benchmark_budget_conditions.schema.json',
 'release/v1.3.0/schemas/benchmark_budget_paired_effects.schema.json',
 'release/v1.3.0/schemas/benchmark_v1_tables.schema.json',
 'release/v1.3.0/schemas/eng_compute_v1.schema.json',
 'release/v1.3.0/statistics_core.py')
V13_LICENSE_SCOPE = ('release/v1.3.0/diagnostics/known-offset-diagnostic.csv',
 'release/v1.3.0/diagnostics/known-offset-diagnostic.json',
 'release/v1.3.0/expected/attacker_restart_summaries.csv',
 'release/v1.3.0/expected/benchmark_budget_paired_effects.csv',
 'release/v1.3.0/expected/budget_method_summaries.csv',
 'release/v1.3.0/expected/compute_group_summary.csv',
 'release/v1.3.0/expected/compute_interface_effects.csv',
 'release/v1.3.0/expected/compute_paired_effects.csv',
 'release/v1.3.0/expected/figure_signatures.json',
 'release/v1.3.0/expected/figure_sources.json',
 'release/v1.3.0/expected/laplace-single-release-statistics.csv',
 'release/v1.3.0/expected/supplementary-statistics.csv',
 'release/v1.3.0/inputs/attacker_restart_rows.csv',
 'release/v1.3.0/inputs/benchmark_budget_conditions.csv',
 'release/v1.3.0/inputs/compute_condition_metrics.csv',
 'release/v1.3.0/inputs/laplace_draw_metrics.csv',
 'release/v1.3.0/inputs/laplace_run_metrics.csv',
 'release/v1.3.0/inputs/primary_seed_results.csv',
 'release/v1.3.0/inputs/semantic_seed_results.csv')

RESULT_LICENSE_SCOPE = tuple(
    sorted(
        set(RESULT_SPECS)
        | set(V13_LICENSE_SCOPE)
        | {
            f"{PUBLIC_RELEASE_ROOT}/analysis/compute_analysis.json",
            f"{PUBLIC_RELEASE_ROOT}/analysis/compute_claim_audit.json",
        }
    )
)
RELEASE_DECISION = {
    "status": "authorized_for_repository_commit",
    "staged_on": "2026-09-30",
    "scope": list(RESULT_LICENSE_SCOPE),
    "license": "CC-BY-4.0 for contributor-owned rights only",
    "publication_authorized": True,
    "third_party_permission_claimed": False,
    "third_party_terms_superseded": False,
}
RELEASE_STATE = {
    "version": "1.3.0",
    "path": "release/v1.3.0",
    "status": "authorized_for_repository_commit",
    "release_url": None,
    "release_date": None,
    "doi": None,
}
EXPECTED_PUBLIC_OUTPUTS = frozenset(
    set(BASE_RESULT_SPECS)
    | {
        "config/study_contract.json",
        "config/implementation_protocol_v2.json",
        "results/protocol/environment.json",
        "results/protocol/frozen_selection_lock.json",
    }
    | set(PUBLIC_RELEASE_FILES)
    | set(V13_FILES)
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def public_allowlist() -> list[str]:
    return sorted(
        set(ROOT_METADATA)
        | set(PUBLIC_SOURCE_MODULES)
        | set(PUBLIC_SCRIPTS)
        | set(PUBLIC_TESTS)
        | set(PUBLIC_WORKFLOWS)
        | set(RESULT_SPECS)
        | set(PROTOCOL_OUTPUTS)
        | set(PUBLIC_RELEASE_FILES)
        | set(V13_FILES)
    )


def _strict_csv(path: Path, fields: tuple[str, ...], expected_rows: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != fields:
            raise RuntimeError(f"Unexpected CSV schema for {path}: {reader.fieldnames}")
        rows = list(reader)
    if len(rows) != expected_rows:
        raise RuntimeError(f"Unexpected row count for {path}: {len(rows)} != {expected_rows}")
    if any(set(row) != set(fields) or None in row.values() for row in rows):
        raise RuntimeError(f"Malformed CSV row in {path}")
    return rows


def _finite(value: Any, field: str) -> None:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"Expected numeric {field}: {value!r}") from error
    if not math.isfinite(numeric):
        raise RuntimeError(f"Non-finite {field}: {value!r}")


def _render_csv(rows: Iterable[dict[str, Any]], fields: tuple[str, ...]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row[field] for field in fields})
    return stream.getvalue().encode("utf-8")


def _validate_result_rows(path: str, rows: list[dict[str, Any]]) -> None:
    forbidden_fields = {"attribute_name", "example_ids", "per_image", "path", "predictions"}
    for row in rows:
        if forbidden_fields & set(row):
            raise RuntimeError(f"Forbidden disclosure field in {path}: {sorted(forbidden_fields & set(row))}")
        for field, value in row.items():
            if isinstance(value, str) and (value.startswith("/") or "/home/" in value):
                raise RuntimeError(f"Absolute/local path in {path}:{field}")

    seeds = ("7", "42", "123", "2024", "2025")
    datasets = ("celeba", "cifar10")
    splits = ("early", "current")
    methods = ("standard", "afd", "laplace", "learned")
    budget_methods = ("standard", "afd")
    fractions = ("0.1", "0.5", "1.0")
    attribute_indices = tuple(str(index) for index in range(40) if index != 31)
    attackers = ("deconv_mse", "residual_lpips")

    expected_keys: set[tuple[str, ...]]
    key_fields: tuple[str, ...]
    if path == "results/raw/primary_seed_results.csv":
        key_fields = ("dataset", "split_point", "method", "seed")
        expected_keys = set(product(datasets, splits, methods, seeds))
    elif path == "results/raw/semantic_seed_results.csv":
        key_fields = ("dataset", "split_point", "method", "seed")
        expected_keys = set(product(("celeba",), splits, methods, seeds))
    elif path == "results/raw/semantic_attribute_results.csv":
        key_fields = ("dataset", "split_point", "method", "seed", "attribute_index")
        expected_keys = set(product(("celeba",), splits, methods, seeds, attribute_indices))
    elif path == "results/raw/budget_cell_results.csv":
        key_fields = ("dataset", "split_point", "method", "seed", "auxiliary_fraction")
        expected_keys = set(product(datasets, splits, budget_methods, ("42",), fractions))
    elif path == "results/raw/budget_attacker_results.csv":
        key_fields = (
            "dataset", "split_point", "method", "seed", "auxiliary_fraction", "architecture",
        )
        expected_keys = set(product(datasets, splits, budget_methods, ("42",), fractions, attackers))
    else:
        raise RuntimeError(f"No public matrix contract for {path}")

    actual_keys = [tuple(str(row[field]) for field in key_fields) for row in rows]
    if len(set(actual_keys)) != len(actual_keys):
        raise RuntimeError(f"Duplicate statistical-unit key in {path}")
    if set(actual_keys) != expected_keys:
        missing = sorted(expected_keys - set(actual_keys))[:5]
        unexpected = sorted(set(actual_keys) - expected_keys)[:5]
        raise RuntimeError(
            f"Incomplete public matrix for {path}: missing={missing}, unexpected={unexpected}"
        )

    numeric_fields = {
        "results/raw/primary_seed_results.csv": (
            "seed", "accuracy_percentage_points", "utility_loss_vs_standard_percentage_points",
            "worst_case_ssim",
        ),
        "results/raw/semantic_seed_results.csv": (
            "seed", "macro_auroc", "macro_balanced_accuracy", "attribute_auroc_sample_sd",
            "attribute_balanced_accuracy_sample_sd", "semantic_draw_count",
        ),
        "results/raw/semantic_attribute_results.csv": (
            "seed", "attribute_index", "auroc", "balanced_accuracy",
        ),
        "results/raw/budget_cell_results.csv": (
            "seed", "auxiliary_fraction", "auxiliary_examples",
            "auxiliary_fitting_examples", "auxiliary_validation_examples", "deconv_mse_ssim",
            "residual_lpips_ssim", "worst_case_test_ssim",
        ),
        "results/raw/budget_attacker_results.csv": (
            "seed", "auxiliary_fraction", "auxiliary_examples",
            "auxiliary_fitting_examples", "auxiliary_validation_examples", "selected_epoch",
            "epochs_completed", "best_validation_loss", "mse", "psnr", "ssim", "lpips",
        ),
    }[path]
    for row in rows:
        for field in numeric_fields:
            _finite(row[field], f"{path}:{field}")
        if "record_passed" in row and str(row["record_passed"]).lower() != "true":
            raise RuntimeError(f"Non-passing public row in {path}: {row.get('run_id', '<no-run-id>')}")

    bounded_fields = {
        "accuracy_percentage_points": (0.0, 100.0),
        "worst_case_ssim": (-1.0, 1.0),
        "macro_auroc": (0.0, 1.0),
        "macro_balanced_accuracy": (0.0, 1.0),
        "auroc": (0.0, 1.0),
        "balanced_accuracy": (0.0, 1.0),
        "auxiliary_fraction": (0.0, 1.0),
        "deconv_mse_ssim": (-1.0, 1.0),
        "residual_lpips_ssim": (-1.0, 1.0),
        "worst_case_test_ssim": (-1.0, 1.0),
        "ssim": (-1.0, 1.0),
    }
    for row in rows:
        for field, (lower, upper) in bounded_fields.items():
            if field in row and not lower <= float(row[field]) <= upper:
                raise RuntimeError(f"Out-of-range {path}:{field}={row[field]}")
        if "auxiliary_examples" in row and int(row["auxiliary_examples"]) != (
            int(row["auxiliary_fitting_examples"]) + int(row["auxiliary_validation_examples"])
        ):
            raise RuntimeError(f"Auxiliary split counts do not sum to the total cap in {path}")

    if path in {
        "results/raw/primary_seed_results.csv", "results/raw/semantic_seed_results.csv",
    }:
        for row in rows:
            selected = str(row["selected_defense_value"])
            if row["method"] == "standard":
                if selected != "":
                    raise RuntimeError(f"Standard row has a selected defense value in {path}")
            else:
                _finite(selected, f"{path}:selected_defense_value")
                if float(selected) <= 0:
                    raise RuntimeError(f"Non-positive selected defense value in {path}")
    if path == "results/raw/semantic_seed_results.csv":
        for row in rows:
            draw_seeds = [item for item in str(row["semantic_draw_seeds"]).split(",") if item]
            if len(draw_seeds) != int(row["semantic_draw_count"]):
                raise RuntimeError(f"Semantic draw count mismatch in {path}: {row['run_id']}")


def _sanitized_environment(
    source: dict[str, Any],
    implementation_protocol: dict[str, Any],
) -> dict[str, Any]:
    fields = (
        "captured_on", "python", "packages", "torch_cuda_build", "cuda_available_at_capture",
        "gpu_inventory", "hardware_measurement_role",
    )
    if set(source) != set(fields):
        raise RuntimeError(f"Unexpected environment fields: {sorted(set(source) - set(fields))}")
    sanitized = {field: source[field] for field in fields}
    sanitized["packages"] = dict(sanitized["packages"])
    dependencies = implementation_protocol["dependencies"]
    sanitized["packages"]["Pillow"] = dependencies["Pillow"]["version"]
    sanitized["packages"]["scikit-learn"] = dependencies["scikit-learn"]["version"]
    return sanitized


def _sanitized_lock(source: dict[str, Any]) -> dict[str, Any]:
    if set(source) != {"frozen_artifact", "schema_version", "seal_type", "selection_contract", "study_contract"}:
        raise RuntimeError("Unexpected frozen-selection lock schema")
    frozen = source["frozen_artifact"]
    contract = source["study_contract"]
    return {
        "schema_version": source["schema_version"],
        "seal_type": source["seal_type"],
        "selection_contract": source["selection_contract"],
        "study_contract": {
            "path": "config/study_contract.json",
            "sha256": contract["sha256"],
            "size_bytes": contract["size_bytes"],
        },
        "frozen_artifact": {
            "published": False,
            "sha256": frozen["sha256"],
            "size_bytes": frozen["size_bytes"],
            "required_mode": frozen["required_mode"],
            "exclusion_reason": "Contains validation examples and is larger than GitHub's 100 MB object limit.",
        },
    }


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def render_from_private_sources() -> dict[str, bytes]:
    missing = [str(path) for path in PRIVATE_SOURCES.values() if not path.is_file()]
    if missing:
        raise RuntimeError("Private canonical source is missing: " + ", ".join(missing))

    primary = _strict_csv(PRIVATE_SOURCES["primary_seed_results"], PRIMARY_FIELDS, 80)
    semantic_seed = _strict_csv(PRIVATE_SOURCES["semantic_seed_results"], SEMANTIC_SEED_FIELDS, 40)
    semantic_attribute_source = _strict_csv(
        PRIVATE_SOURCES["semantic_attribute_results"], SEMANTIC_ATTRIBUTE_SOURCE_FIELDS, 1560
    )
    semantic_attribute = [
        {field: row[field] for field in SEMANTIC_ATTRIBUTE_FIELDS}
        for row in semantic_attribute_source
    ]
    budget_audit = json.loads(PRIVATE_SOURCES["budget_result_audit"].read_text(encoding="utf-8"))
    if budget_audit.get("passed") is not True or budget_audit.get("issues") or budget_audit.get("quarantined_run_ids"):
        raise RuntimeError("Budget public export requires a clean passing raw-evidence audit")
    budget_cells = budget_audit.get("budget_cell_results", [])
    budget_attackers = budget_audit.get("budget_attacker_results", [])
    if len(budget_cells) != 24 or len(budget_attackers) != 48:
        raise RuntimeError("Budget public export requires exactly 24 cells and 48 attacker rows")

    for row in primary:
        for field in ("seed", "selected_defense_value", "accuracy_percentage_points", "utility_loss_vs_standard_percentage_points", "worst_case_ssim"):
            if row[field] != "":
                _finite(row[field], field)
    for row in semantic_seed:
        for field in ("seed", "macro_auroc", "macro_balanced_accuracy", "semantic_draw_count"):
            _finite(row[field], field)
    for row in semantic_attribute:
        for field in ("seed", "attribute_index", "auroc", "balanced_accuracy"):
            _finite(row[field], field)
    for row in budget_cells:
        for field in ("seed", "auxiliary_fraction", "auxiliary_examples", "auxiliary_fitting_examples", "auxiliary_validation_examples", "deconv_mse_ssim", "residual_lpips_ssim", "worst_case_test_ssim"):
            _finite(row[field], field)
    for row in budget_attackers:
        for field in ("seed", "auxiliary_fraction", "auxiliary_examples", "auxiliary_fitting_examples", "auxiliary_validation_examples", "selected_epoch", "epochs_completed", "best_validation_loss", "mse", "psnr", "ssim", "lpips"):
            _finite(row[field], field)

    primary.sort(key=lambda row: (row["dataset"], row["split_point"], row["method"], int(row["seed"])))
    semantic_seed.sort(key=lambda row: (row["dataset"], row["split_point"], row["method"], int(row["seed"])))
    semantic_attribute.sort(key=lambda row: (row["dataset"], row["split_point"], row["method"], int(row["seed"]), int(row["attribute_index"])))
    budget_cells.sort(key=lambda row: (row["dataset"], row["split_point"], row["method"], float(row["auxiliary_fraction"])))
    budget_attackers.sort(key=lambda row: (row["dataset"], row["split_point"], row["method"], float(row["auxiliary_fraction"]), row["architecture"]))

    result_rows = {
        "results/raw/primary_seed_results.csv": primary,
        "results/raw/semantic_seed_results.csv": semantic_seed,
        "results/raw/semantic_attribute_results.csv": semantic_attribute,
        "results/raw/budget_cell_results.csv": budget_cells,
        "results/raw/budget_attacker_results.csv": budget_attackers,
    }
    for path, rows in result_rows.items():
        _validate_result_rows(path, rows)

    contract_bytes = PRIVATE_SOURCES["study_contract"].read_bytes()
    implementation_protocol = json.loads(
        PRIVATE_SOURCES["implementation_protocol"].read_text(encoding="utf-8")
    )
    environment = _sanitized_environment(
        json.loads(PRIVATE_SOURCES["environment"].read_text(encoding="utf-8")),
        implementation_protocol,
    )
    selection_lock = _sanitized_lock(
        json.loads(PRIVATE_SOURCES["frozen_selection_lock"].read_text(encoding="utf-8"))
    )
    outputs: dict[str, bytes] = {
        path: _render_csv(rows, BASE_RESULT_SPECS[path][1])
        for path, rows in result_rows.items()
    }
    outputs.update(
        {
            "config/study_contract.json": contract_bytes,
            "config/implementation_protocol_v2.json": _json_bytes(implementation_protocol),
            "results/protocol/environment.json": _json_bytes(environment),
            "results/protocol/frozen_selection_lock.json": _json_bytes(selection_lock),
        }
    )
    release_payloads = build_private_release_payloads()
    outputs.update(
        {
            f"{PUBLIC_RELEASE_ROOT}/{relative}": payload
            for relative, payload in release_payloads.items()
        }
    )

    for relative in V13_FILES:
        payload = (PROJECT_ROOT / relative).read_bytes()
        if relative in outputs and outputs[relative] != payload:
            raise RuntimeError(f"Scientific bytes differ from private regeneration: {relative}")
        outputs[relative] = payload

    source_inputs = {
        role: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for role, path in sorted(PRIVATE_SOURCES.items())
    }
    public_outputs: dict[str, dict[str, Any]] = {}
    for path, payload in sorted(outputs.items()):
        row: dict[str, Any] = {"sha256": sha256_bytes(payload), "size_bytes": len(payload)}
        if path in RESULT_SPECS:
            row.update({"row_count": RESULT_SPECS[path][0], "fields": list(RESULT_SPECS[path][1])})
        public_outputs[path] = row
    manifest = {
        "schema_version": 3,
        "repository_name": "adaptive-reconstruction-split-computing",
        "repository_description": "Companion code and aggregate evidence for the Eng reconstruction-utility-compute study in split computing.",
        "disclosure_date": "2026-09-30",
        "result_row_count": sum(spec[0] for spec in RESULT_SPECS.values()),
        "base_result_row_count": sum(spec[0] for spec in BASE_RESULT_SPECS.values()),
        "supplemental_result_row_count": sum(
            spec[0] for spec in RELEASE_RESULT_SPECS.values()
        ),
        "result_table_counts": {path: spec[0] for path, spec in sorted(RESULT_SPECS.items())},
        "semantic_attribute_boundary": {
            "published_indices": [index for index in range(40) if index != 31],
            "excluded_index": 31,
            "reason": "The supervised task-target column is excluded; indices are schema references, not anonymized labels.",
        },
        "source_inputs": source_inputs,
        "public_outputs": public_outputs,
        "public_allowlist": public_allowlist(),
        "release": RELEASE_STATE,
        "excluded": [
            "datasets and third-party labels",
            "images and sample tensors",
            "checkpoints and model weights",
            "per-image metrics and example identifiers",
            "semantic prediction arrays",
            "the 240 MB frozen-selection artifact",
            "manuscript and submission materials",
            "private audit and analysis evidence",
        ],
        "licenses": {
            "authored_code_schemas_and_documentation": "MIT",
            "contributor_owned_numeric_results_and_compute_analysis": "CC-BY-4.0",
            "third_party_data_and_assets": "not licensed or redistributed",
        },
        "release_decision": RELEASE_DECISION,
    }
    outputs["results/protocol/disclosure_manifest.json"] = _json_bytes(manifest)
    return outputs


def _read_public_csv(path: Path, fields: tuple[str, ...], expected_rows: int) -> list[dict[str, str]]:
    rows = _strict_csv(path, fields, expected_rows)
    _validate_result_rows(path.relative_to(PROJECT_ROOT).as_posix(), rows)
    return rows


def verify_public_outputs() -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        raise RuntimeError(f"Missing disclosure manifest: {MANIFEST_PATH}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 3:
        raise RuntimeError("Public disclosure manifest is not schema version 3")
    if manifest.get("result_row_count") != 2308:
        raise RuntimeError("Public disclosure manifest does not declare exactly 2,308 result rows")
    if manifest.get("base_result_row_count") != 1752:
        raise RuntimeError("Public disclosure manifest base row count differs from 1,752")
    if manifest.get("supplemental_result_row_count") != 556:
        raise RuntimeError("Public disclosure manifest supplemental row count differs from 556")
    if manifest.get("public_allowlist") != public_allowlist():
        raise RuntimeError("Public allowlist differs from the declared repository boundary")
    if manifest.get("release_decision") != RELEASE_DECISION:
        raise RuntimeError("Public release decision differs from the authorized repository-commit boundary")
    if manifest.get("release") != RELEASE_STATE:
        raise RuntimeError("Public v1.3.0 release state differs from the authorized repository-commit boundary")
    if "historical_protected_updates" in manifest:
        raise RuntimeError("Public disclosure manifest cannot authorize protected-file updates")
    if manifest.get("result_table_counts") != {
        path: spec[0] for path, spec in sorted(RESULT_SPECS.items())
    }:
        raise RuntimeError("Public disclosure table counts differ from the 13-table policy")
    if set(manifest.get("public_outputs", {})) != EXPECTED_PUBLIC_OUTPUTS:
        raise RuntimeError("Public disclosure output set differs from the fixed output policy")
    if manifest.get("licenses") != {
        "authored_code_schemas_and_documentation": "MIT",
        "contributor_owned_numeric_results_and_compute_analysis": "CC-BY-4.0",
        "third_party_data_and_assets": "not licensed or redistributed",
    }:
        raise RuntimeError("Public disclosure license boundary differs from policy")
    verified_rows = 0
    for relative, (count, fields) in BASE_RESULT_SPECS.items():
        rows = _read_public_csv(PROJECT_ROOT / relative, fields, count)
        verified_rows += len(rows)
    from scripts.audit_public_repository import V13_CHECKSUMS
    for relative, digest in V13_CHECKSUMS.items():
        if sha256_file(PROJECT_ROOT / relative) != digest:
            raise RuntimeError(f"v1.3.0 payload checksum differs: {relative}")
    release_report = validate_public_release(PROJECT_ROOT / PUBLIC_RELEASE_ROOT)
    verified_rows += int(release_report["aggregate_rows"])
    for relative, record in manifest.get("public_outputs", {}).items():
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"Missing declared public output: {relative}")
        if sha256_file(path) != record.get("sha256") or path.stat().st_size != record.get("size_bytes"):
            raise RuntimeError(f"Public output hash/size mismatch: {relative}")
    contract = PROJECT_ROOT / "config" / "study_contract.json"
    lock = json.loads((PROTOCOL_ROOT / "frozen_selection_lock.json").read_text(encoding="utf-8"))
    if sha256_file(contract) != lock["study_contract"]["sha256"]:
        raise RuntimeError("Public contract does not match the frozen-selection lock")
    if lock["frozen_artifact"].get("published") is not False:
        raise RuntimeError("Frozen-selection artifact must remain excluded")
    return {
        "passed": True,
        "result_rows_verified": verified_rows,
        "public_outputs_verified": len(manifest["public_outputs"]),
        "allowlist_files": len(manifest["public_allowlist"]),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
    }


def _atomic_create(path: Path, payload: bytes, *, allow_replace: bool = False) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_file() and path.read_bytes() == payload:
            return "verified_existing"
        if not allow_replace:
            raise RuntimeError(f"Refusing to overwrite byte-different public output: {path}")
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    try:
        if path.exists():
            os.replace(temporary, path)
            return "replaced_superseded_v1"
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return "created"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify without writing")
    parser.add_argument(
        "--promote-corrected",
        action="store_true",
        help="Replace only rendered public outputs after the corrected v2 audit passes.",
    )
    parser.add_argument(
        "--public-only",
        action="store_true",
        help="Verify published hashes/schemas when private canonical inputs are unavailable",
    )
    args = parser.parse_args()
    if args.public_only and not args.check:
        parser.error("--public-only requires --check")
    if args.check:
        public_result = verify_public_outputs()
        private_reproduction = not args.public_only
        if private_reproduction:
            expected = render_from_private_sources()
            differences = [
                relative
                for relative, payload in expected.items()
                if not (PROJECT_ROOT / relative).is_file()
                or (PROJECT_ROOT / relative).read_bytes() != payload
            ]
            if differences:
                raise RuntimeError("Public outputs are not byte-identical to private-source regeneration: " + ", ".join(differences))
        print(json.dumps({**public_result, "private_source_regeneration_verified": private_reproduction}, indent=2, sort_keys=True))
        return
    outputs = render_from_private_sources()
    states = {
        relative: _atomic_create(
            PROJECT_ROOT / relative,
            payload,
            allow_replace=(
                args.promote_corrected
                and relative in CORRECTED_PROMOTION_REPLACEABLE_OUTPUTS
            ),
        )
        for relative, payload in outputs.items()
    }
    result = verify_public_outputs()
    print(json.dumps({**result, "output_states": states}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
